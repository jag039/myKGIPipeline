#!/usr/bin/env python3
"""
process_tps.py (no‐args version, Semantic TPS for a single pair)

Assumes this folder structure:

  ./input/
      image.jpg                   # Person image (BGR)
      cloth.jpg                   # Garment image (BGR)
      ag_mask.png                 # Agnostic‐parse mask (grayscale 0..255)
      skin_mask.png               # Skin‐only mask (grayscale 0..255)
      label.json                  # Maps cloth‐basename → 0 or 1 (“long” vs “vest”)
                                  #   (e.g. { "cloth": 1, "shirtA": 0, … })
      cloth.json                  # Raw garment landmarks JSON with keys "long"/"vest"

  ./output/
      predicted_parse.png         # Palette‐PNG from ParseGenerator (each pixel’s palette index ∈ 0..12)
      predicted_cloth.json        # KG‐predicted garment keypoints JSON: {"keypoints":[[x,y],…]}

  ./checkpoints_pretrained/tps/
      step_9999.pt                # Pretrained TPS checkpoint

  ./output/                       # (will be created if it does not exist)
      repainted_image.png         # TPS‐repainted person+cloth BGR output
      repainted_mask.png          # TPS‐composite mask output

On each run, this script will:
  1) Verify that all input files exist.
  2) Load the palette‐encoded parse (predicted_parse.png) in “P” mode so that
     each pixel’s value is 0..12 (class index). Any index ≥ 13 is clamped to 0.
  3) Load “label.json” and pick which 32 cloth landmarks (“long” or “vest”).
  4) Load KG‐predicted cloth keypoints (predicted_cloth.json).
  5) Run semantic TPS to warp cloth onto the person.
  6) Save “repainted_image.png” (BGR) and “repainted_mask.png” (grayscale) under ./output/.

If you need to change filenames or paths, edit the constants under “CONFIGURE PATHS” below.
"""

from __future__ import print_function, absolute_import, division
import os
import cv2
import json
import numpy as np
import torch
import pyclipper

from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

# ----------------------------------------
# === CONFIGURE PATHS & PARAMETERS ===
# ----------------------------------------

INPUT_DIR            = "./input"
OUT_DIR              = "./output"

IMAGE_PNG            = os.path.join(INPUT_DIR, "image.jpg")
CLOTH_PNG            = os.path.join(INPUT_DIR, "cloth.jpg")
AG_MASK_PNG          = os.path.join(INPUT_DIR, "ag_mask.png")
SKIN_MASK_PNG        = os.path.join(INPUT_DIR, "skin_mask.png")
LABEL_JSON           = os.path.join(INPUT_DIR, "label.json")
CLOTH_LANDMARK_JSON  = os.path.join(INPUT_DIR, "cloth.json")

PARSE_PNG            = os.path.join(OUT_DIR, "predicted_parse.png")
PRED_CLOTH_JSON      = os.path.join(OUT_DIR, "predicted_cloth.json")

OUT_IMAGE_PNG        = os.path.join(OUT_DIR, "repainted_image.png")
OUT_MASK_PNG         = os.path.join(OUT_DIR, "repainted_mask.png")

TPS_CHECKPOINT       = "./checkpoints_pretrained/sci/ckpt_1024/ema_0.9999_300000.pt"

# Dimensions must match dataset_tps.py (fine_width=192*4=768, fine_height=256*4=1024)
FINE_WIDTH           = 192 * 4    # = 768
FINE_HEIGHT          = 256 * 4    # = 1024
MARGIN               = -5         # Shrink/grow margin for part masks

NUM_CLASSES          = 13         # number of parse channels

# ----------------------------------------
# === HELPER CLASSES & FUNCTIONS ===
# ----------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TPS(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, Y, w, h):
        """
        X: (1, k, 2) source control points ∈ [-1..1]
        Y: (1, k, 2) target control points ∈ [-1..1]
        w, h: output width and height
        Returns: warped grid (1, h, w, 2)
        """
        # Build normalized grid over [−1,1] × [−1,1]
        grid = torch.ones(1, h, w, 2, device=device)
        grid[..., 0] = torch.linspace(-1, 1, w, device=device)
        grid[..., 1] = torch.linspace(-1, 1, h, device=device)[..., None]
        grid_flat = grid.view(1, h * w, 2)  # shape (1, h*w, 2)

        n, k = X.shape[:2]  # n=1, k=num_control_points
        # Build L matrix
        Z = torch.zeros(1, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device)

        eps = 1e-9
        # pairwise squared distances among source points
        D2_src = torch.sum((X[:, :, None, :] - X[:, None, :, :]) ** 2, dim=-1)  # (1, k, k)
        K = D2_src * torch.log(D2_src + eps)                                    # (1, k, k)

        P[:, :, 1:] = X  # fill [x, y]
        Z[:, :k, :] = Y  # fill target coordinates

        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        # Solve for W (k×2) and A (3×2)
        Q = torch.linalg.solve(L, Z)
        W, A = Q[:, :k, :], Q[:, k:, :]

        # Build U for each grid point wrt each source
        D2_grid = torch.sum((grid_flat[:, :, None, :] - X[:, None, :, :]) ** 2, dim=-1)  # (1, h*w, k)
        U = D2_grid * torch.log(D2_grid + eps)                                          # (1, h*w, k)

        # Build P for grid
        n_grid = grid_flat.shape[1]
        P_grid = torch.ones(1, n_grid, 3, device=device)
        P_grid[:, :, 1:] = grid_flat  # shape (1, h*w, 3)

        # Compute warped grid: P_grid @ A + U @ W
        warped_flat = torch.matmul(P_grid, A) + torch.matmul(U, W)  # shape (1, h*w, 2)
        warped = warped_flat.view(1, h, w, 2)                        # shape (1, h, w, 2)
        return warped


def dedup(source_pts, target_pts, source_center, target_center):
    """
    Remove duplicate source↔target pairs. If only 2 unique remain, add the center point.
    """
    old_src = source_pts.tolist()
    old_tgt = target_pts.tolist()
    idx_list, new_src, new_tgt = [], [], []
    for i in range(len(old_src)):
        if old_src[i] not in new_src and old_tgt[i] not in new_tgt:
            new_src.append(old_src[i])
            new_tgt.append(old_tgt[i])
            idx_list.append(i)

    if len(idx_list) == 2:
        new_src_pts = torch.cat([source_pts[idx_list], source_center], dim=0)[None, ...]
        new_tgt_pts = torch.cat([target_pts[idx_list], target_center], dim=0)[None, ...]
    elif len(idx_list) > 2:
        new_src_pts = source_pts[idx_list][None, ...]
        new_tgt_pts = target_pts[idx_list][None, ...]
    else:
        raise RuntimeError("Less than 2 valid keypoints for TPS!")

    return new_src_pts, new_tgt_pts


def equidistant_zoom_contour(contour, margin):
    """
    Expand/contract a polygon by `margin` pixels using pyclipper.
    contour: (N,2) int numpy array
    Returns: (M,2) int numpy array of offset polygon
    """
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(contour.astype(int).tolist(), pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    solution = pco.Execute(margin)
    if len(solution) == 0:
        return np.zeros((3, 2), dtype=int)
    return np.array(solution[0], dtype=int)


def remove_background(args, s_mask, warped_bgr):
    """
    Set s_mask pixels to 0 wherever warped_bgr is near‐white (all channels > 240).
    """
    r_mask = s_mask.copy()
    h, w = args["fine_height"], args["fine_width"]
    for i in range(h):
        for j in range(w):
            b, g, r = int(warped_bgr[i, j, 0]), int(warped_bgr[i, j, 1]), int(warped_bgr[i, j, 2])
            if (b > 240 and g > 240 and r > 240):
                r_mask[i, j] = 0
    return r_mask


def draw_part(args, group_id, ten_source, ten_target, ten_source_center, ten_target_center, ten_img):
    """
    Warp one semantic “part” of the cloth and return:
      • out_img_bgr:  (H,W,3) uint8 BGR numpy
      • l_mask:       (H,W) uint8 polygon mask of original part
      • s_mask:       (H,W) uint8 shrunk/expanded polygon
      • r_mask:       (H,W) uint8 final mask after removing background
    """
    ten_img = ten_img.to(device)  # (3,H,W) float Tensor

    # Extract coordinates for this part
    src_pts = ten_source[group_id]   # (num_pts, 2)
    tgt_pts = ten_target[group_id]   # (num_pts, 2)

    # Build pixel‐space polygon for target points:
    poly = tgt_pts.clone().cpu().numpy()
    poly[:, 0] = ((poly[:, 0] * 0.5) + 0.5) * args["fine_width"]
    poly[:, 1] = ((poly[:, 1] * 0.5) + 0.5) * args["fine_height"]
    poly = poly.astype(int)

    # Expand/contract
    new_poly = equidistant_zoom_contour(poly, args["margin"])

    # Create l_mask and s_mask
    l_mask = np.zeros((args["fine_height"], args["fine_width"]), dtype=np.uint8)
    s_mask = np.zeros((args["fine_height"], args["fine_width"]), dtype=np.uint8)
    cv2.fillPoly(l_mask, [poly], 255)
    cv2.fillPoly(s_mask, [new_poly], 255)

    # Deduplicate points and ensure ≥3 for TPS
    ten_src_pts, ten_tgt_pts = dedup(src_pts, tgt_pts, ten_source_center, ten_target_center)

    # Compute TPS warp
    tps = TPS().to(device)
    warped_grid = tps(ten_tgt_pts, ten_src_pts, args["fine_width"], args["fine_height"])  # (1,H,W,2)

    # Sample cloth image using grid_sample
    ten_img_batch = ten_img.unsqueeze(0)  # (1,3,H,W)
    warped = torch.nn.functional.grid_sample(
        ten_img_batch, warped_grid, mode='bilinear', padding_mode='zeros', align_corners=False
    )  # (1,3,H,W)
    out_img_rgb = warped[0].cpu()  # (3,H,W) float in [0,1]

    # Convert to BGR uint8
    out_img_pil = transforms.ToPILImage()(out_img_rgb)
    out_img_bgr = cv2.cvtColor(np.array(out_img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    # Remove near‐white background from s_mask
    r_mask = remove_background(args, s_mask, out_img_bgr)

    return out_img_bgr, l_mask, s_mask, r_mask


def paste_cloth(mask, image_bgr, part_bgr, l_mask, r_mask, parse_13):
    """
    Composite one warped part onto the person image:
      • Only paste where parse_13[3] == 1 (torso) so we don’t paint outside the body
      • Erase original pixels under l_mask, then paint part_bgr where r_mask==255
    Returns updated (out_mask, out_image_bgr).
    """
    out_image = image_bgr.copy()
    out_mask = mask.copy()

    # Body (torso) region = parse_13 channel index 3
    body_mask = (parse_13[3].cpu().numpy() > 0).astype(np.uint8)

    # Zero out l_mask/r_mask outside torso
    l_mask[body_mask == 0] = 0
    r_mask[body_mask == 0] = 0

    # Erase where new cloth should go
    out_mask[l_mask == 255] = 0
    out_mask[r_mask == 255] = 255

    # “Black out” torso in original image (where l_mask)
    out_image[l_mask == 255] = 0
    # Paste warped cloth onto region (where r_mask)
    out_image[r_mask == 255] = part_bgr[r_mask == 255]

    return out_mask, out_image


def generate_repaint(args, image_bgr, cloth_bgr, v_pos, e_pos, ag_mask, skin_mask, parse_13):
    """
    Composite skin + cloth parts:
      1) Mask out background of person (ag_mask==0 → black)
      2) Paste skin where parse_13[5] (face), [6] (neck), [11] (arms) == 1
      3) For each semantic cloth group, call draw_part + paste_cloth
    """
    # Start with agnostic mask
    out_mask = ag_mask.copy()
    out_image = image_bgr.copy()
    out_image[ag_mask == 0] = 0

    # Paste skin (face=5, neck=6, arms=11)
    skin_region = ((parse_13[5] + parse_13[6] + parse_13[11]) > 0).cpu().numpy().astype(np.uint8)
    new_skin_mask = skin_mask.copy()
    new_skin_mask[skin_region == 0] = 0
    out_mask[new_skin_mask == 255] = 255
    out_image[new_skin_mask == 255] = image_bgr[new_skin_mask == 255]

    # Semantic groups of keypoint indices:
    group_backbone    = [ 4,  3,  2,  1,  0,  5, 14, 15, 16, 17, 18, 19, 20, 21, 22, 31 ]
    group_left_up     = [ 5,  6,  7, 12, 13, 14 ]
    group_left_low    = [ 7,  8,  9, 10, 11, 12 ]
    group_right_up    = [22, 23, 24, 29, 30, 31 ]
    group_right_low   = [24, 25, 26, 27, 28, 29 ]

    # Convert cloth to Tensor (3,H,W) for sampling
    ten_cloth = ToTensor()(cloth_bgr).to(device)

    # Normalize keypoints from [0..1] → [-1..1]
    ten_source = (v_pos - 0.5) * 2  # (32,2)
    ten_target = (e_pos - 0.5) * 2

    # Compute “center” points for possible duplication
    ten_src_center = (0.5 * (ten_source[18] - ten_source[2]))[None, ...]  # (1,2)
    ten_tgt_center = (0.5 * (ten_target[18] - ten_target[2]))[None, ...]

    # Warp backbone
    im_bck, l_bck, s_bck, r_bck = draw_part(
        args, group_backbone, ten_source, ten_target, ten_src_center, ten_tgt_center, ten_cloth
    )
    # Warp left‐up
    im_lu, l_lu, s_lu, r_lu = draw_part(
        args, group_left_up, ten_source, ten_target, ten_src_center, ten_tgt_center, ten_cloth
    )
    # Warp left‐low
    im_ll, l_ll, s_ll, r_ll = draw_part(
        args, group_left_low, ten_source, ten_target, ten_src_center, ten_tgt_center, ten_cloth
    )
    # Warp right‐up
    im_ru, l_ru, s_ru, r_ru = draw_part(
        args, group_right_up, ten_source, ten_target, ten_src_center, ten_tgt_center, ten_cloth
    )
    # Warp right‐low
    im_rl, l_rl, s_rl, r_rl = draw_part(
        args, group_right_low, ten_source, ten_target, ten_src_center, ten_tgt_center, ten_cloth
    )

    # If backbone warp fails too much, revert r_bck = s_bck
    if (r_bck.sum() / max(1, s_bck.sum())) < 0.9:
        r_bck = s_bck

    # Composite in order
    out_mask, out_image = paste_cloth(out_mask, out_image, im_bck,  l_bck,  r_bck,  parse_13)
    out_mask, out_image = paste_cloth(out_mask, out_image, im_lu,  l_lu,  r_lu,  parse_13)
    out_mask, out_image = paste_cloth(out_mask, out_image, im_ll,  l_ll,  r_ll,  parse_13)
    out_mask, out_image = paste_cloth(out_mask, out_image, im_ru,  l_ru,  r_ru,  parse_13)
    out_mask, out_image = paste_cloth(out_mask, out_image, im_rl,  l_rl,  r_rl,  parse_13)

    return out_image, out_mask


# ----------------------------------------
# === MAIN (no‐args) ===
# ----------------------------------------
def main():
    # 1) Verify inputs exist
    missing = []
    for path in [
        IMAGE_PNG, CLOTH_PNG,
        AG_MASK_PNG, SKIN_MASK_PNG,
        LABEL_JSON, CLOTH_LANDMARK_JSON,
        PARSE_PNG, PRED_CLOTH_JSON
    ]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        raise FileNotFoundError(f"Missing required file(s): {missing}")

    if not os.path.isfile(TPS_CHECKPOINT):
        raise FileNotFoundError(f"Missing TPS checkpoint: {TPS_CHECKPOINT}")

    # 2) Create output directory if needed
    os.makedirs(OUT_DIR, exist_ok=True)

    # 3) Load label map (maps cloth‐basename → 0/1 for “long” vs “vest”)
    label_map = json.load(open(LABEL_JSON, "r"))

    # 4) Derive base names
    #    We look up by cloth‐filename (without extension), not by image name.
    cloth_basename = os.path.splitext(os.path.basename(CLOTH_PNG))[0]  # e.g. “shirtA” → “shirtA”
    cloth_key = label_map.get(cloth_basename, None)
    if cloth_key is None:
        raise KeyError(f"Cloth basename '{cloth_basename}' not found in {LABEL_JSON}")

    # 5) Load person image (BGR) and resize so shorter side = FINE_WIDTH (768),
    #    then convert back to BGR numpy
    image_bgr = cv2.imread(IMAGE_PNG, cv2.IMREAD_COLOR)
    pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    pil_image = transforms.Resize(FINE_WIDTH, interpolation=Image.BILINEAR)(pil_image)
    pil_arr   = np.array(pil_image)
    image_bgr = cv2.cvtColor(pil_arr, cv2.COLOR_RGB2BGR)

    # 6) Load cloth image (BGR) and resize so shorter side = FINE_WIDTH
    cloth_bgr = cv2.imread(CLOTH_PNG, cv2.IMREAD_COLOR)
    pil_cloth = Image.fromarray(cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2RGB))
    pil_cloth = transforms.Resize(FINE_WIDTH, interpolation=Image.BILINEAR)(pil_cloth)
    pil_arr   = np.array(pil_cloth)
    cloth_bgr = cv2.cvtColor(pil_arr, cv2.COLOR_RGB2BGR)

    # 7) Load agnostic mask (grayscale), invert (so 255=foreground), and resize
    ag_raw    = cv2.imread(AG_MASK_PNG, cv2.IMREAD_GRAYSCALE)
    pil_ag    = Image.fromarray(ag_raw)
    pil_ag    = transforms.Resize(FINE_WIDTH, interpolation=Image.NEAREST)(pil_ag)
    ag_np     = np.array(pil_ag, dtype=np.uint8)
    ag_mask   = 255 - ag_np  # invert: 255=foreground

    # 8) Load skin mask (grayscale) and resize
    skin_raw  = cv2.imread(SKIN_MASK_PNG, cv2.IMREAD_GRAYSCALE)
    pil_skin  = Image.fromarray(skin_raw)
    pil_skin  = transforms.Resize(FINE_WIDTH, interpolation=Image.NEAREST)(pil_skin)
    skin_mask = np.array(pil_skin, dtype=np.uint8)

    # 9) Load full parse (palette‐PNG from ParseGenerator) in “P” mode
    #    so that every pixel’s value = class index (0..12). Any index >= 13 is clamped to 0.
    palette_img = Image.open(PARSE_PNG).convert("P")  # "P" forces palette‐index mode
    parse_np    = np.array(palette_img, dtype=np.uint8)  # shape (H', W'), each ∈ [0..255]
    parse_np[parse_np >= NUM_CLASSES] = 0               # clamp any 13,14,… back to 0

    Hp, Wp      = parse_np.shape
    # Build parse_13 one‐hot with padding to (FINE_HEIGHT, FINE_WIDTH)
    parse_tensor = torch.from_numpy(parse_np)[None, :, :].long().to(device)  # (1, H', W')
    parse_13     = torch.zeros((NUM_CLASSES, FINE_HEIGHT, FINE_WIDTH),
                                dtype=torch.float32, device=device)
    # Zero‐padding region is already zero; now scatter
    parse_13.scatter_(0, parse_tensor, 1.0)  # shape now (13, H', W'), padded to (13, FINE_HEIGHT, FINE_WIDTH)

    # 10) Load raw cloth landmarks JSON and select 32 points (“long” vs “vest”)
    raw_cloth = json.load(open(CLOTH_LANDMARK_JSON, "r"))
    if cloth_key == 0:
        pts_arr = np.array(raw_cloth["long"], dtype=np.float32)
        ck_idx  = list(range(1, 33))            # take indices 1..32
        c32     = pts_arr[ck_idx, :].copy()     # shape (32, 2)
    else:
        pts_arr = np.array(raw_cloth["vest"], dtype=np.float32)
        ck_idx  = [1,2,3,4,5,6,6,6,6,6,
                   7,7,7,7,7,7,
                   8, 9,10,11,12,13,13,13,13,13,13,14,14,14,14,14]
        c32     = pts_arr[ck_idx, :].copy()     # shape (32, 2)

    # Normalize c32: x/3, y/4 (so they match the dataset convention)
    c32[:, 0] /= 3.0
    c32[:, 1] /= 4.0
    v_pos = torch.from_numpy(c32).to(device)   # (32, 2), normalized

    # 11) Load KG‐predicted cloth keypoints JSON
    pred_ck = json.load(open(PRED_CLOTH_JSON, "r"))
    e_arr   = np.array(pred_ck["keypoints"], dtype=np.float32)  # (32, 2), normalized ∈ [0..1]
    e_pos   = torch.from_numpy(e_arr).to(device)

    # 12) Build argument dictionary for helper functions
    args = {
        "fine_width":  FINE_WIDTH,
        "fine_height": FINE_HEIGHT,
        "margin":      MARGIN
    }

    # 13) Run TPS repainted composite
    out_image_bgr, out_mask = generate_repaint(
        args,
        image_bgr, cloth_bgr,
        v_pos, e_pos,
        ag_mask, skin_mask,
        parse_13
    )

    # 14) Save outputs
    cv2.imwrite(OUT_IMAGE_PNG, out_image_bgr)
    cv2.imwrite(OUT_MASK_PNG,  out_mask)

    print("TPS repaint completed successfully.")
    print(f"  • Input person image       → {IMAGE_PNG}")
    print(f"  • Input cloth image        → {CLOTH_PNG}")
    print(f"  • Input agnostic mask      → {AG_MASK_PNG}")
    print(f"  • Input skin mask          → {SKIN_MASK_PNG}")
    print(f"  • Input label.json         → {LABEL_JSON}")
    print(f"  • Input cloth landmarks    → {CLOTH_LANDMARK_JSON}")
    print(f"  • Input palette‐parse PNG  → {PARSE_PNG}")
    print(f"  • Input KG keypoints JSON  → {PRED_CLOTH_JSON}")
    print(f"  • Output repaint image     → {OUT_IMAGE_PNG}")
    print(f"  • Output composite mask    → {OUT_MASK_PNG}")


if __name__ == "__main__":
    main()
