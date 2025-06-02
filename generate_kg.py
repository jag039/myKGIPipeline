#!/usr/bin/env python3
"""
process_frame.py (no-args version, using skeleton.json + cloth.json + label.json)

Folder structure assumed:

  ./input/
      skeleton.json       # OpenPose output for one frame
                          #   { "people":[ { "pose_keypoints_2d": [x1,y1,c1, x2,y2,c2, … x25,y25,c25] } ] }
      cloth.json          # Garment JSON with exactly one of "long" or "vest"
                          #   { "long": [ [x0,y0], … ],   // OR   "vest": [ [x0,y0], … ] }
      label.json          # Maps cloth basename (without .json) → 0 or 1

  ./checkpoints_pretrained/kg/
      step_299999.pt      # Pretrained GCN_2 checkpoint

On each run, this will:
  1) Load ./input/skeleton.json, extract 10 skeleton joints, normalize to [0,1].
  2) Load ./input/cloth.json, check label.json to pick "long" vs "vest", take 32 points,
     normalize (/3, /4) and center around mid-hip.
  3) Run GCN_2 inference to predict 32 garment keypoints (normalized).
  4) Denormalize predictions to pixel coords (768×1024).
  5) Save predicted pixel‐coords → ./output/predicted_cloth.json
  6) Render skeleton joints → ./output/skeleton_vis.png
  7) Render predicted garment joints → ./output/cloth_vis.png

If you need to change any file/folder names or paths, edit the variables under “CONFIGURE PATHS.”
"""

from __future__ import print_function, absolute_import, division
import os
import json

import numpy as np
import torch
import cv2
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

# ---------------------------------------------------
# === CONFIGURE PATHS BELOW ===
# ---------------------------------------------------

# Input file paths (must exist before you run):
OPENPOSE_JSON = "./input/skeleton.json"    # OpenPose JSON (25×3 flat array under people[0].pose_keypoints_2d)
CLOTH_JSON    = "./input/cloth.json"       # Cloth JSON (must contain exactly one key: "long" or "vest")
LABEL_JSON    = "./input/label.json"       # Label map: { "cloth": 0 } or { "cloth": 1 }

# Pretrained KG checkpoint (must exist):
KG_CHECKPOINT = "./checkpoints_pretrained/kg/step_299999.pt"

# Output folder & filenames (will be created/overwritten):
OUT_DIR            = "./output"
OUT_KEYPOINTS_JSON = os.path.join(OUT_DIR, "predicted_cloth.json")
OUT_SKEL_VIS_PNG   = os.path.join(OUT_DIR, "skeleton_vis.png")
OUT_CLOTH_VIS_PNG  = os.path.join(OUT_DIR, "cloth_vis.png")

# The image dimensions used to denormalize/visualize (must match KG training):
IMG_WIDTH  = 768    # width used during KG training
IMG_HEIGHT = 1024   # height used during KG training

# ---------------------------------------------------
# === END CONFIGURATION ===
# ---------------------------------------------------

# Import model‐building utilities from your “kg” package
from kg.semgcn import GCN_2
from kg.graph_utils import adj_mx_from_edges

# -----------------------------
# 1) Utilities for drawing
# -----------------------------
def draw_skeleton(sk_pos: torch.Tensor) -> torch.Tensor:
    """
    Render 10 skeleton joints (normalized [0,1]) on a blank IMG_HEIGHT×IMG_WIDTH canvas.
    Returns a torch.Tensor of shape (3×IMG_HEIGHT×IMG_WIDTH) with RGB skeleton overlay.
    """
    sk = sk_pos.clone().cpu().numpy()
    sk[:, 0] = sk[:, 0] * IMG_WIDTH
    sk[:, 1] = sk[:, 1] * IMG_HEIGHT

    # Indices of the 10 skeleton keypoints and the edges between them
    sk_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sk_Seq = [
        [0,1], [1,8], [1,9], [1,2], [2,3], [3,4],
        [1,5], [5,6], [6,7]
    ]
    stickwidth = 10

    # BGR colors for joints and sticks
    jk_colors = [
        [255, 85,   0], [0, 255, 255], [255, 170, 0], [255, 255,  0],
        [255, 255,  0], [255, 170, 0], [ 85, 255,  0], [ 85, 255,  0],
        [  0, 255, 255], [  0, 255, 255]
    ]
    sk_colors = [
        [255,  85,   0], [  0, 255, 255], [  0, 255, 255], [255, 170,  0],
        [255, 255,  0], [255, 255,  0], [255, 170,  0], [ 85, 255,  0],
        [ 85, 255,  0]
    ]

    canvas = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

    # Draw circles at each of the 10 joints
    for i in range(len(sk_idx)):
        x = int(sk[sk_idx[i], 0])
        y = int(sk[sk_idx[i], 1])
        cv2.circle(canvas, (x, y), stickwidth, jk_colors[i], thickness=-1)

    # Draw connecting “sticks” (lines) between the pairs
    for i in range(len(sk_Seq)):
        idx_pair = np.array(sk_Seq[i])
        Y = [ sk[idx_pair[0], 0], sk[idx_pair[1], 0] ]
        X = [ sk[idx_pair[0], 1], sk[idx_pair[1], 1] ]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = np.sqrt((X[0]-X[1])**2 + (Y[0]-Y[1])**2)
        angle  = np.degrees(np.arctan2(X[0]-X[1], Y[0]-Y[1]))
        polygon = cv2.ellipse2Poly(
            (int(mY), int(mX)),
            (int(length/2), stickwidth),
            int(angle), 0, 360, 1
        )
        cur_canvas = canvas.copy()
        cv2.fillConvexPoly(cur_canvas, polygon, sk_colors[i])
        canvas = cv2.addWeighted(canvas, 0, cur_canvas, 1, 0)

    # Convert BGR → RGB, then to torch.Tensor (3×H×W) normalized [0,1]
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()
    return transform(canvas_rgb)


def draw_cloth(ck_pos: torch.Tensor) -> torch.Tensor:
    """
    Render 32 cloth landmarks (normalized [0,1]) on a blank IMG_HEIGHT×IMG_WIDTH canvas.
    Returns a torch.Tensor (3×IMG_HEIGHT×IMG_WIDTH) with the RGB cloth overlay.
    """
    ck = ck_pos.clone().cpu().numpy()
    ck[:, 0] = ck[:, 0] * IMG_WIDTH
    ck[:, 1] = ck[:, 1] * IMG_HEIGHT

    # Indices & edges for 32 cloth landmarks
    ck_idx = list(range(32))
    ck_Seq = [
        [0,1], [1,2], [2,3], [3,4],
        [4,31], [31,30], [30,29], [29,28],
        [28,27], [27,26], [26,25], [25,24],
        [24,23], [23,22], [22,21], [21,20],
        [20,19], [19,18], [18,17], [17,16],
        [16,15], [15,14], [14,13], [13,12],
        [12,11], [11,10], [10,9],  [9,8],
        [8,7],   [7,6],   [6,5],   [5,0]
    ]
    stickwidth = 10
    ck_color = [255, 0, 0]  # BGR red

    canvas = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

    # Draw 32 circles
    for i in ck_idx:
        x = int(ck[i, 0])
        y = int(ck[i, 1])
        cv2.circle(canvas, (x, y), stickwidth, ck_color, thickness=-1)

    # Draw connecting lines
    for i in range(len(ck_Seq)):
        idx_pair = np.array(ck_Seq[i])
        Y = [ ck[idx_pair[0], 0], ck[idx_pair[1], 0] ]
        X = [ ck[idx_pair[0], 1], ck[idx_pair[1], 1] ]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = np.sqrt((X[0]-X[1])**2 + (Y[0]-Y[1])**2)
        angle  = np.degrees(np.arctan2(X[0]-X[1], Y[0]-Y[1]))
        polygon = cv2.ellipse2Poly(
            (int(mY), int(mX)),
            (int(length/2), stickwidth),
            int(angle), 0, 360, 1
        )
        cur_canvas = canvas.copy()
        cv2.fillConvexPoly(cur_canvas, polygon, ck_color)
        canvas = cv2.addWeighted(canvas, 0, cur_canvas, 1, 0)

    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()
    return transform(canvas_rgb)


# -----------------------------------------
# 2) Main processing (no argparse – fixed paths)
# -----------------------------------------
def main():
    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #
    # 2.1) Load and parse person skeleton JSON (OpenPose output)
    #
    if not os.path.isfile(OPENPOSE_JSON):
        raise FileNotFoundError(f"Missing OpenPose JSON: {OPENPOSE_JSON}")
    with open(OPENPOSE_JSON, "r") as f:
        sk_json = json.load(f)

    # Expect format: { "people": [ { "pose_keypoints_2d": [x1,y1,c1, x2,y2,c2, …] } ] }
    if "people" not in sk_json or len(sk_json["people"]) == 0:
        raise RuntimeError(f"No detected person in {OPENPOSE_JSON}")
    flat_pts = sk_json["people"][0]["pose_keypoints_2d"]
    flat_arr = np.array(flat_pts, dtype=np.float32).reshape(25, 3)

    # Select exactly 10 joints by index:
    sk_idx = [0, 1, 2, 3, 4, 5, 6, 7, 9, 12]
    selected = flat_arr[sk_idx, 0:2]  # shape: (10,2)

    # Normalize by image width/height
    selected[:, 0] /= float(IMG_WIDTH)
    selected[:, 1] /= float(IMG_HEIGHT)

    # Fill missing points (both x and y == 0) by copying “neck” or previous joint
    for l in range(len(selected)):  # l = 0..9
        if selected[l, 0] == 0 and selected[l, 1] == 0:
            if l in [0, 2, 5, 8, 9]:
                selected[l, :] = selected[1, :]    # copy “neck” at index 1
            else:
                selected[l, :] = selected[l - 1, :]  # copy previous joint

    s_pos = torch.from_numpy(selected).to(device)  # (10,2) normalized

    #
    # 2.2) Load and parse cloth‐landmark JSON (use label.json to pick "long" vs "vest")
    #
    if not os.path.isfile(CLOTH_JSON):
        raise FileNotFoundError(f"Missing cloth JSON: {CLOTH_JSON}")
    with open(CLOTH_JSON, "r") as f:
        c_json = json.load(f)

    # Load label mapping (dict: cloth basename → 0 or 1)
    if not os.path.isfile(LABEL_JSON):
        raise FileNotFoundError(f"Missing label JSON: {LABEL_JSON}")
    with open(LABEL_JSON, "r") as f:
        label_map = json.load(f)

    # Determine cloth basename (strip path & extension)
    cloth_base = os.path.splitext(os.path.basename(CLOTH_JSON))[0]  # e.g. "cloth"
    if cloth_base not in label_map:
        raise KeyError(f"Cloth base '{cloth_base}' not found in {LABEL_JSON}")
    cloth_type = label_map[cloth_base]  # 0 = long, 1 = vest

    # Extract raw landmarks list from either "long" or "vest"
    if cloth_type == 0:
        raw_landmarks = c_json.get("long", None)
        key_name = "long"
    else:
        raw_landmarks = c_json.get("vest", None)
        key_name = "vest"
    if raw_landmarks is None:
        raise KeyError(f"Key '{key_name}' missing in {CLOTH_JSON}")

    arr = np.array(raw_landmarks, dtype=np.float32)  # shape: (N,2), N ≥ 32
    if arr.shape[0] < 32 or arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected ≥32 [x,y] pairs under '{key_name}' in {CLOTH_JSON}, got shape {arr.shape}")
    c32 = arr[0:32, :].copy()  # take exactly first 32 points

    # Normalize c32 by dividing x/3 and y/4 (same as original KGI)
    c32[:, 0] /= 3.0
    c32[:, 1] /= 4.0

    # Re-center around mid-hip (average of points 2 and 18, zero-based)
    c_w = 0.5 * (c32[2, 0] + c32[18, 0])
    c_h = 0.5 * (c32[2, 1] + c32[18, 1])
    c32[:, 0] -= c_w
    c32[:, 1] -= c_h

    c_pos = torch.from_numpy(c32).to(device)  # (32,2) normalized

    #
    # 2.3) Build adjacency matrices (identical to original KG code)
    #
    num_pts_c = 32
    num_pts_s = 10

    contour_edges = [
        [0, 1],  [1, 2],  [2, 3],  [3, 4],  [4, 31], [31, 30], [30, 29],
        [29, 28], [28, 27], [27, 26], [26, 25], [25, 24], [24, 23],
        [23, 22], [22, 21], [21, 20], [20, 19], [19, 18], [18, 17],
        [17, 16], [16, 15], [15, 14], [14, 13], [13, 12], [12, 11],
        [11, 10], [10, 9],  [9, 8],   [8, 7],   [7, 6],   [6, 5],
        [5, 0]
    ]
    symmetry_edges = [
        [0, 4], [1, 3], [5, 31], [14, 22], [15, 21], [16, 20], [17, 19],
        [6, 13], [7, 12], [8, 11], [23, 30], [24, 29], [25, 28], [2, 18]
    ]
    edges_c = contour_edges + symmetry_edges

    edges_s = [
        [0, 1], [1, 2], [1, 5], [2, 3], [5, 6],
        [3, 4], [6, 7], [1, 8], [1, 9]
    ]

    adj_c = adj_mx_from_edges(num_pts_c, edges_c, False).to(device)  # (32×32)
    adj_s = adj_mx_from_edges(num_pts_s, edges_s, False).to(device)  # (10×10)

    #
    # 2.4) Load the pretrained GCN_2 network
    #
    if not os.path.isfile(KG_CHECKPOINT):
        raise FileNotFoundError(f"Missing KG checkpoint: {KG_CHECKPOINT}")
    net = GCN_2(adj_c, adj_s, hid_dim=160).to(device)
    ckpt = torch.load(KG_CHECKPOINT, map_location=device)
    net.load_state_dict(ckpt)
    net.eval()

    #
    # 2.5) Run inference to predict garment keypoints (normalized)
    #
    with torch.no_grad():
        c_in = c_pos.unsqueeze(0).float()  # (1,32,2)
        s_in = s_pos.unsqueeze(0).float()  # (1,10,2)
        pred = net(c_in, s_in)             # (1,32,2)
        p_pos = pred.squeeze(0)            # (32,2) normalized

    #
    # 2.6) Denormalize predicted keypoints to pixel coordinates
    #
    p_np = p_pos.cpu().numpy().astype(np.float32)  # (32,2)
    p_np[:, 0] = p_np[:, 0] * IMG_WIDTH
    p_np[:, 1] = p_np[:, 1] * IMG_HEIGHT

    #
    # 2.7) Save predicted garment keypoints JSON (pixel coords)
    #
    out_data = {"keypoints": p_np.tolist()}
    with open(OUT_KEYPOINTS_JSON, "w") as f:
        json.dump(out_data, f, indent=2)

    #
    # 2.8) Render & save skeleton visualization
    #
    sk_vis = draw_skeleton(s_pos)  # torch.Tensor (3×IMG_HEIGHT×IMG_WIDTH)
    save_image(sk_vis, OUT_SKEL_VIS_PNG)

    #
    # 2.9) Render & save cloth‐landmark visualization (using predicted p_pos)
    #
    # Re‐normalize p_np back to [0,1] so draw_cloth works:
    p_norm = torch.from_numpy(p_np.copy()).to(device)
    p_norm[:, 0] /= float(IMG_WIDTH)
    p_norm[:, 1] /= float(IMG_HEIGHT)
    cloth_vis = draw_cloth(p_norm)
    save_image(cloth_vis, OUT_CLOTH_VIS_PNG)

    print("Finished processing single frame:")
    print(f"  • Predicted keypoints JSON → {OUT_KEYPOINTS_JSON}")
    print(f"  • Skeleton visualization  → {OUT_SKEL_VIS_PNG}")
    print(f"  • Cloth visualization     → {OUT_CLOTH_VIS_PNG}")


if __name__ == "__main__":
    main()
