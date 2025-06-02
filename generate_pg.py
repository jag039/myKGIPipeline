#!/usr/bin/env python3
"""
process_parse.py (no-args version, without explicit RGB/L conversions)

Assumes this folder structure:

  ./input/
      parse_ag.png       # 1×H×W grayscale PNG (agnostic parse, values 0..12)

  ./output/
      skeleton_vis.png   # 3×H×W RGB PNG from KG (skeleton vis)
      cloth_vis.png      # 3×H×W RGB PNG from KG (cloth vis)

  ./checkpoints_pretrained/pg/
      step_9999.pt       # Pretrained ParseGenerator checkpoint

On each run, this script will:
  1) Load ./output/skeleton_vis.png   → (3×H×W) float‐Tensor
  2) Load ./output/cloth_vis.png      → (3×H×W) float‐Tensor
  3) Load ./input/parse_ag.png        → (1×H×W) Long‐Tensor; convert to one‐hot (13×H×W)
  4) Concatenate into (1×19×H×W), run ParseGenerator inference
  5) Convert logits (13×H×W) → argmax → colorized PIL PNG
  6) Save as ./output/predicted_parse.png

Edit the CONFIGURE PATHS section if you rename files or folders.
"""

from __future__ import print_function, absolute_import, division
import os
import numpy as np
import torch
from PIL import Image
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

# ---------------------------------------------------
# === CONFIGURE PATHS BELOW ===
# ---------------------------------------------------

INPUT_DIR       = "./input"
PARSE_AG_PNG    = os.path.join(INPUT_DIR, "parse_ag.png")     # 1×H×W (values 0..12)

OUT_DIR         = "./output"
SKEL_VIS_PNG    = os.path.join(OUT_DIR, "skeleton_vis.png")   # (3×H×W) from KG
CLOTH_VIS_PNG   = os.path.join(OUT_DIR, "cloth_vis.png")      # (3×H×W) from KG

PG_CHECKPOINT   = "./checkpoints_pretrained/pg/step_9999.pt"   # Pretrained ParseGenerator checkpoint

OUT_PARSE_PNG   = os.path.join(OUT_DIR, "predicted_parse.png")# Output parse PNG

IMG_WIDTH   = 768    # width used during PG training
IMG_HEIGHT  = 1024   # height used during PG training

NUM_CLASSES = 13     # output_nc of PG

# ---------------------------------------------------
# === END CONFIGURATION ===
# ---------------------------------------------------

from pg.networks_pg import ParseGenerator

def visualize_segmap_from_logits(input_tensor, multi_channel=True, tensor_out=True, batch=0):
    palette = [
        0,   0,   0,    128,   0,   0,    254,   0,   0,
        0,   85,  0,    169,   0,  51,    254,  85,   0,
        0,   0,  85,      0, 119, 220,     85,  85,   0,
        0,   85,  85,    85,   51,  0,     52,  86, 128,
        0,  128,   0,     0,    0, 254,     51, 169, 220,
        0,   254, 254,   85,  254, 169,    169, 254,  85,
        254, 254,   0,   254, 169,   0
    ]
    if len(palette) < 256*3:
        palette += [0] * (256*3 - len(palette))

    inp = input_tensor.detach()
    if multi_channel:
        if inp.ndimension() == 4:
            label_map = torch.argmax(inp, dim=1)
        elif inp.ndimension() == 3:
            label_map = torch.argmax(inp.unsqueeze(0), dim=1)
        else:
            raise ValueError("Expected a 3D or 4D tensor when multi_channel=True")
        label_map_np = label_map[batch].cpu().numpy().astype(np.uint8)
    else:
        if inp.ndimension() == 4:
            label_map_np = inp[batch, 0].cpu().numpy().astype(np.uint8)
        elif inp.ndimension() == 3:
            label_map_np = inp[0].cpu().numpy().astype(np.uint8)
        else:
            label_map_np = inp.cpu().numpy().astype(np.uint8)

    pil_img = Image.fromarray(label_map_np, mode="P")
    pil_img.putpalette(palette)

    if tensor_out:
        to_tensor = transforms.ToTensor()
        return to_tensor(pil_img.convert("RGB"))
    else:
        return pil_img.convert("RGB")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Verify required files/folders exist
    missing = []
    for path in [OUT_DIR, SKEL_VIS_PNG, CLOTH_VIS_PNG, PARSE_AG_PNG]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        raise FileNotFoundError(f"Missing required file or folder(s): {missing}")
    if not os.path.isfile(PG_CHECKPOINT):
        raise FileNotFoundError(f"Missing ParseGenerator checkpoint: {PG_CHECKPOINT}")

    # 2) Ensure output folder exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # 3) Load and preprocess inputs
    # 3a) Skeleton visualization (RGB) → (3×H×W) float Tensor
    sk_img = Image.open(SKEL_VIS_PNG)
    sk_img = transforms.Resize(IMG_WIDTH)(sk_img)
    to_tensor_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    sk_vis = to_tensor_norm(sk_img).to(device)  # (3, H, W)

    # 3b) Cloth visualization (RGB) → (3×H×W) float Tensor
    cloth_img = Image.open(CLOTH_VIS_PNG)
    cloth_img = transforms.Resize(IMG_WIDTH)(cloth_img)
    ck_vis = to_tensor_norm(cloth_img).to(device)  # (3, H, W)

    # 3c) Agnostic parse (single‐channel) → (1×H×W) LongTensor
    parse_ag_img = Image.open(PARSE_AG_PNG)
    parse_ag_resized = transforms.Resize(IMG_WIDTH)(parse_ag_img)
    parse_ag_np = np.array(parse_ag_resized, dtype=np.uint8)  # (H, W)
    parse_ag_tensor = torch.from_numpy(parse_ag_np)[None, :, :].long().to(device)  # (1, H, W)

    # Convert (1×H×W) → (13×H×W) one‐hot
    parse_ag_13 = torch.zeros((NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH), dtype=torch.float32, device=device)
    parse_ag_13 = parse_ag_13.scatter_(0, parse_ag_tensor, 1.0)  # (13, H, W)
    parse_ag_onehot = parse_ag_13

    # Concatenate into (1,19,H,W)
    input_tensor = torch.cat([parse_ag_onehot, sk_vis, ck_vis], dim=0).unsqueeze(0)

    # 4) Load ParseGenerator, run inference
    cudnn.benchmark = True
    net = ParseGenerator(input_nc=19, output_nc=NUM_CLASSES, ngf=64).to(device)
    ckpt = torch.load(PG_CHECKPOINT, map_location=device)
    net.load_state_dict(ckpt)
    net.eval()

    with torch.no_grad():
        logits = net(input_tensor)[0]  # (13, H, W)

    # 5) Convert logits → colorized parse and save
    out_pil = visualize_segmap_from_logits(logits.unsqueeze(0),
                                           multi_channel=True,
                                           tensor_out=False,
                                           batch=0)
    out_pil.save(OUT_PARSE_PNG)

    print("Parse Generator completed successfully.")
    print(f"  • Input skeleton_vis: {SKEL_VIS_PNG}")
    print(f"  • Input cloth_vis:    {CLOTH_VIS_PNG}")
    print(f"  • Input parse_ag:     {PARSE_AG_PNG}")
    print(f"  • Output parse PNG:   {OUT_PARSE_PNG}")

if __name__ == "__main__":
    main()
