"""
Run the trained model on a folder of images and write the Kaggle submission CSV
(one row per image: ImageID, list of building polygons).

    python src/predict.py --test-dir data/test/images --checkpoint models/unet_resnet34_buildings.pth

Without --checkpoint, the checkpoint with the best validation IoU in --checkpoint-dir is used.
The functions below (predict_mask, mask_to_polygons, ...) are also used by the demo notebook.
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from common import FINAL_PREFIX, IMAGENET_MEAN, IMAGENET_STD, ensure_drive_mounted, find_checkpoint, get_device, load_model

IMAGE_EXTS = (".tif", ".tiff", ".jpg", ".jpeg", ".png")


def read_rgb(path):
    """Read an image file as an RGB uint8 array (None if it cannot be read)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_image(img_rgb, input_size=(512, 512)):
    """HxWx3 uint8 RGB -> 1x3xHxW float tensor, resized and ImageNet-normalized (as in training)."""
    img = cv2.resize(img_rgb, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)
    img = (img.astype(np.float32) / 255.0 - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


@torch.no_grad()
def predict_probabilities(model, img_rgb, device=None, input_size=(512, 512)):
    """Building probability per pixel at the MODEL input resolution (input_size, float32)."""
    device = device or next(model.parameters()).device
    inp = preprocess_image(img_rgb, input_size).to(device)
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        logits = model(inp)
    return torch.sigmoid(logits.float())[0, 0].cpu().numpy()


def predict_mask(model, img_rgb, device=None, input_size=(512, 512), threshold=0.5, clean=True):
    """Binary building mask (uint8, 0/255) at the original image resolution."""
    probs = predict_probabilities(model, img_rgb, device, input_size)
    h, w = img_rgb.shape[:2]
    mask = (probs > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if clean:
        # close small holes, then remove isolated speckles
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def mask_to_polygons(mask, min_area=100, simplify_eps_ratio=0.01):
    """
    Outer contours of a binary mask as simplified polygons.
    Returns a list of polygons, each a list of (x, y) int tuples.
    """
    polys = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        eps = max(1.0, simplify_eps_ratio * cv2.arcLength(cnt, True))
        approx = cv2.approxPolyDP(cnt, epsilon=eps, closed=True)
        coords = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
        if len(coords) >= 3:
            polys.append(coords)
    return polys


def polygons_to_string(polys):
    """[[(x,y),...],...] -> '[[(x,y),(x,y)],[...]]' (submission format)."""
    return "[" + ",".join("[" + ",".join(f"({x},{y})" for x, y in poly) + "]" for poly in polys) + "]"


def imageid_from_filename(fname):
    """'0007.tif' -> '7'; non-numeric stems are kept as is."""
    stem = Path(fname).stem
    try:
        return str(int(stem))
    except ValueError:
        return stem


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-dir", default="data/test/images")
    p.add_argument("--checkpoint", default=None, help="path to a .pth file (overrides --checkpoint-dir)")
    p.add_argument("--checkpoint-dir", default="models")
    p.add_argument("--output-csv", default="output/submission.csv")
    p.add_argument("--mask-dir", default=None, help="if set, also save the predicted masks as PNG here")
    p.add_argument("--img-size", type=int, default=512, help="model input size (as in training)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-area", type=float, default=100, help="drop polygons smaller than this (pixels)")
    p.add_argument("--simplify-eps", type=float, default=0.01, help="polygon simplification, fraction of perimeter")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ensure_drive_mounted(args.test_dir, args.checkpoint, args.checkpoint_dir, args.output_csv, args.mask_dir)

    if args.checkpoint:
        ckpt_file = args.checkpoint
        if not os.path.isfile(ckpt_file):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
    else:
        ckpt_file = find_checkpoint(args.checkpoint_dir, FINAL_PREFIX, mode="best")
        if ckpt_file is None:
            raise FileNotFoundError(f"No {FINAL_PREFIX}*.pth in {args.checkpoint_dir}; pass --checkpoint.")
    device = get_device()
    print("Device:", device, "| checkpoint:", ckpt_file)
    model = load_model(ckpt_file, device)

    files = sorted(p for p in Path(args.test_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    print(f"Found {len(files)} test images. Running inference...")
    if args.mask_dir:
        os.makedirs(args.mask_dir, exist_ok=True)

    rows = []
    size = (args.img_size, args.img_size)
    for path in tqdm(files):
        img = read_rgb(path)
        if img is None:
            print("Warning: couldn't read", path)
            continue
        mask = predict_mask(model, img, device, size, args.threshold)
        if args.mask_dir:
            cv2.imwrite(os.path.join(args.mask_dir, path.stem + "_pred_mask.png"), mask)
        polys = mask_to_polygons(mask, args.min_area, args.simplify_eps)
        rows.append({"ImageID": imageid_from_filename(path.name), "Coordinates": polygons_to_string(polys)})

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    pd.DataFrame(rows, columns=["ImageID", "Coordinates"]).to_csv(args.output_csv, index=False)
    print("Saved submission CSV to:", args.output_csv)


if __name__ == "__main__":
    main()
