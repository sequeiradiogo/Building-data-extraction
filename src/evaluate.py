"""
Evaluate the full pipeline (model + post-processing -> polygons) on a COCO-annotated split.

    python src/evaluate.py --images raw/val/image --annotations raw/val/val.json \
                           --checkpoint models/unet_resnet34_buildings.pth

Reports a per-building F1: a predicted polygon is a true positive when it overlaps an
annotated building with IoU > 0.5 (such matches are necessarily one-to-one), every other
polygon is a false positive and every unmatched building a false negative. This mirrors how
the leaderboard scores polygons, unlike the pixel IoU used during training.
"""
import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from common import get_device, load_model
from predict import (add_postprocessing_args, postprocessing_kwargs, predict_probabilities,
                     probs_to_polygons, read_rgb)


def rasterize_polygons(polys, shape):
    """List of polygons [(x, y), ...] -> uint16 label image (polygon i gets label i + 1)."""
    labels = np.zeros(shape, np.uint16)
    for i, poly in enumerate(polys, start=1):
        cv2.fillPoly(labels, [np.asarray(poly, np.int32).reshape(-1, 1, 2)], i)
    return labels


def load_coco_instances(json_path):
    """file name -> (height, width, list of polygons); one polygon per annotated building."""
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    polys = defaultdict(list)
    for ann in coco.get("annotations", []):
        seg = ann.get("segmentation")
        if not isinstance(seg, list):
            continue  # RLE annotations are not used in this dataset
        parts = [np.asarray(s, np.float64).reshape(-1, 2).round().astype(np.int32) for s in seg if len(s) >= 6]
        if parts:
            # buildings made of several parts are kept together as one instance
            polys[ann["image_id"]].append(parts)
    return {os.path.basename(img["file_name"]): (img["height"], img["width"], polys[img["id"]])
            for img in coco["images"]}


def rasterize_instances(instances, shape):
    labels = np.zeros(shape, np.uint16)
    for i, parts in enumerate(instances, start=1):
        cv2.fillPoly(labels, [p.reshape(-1, 1, 2) for p in parts], i)
    return labels


def count_matches(pred_labels, n_pred, gt_labels, n_gt, iou_threshold=0.5):
    """Number of predicted instances with IoU > iou_threshold to a ground-truth instance."""
    if n_pred == 0 or n_gt == 0:
        return 0
    idx = pred_labels.astype(np.int64) * (n_gt + 1) + gt_labels
    inter = np.bincount(idx.ravel(), minlength=(n_pred + 1) * (n_gt + 1)).reshape(n_pred + 1, n_gt + 1)
    area_p, area_g = inter.sum(1), inter.sum(0)
    inter = inter[1:, 1:]
    union = area_p[1:, None] + area_g[None, 1:] - inter
    iou = inter / np.maximum(union, 1)
    return int((iou > iou_threshold).sum())


def f1_from_counts(tp, n_pred, n_gt):
    fp, fn = n_pred - tp, n_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    return precision, recall, f1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="folder with the images referenced in the COCO file")
    p.add_argument("--annotations", required=True, help="COCO json with building polygons")
    p.add_argument("--checkpoint", default="models/unet_resnet34_buildings.pth")
    p.add_argument("--img-size", type=int, default=512)
    add_postprocessing_args(p)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    gt = load_coco_instances(args.annotations)
    device = get_device()
    model = load_model(args.checkpoint, device)

    tp = n_pred = n_gt = 0
    for name, (h, w, instances) in tqdm(sorted(gt.items())):
        img = read_rgb(os.path.join(args.images, name))
        if img is None:
            print("Warning: couldn't read", name)
            continue
        probs = predict_probabilities(model, img, device, (args.img_size, args.img_size))
        polys = probs_to_polygons(probs, img.shape[:2], **postprocessing_kwargs(args))
        tp += count_matches(rasterize_polygons(polys, img.shape[:2]), len(polys),
                            rasterize_instances(instances, img.shape[:2]), len(instances))
        n_pred += len(polys)
        n_gt += len(instances)

    precision, recall, f1 = f1_from_counts(tp, n_pred, n_gt)
    print(f"Buildings: {n_gt} annotated, {n_pred} predicted, {tp} matched (IoU > 0.5)")
    print(f"Precision {precision:.4f} | Recall {recall:.4f} | F1 {f1:.4f}")


if __name__ == "__main__":
    main()
