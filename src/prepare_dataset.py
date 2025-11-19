#prepare_dataset.py
import os
import json
import shutil
from PIL import Image, ImageDraw, ImageOps

def clamp_point(x, y, w, h):
    """Clamp polygon coordinates to image bounds [0, w-1], [0, h-1] and convert to int."""
    xi = int(round(x))
    yi = int(round(y))
    xi = max(0, min(w-1, xi))
    yi = max(0, min(h-1, yi))
    return xi, yi

def coco_to_masks_with_logging(json_path, src_img_dir, dst_img_dir, dst_mask_dir,
                               preview_dir=None, max_previews=5):
    """
    Convert COCO-format annotations to binary masks.
    Optionally creates preview images overlaying masks on the originals.
    """
    print(f"\nProcessing JSON: {json_path}")
    print(f"Source images: {src_img_dir}")

    # Ensure output directories exist
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)

    # Load COCO JSON
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    if "images" not in coco:
        raise ValueError("JSON doesn't contain 'images' key")

    images_list = coco["images"]
    print(f"Found {len(images_list)} images in JSON.")

    anns = coco.get("annotations", [])
    print(f"Found {len(anns)} annotations in JSON.")

    # Map image_id -> image info for easy lookup
    images = {img["id"]: img for img in images_list}

    # Prepare empty masks and copy source images
    masks = {}
    missing_images = []
    for img in images_list:
        w, h = img["width"], img["height"]
        masks[img["id"]] = Image.new("L", (w, h), 0)  # initialize blank mask

        src_path = os.path.join(src_img_dir, os.path.basename(img["file_name"]))
        dst_path = os.path.join(dst_img_dir, os.path.basename(img["file_name"]))
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            missing_images.append(src_path)

    if missing_images:
        print("⚠️ Missing image files (will still create blank masks for them):")
        for m in missing_images[:10]:
            print("   ", m)
        print(f"   ... total missing: {len(missing_images)}")

    # Draw polygon annotations onto masks
    rle_found = 0
    skipped_ann = 0
    for ann in anns:
        img_id = ann["image_id"]
        if img_id not in masks:
            skipped_ann += 1
            continue

        segmentation = ann.get("segmentation", [])
        if isinstance(segmentation, dict):
            # RLE format detected — skip
            rle_found += 1
            continue

        mask_img = masks[img_id]
        draw = ImageDraw.Draw(mask_img)
        w, h = mask_img.size

        for seg in segmentation:
            if not isinstance(seg, (list, tuple)) or len(seg) < 6:
                # Not enough points for a polygon
                continue
            # Convert flat list to (x, y) tuples with clamping
            poly_pts = [clamp_point(seg[i], seg[i+1], w, h) for i in range(0, len(seg), 2)]
            # Draw filled polygon
            try:
                draw.polygon(poly_pts, outline=1, fill=1)
            except Exception as e:
                print(f"   Error drawing polygon for image_id {img_id}: {e}")
                skipped_ann += 1

    if rle_found:
        print(f"Found {rle_found} RLE-encoded annotations;")

    # Save masks and optionally create preview overlays
    saved_masks = 0
    preview_count = 0
    for img in images_list:
        base_fn = os.path.basename(img["file_name"])
        mask_name = base_fn.replace(".jpg", ".png")
        mask_path = os.path.join(dst_mask_dir, mask_name)
        masks[img["id"]].save(mask_path)
        saved_masks += 1

        if preview_dir and preview_count < max_previews:
            src_img_path = os.path.join(dst_img_dir, base_fn)
            if os.path.exists(src_img_path):
                try:
                    im = Image.open(src_img_path).convert("RGBA")
                    mask = masks[img["id"]]
                    # Create red overlay where mask==1
                    red = Image.new("RGBA", im.size, (255,0,0,120))
                    # Composite red on original image using mask
                    composed = Image.composite(red, im, mask.convert("L"))
                    # Blend original and overlay for context
                    blended = Image.blend(im, composed, alpha=0.5)
                    preview_path = os.path.join(preview_dir, base_fn.replace(".jpg", "_preview.png"))
                    blended.save(preview_path)
                    preview_count += 1
                except Exception as e:
                    print(f"   Could not create preview for {src_img_path}: {e}")

    print(f"Saved {saved_masks} masks to: {dst_mask_dir}")
    if preview_dir:
        print(f"Saved up to {preview_count} preview overlays to: {preview_dir}")

    print("Done.\n")

if __name__ == "__main__":
    # --- USER CONFIG: set paths for your system ---
    base_src = r"/content/drive/MyDrive/raw"
    out_base = r"/content/drive/MyDrive/data"

    # Process training set
    coco_to_masks_with_logging(
        json_path=os.path.join(base_src, "train", "train.json"),
        src_img_dir=os.path.join(base_src, "train", "image"),
        dst_img_dir=os.path.join(out_base, "train", "images"),
        dst_mask_dir=os.path.join(out_base, "train", "masks"),
        preview_dir=os.path.join(out_base, "previews", "train"),
        max_previews=5
    )

    # Process validation set
    coco_to_masks_with_logging(
        json_path=os.path.join(base_src, "val", "val.json"),
        src_img_dir=os.path.join(base_src, "val", "image"),
        dst_img_dir=os.path.join(out_base, "val", "images"),
        dst_mask_dir=os.path.join(out_base, "val", "masks"),
        preview_dir=os.path.join(out_base, "previews", "val"),
        max_previews=5
    )

    # Copy test images only (no masks)
    src_test = os.path.join(base_src, "test")
    dst_test = os.path.join(out_base, "test", "images")
    os.makedirs(dst_test, exist_ok=True)
    test_files = 0
    for f in os.listdir(src_test):
        if f.lower().endswith((".jpg", ".tif", ".png")):
            shutil.copy2(os.path.join(src_test, f), os.path.join(dst_test, f))
            test_files += 1
    print(f"Copied {test_files} test images to {dst_test}")
