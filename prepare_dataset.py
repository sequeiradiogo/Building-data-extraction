import os
import json
import shutil
from PIL import Image, ImageDraw, ImageOps

def clamp_point(x, y, w, h):
    """Clamp polygon coords inside image bounds [0, w-1], [0, h-1] and convert to int."""
    xi = int(round(x))
    yi = int(round(y))
    xi = max(0, min(w-1, xi))
    yi = max(0, min(h-1, yi))
    return xi, yi

def coco_to_masks_with_logging(json_path, src_img_dir, dst_img_dir, dst_mask_dir,
                               preview_dir=None, max_previews=5):
    print(f"\nProcessing JSON: {json_path}")
    print(f"Source images: {src_img_dir}")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    if "images" not in coco:
        raise ValueError("JSON doesn't contain 'images' key")

    images_list = coco["images"]
    print(f"Found {len(images_list)} images in JSON.")

    anns = coco.get("annotations", [])
    print(f"Found {len(anns)} annotations in JSON.")

    # Build dict image_id -> image info
    images = {img["id"]: img for img in images_list}

    # Prepare empty masks and copy images
    masks = {}
    missing_images = []
    for img in images_list:
        w, h = img["width"], img["height"]
        mask = Image.new("L", (w, h), 0)
        masks[img["id"]] = mask

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

    # Draw polygons to masks
    rle_found = 0
    skipped_ann = 0
    for ann in anns:
        img_id = ann["image_id"]
        if img_id not in masks:
            skipped_ann += 1
            continue

        segmentation = ann.get("segmentation", [])
        if isinstance(segmentation, dict):
            # RLE format — warn and skip (could use pycocotools to decode)
            rle_found += 1
            continue

        mask_img = masks[img_id]
        draw = ImageDraw.Draw(mask_img)
        w, h = mask_img.size

        # segmentation is a list of polygons (each polygon is a flat list)
        for seg in segmentation:
            if not isinstance(seg, (list, tuple)) or len(seg) < 6:
                # not a polygon (need at least 3 points)
                continue
            # Convert to list of (x,y) with clamping
            poly_pts = []
            for i in range(0, len(seg), 2):
                x = seg[i]; y = seg[i+1]
                xi, yi = clamp_point(x, y, w, h)
                poly_pts.append((xi, yi))
            # Draw polygon (fill=1)
            try:
                draw.polygon(poly_pts, outline=1, fill=1)
            except Exception as e:
                print(f"   Error drawing polygon for image_id {img_id}: {e}")
                skipped_ann += 1

    if rle_found:
        print(f"⚠️ Found {rle_found} annotations encoded as RLE; this script skips them (use pycocotools to decode).")

    # Save masks and produce previews for the first N images
    saved_masks = 0
    preview_count = 0
    for img in images_list:
        base_fn = os.path.basename(img["file_name"])
        mask_name = base_fn.replace(".jpg", ".png")
        mask_path = os.path.join(dst_mask_dir, mask_name)
        masks[img["id"]].save(mask_path)
        saved_masks += 1

        # preview overlay
        if preview_dir and preview_count < max_previews:
            src_img_path = os.path.join(dst_img_dir, base_fn)
            if os.path.exists(src_img_path):
                try:
                    im = Image.open(src_img_path).convert("RGBA")
                    mask = masks[img["id"]]
                    # create a red overlay where mask==1
                    red = Image.new("RGBA", im.size, (255,0,0,120))
                    mask_rgb = ImageOps.colorize(mask.convert("L"), black="black", white="white").convert("L")
                    # composite red over image using mask
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
    # --- EDIT THESE paths to match your system ---
    base_src = r"C:\Users\35193\OneDrive\Ambiente de Trabalho\data"
    out_base = r"C:\Users\35193\OneDrive\Ambiente de Trabalho\data_prepared"

    # Train
    coco_to_masks_with_logging(
        json_path=os.path.join(base_src, "train", "train.json"),
        src_img_dir=os.path.join(base_src, "train", "image"),
        dst_img_dir=os.path.join(out_base, "train", "images"),
        dst_mask_dir=os.path.join(out_base, "train", "masks"),
        preview_dir=os.path.join(out_base, "previews", "train"),
        max_previews=5
    )

    # Val
    coco_to_masks_with_logging(
        json_path=os.path.join(base_src, "val", "val.json"),
        src_img_dir=os.path.join(base_src, "val", "image"),
        dst_img_dir=os.path.join(out_base, "val", "images"),
        dst_mask_dir=os.path.join(out_base, "val", "masks"),
        preview_dir=os.path.join(out_base, "previews", "val"),
        max_previews=5
    )

    # Test images copy (no masks)
    src_test = os.path.join(base_src, "test")
    dst_test = os.path.join(out_base, "test", "images")
    os.makedirs(dst_test, exist_ok=True)
    test_files = 0
    for f in os.listdir(src_test):
        if f.lower().endswith((".jpg", ".tif", ".png")):
            shutil.copy2(os.path.join(src_test, f), os.path.join(dst_test, f))
            test_files += 1
    print(f"Copied {test_files} test images to {dst_test}")
