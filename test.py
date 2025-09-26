import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import segmentation_models_pytorch as smp
import importlib, traceback


TEST_DIR = "/content/drive/MyDrive/data/test/image"   # folder with .tif test images

CHECKPOINT_PATH = None #picks the latest checkpoint
CHECKPOINT_DIR = "/content/drive/MyDrive/models"
OUTPUT_CSV = "/content/drive/MyDrive/submission.csv"
MODEL_INPUT_SIZE = (512, 512)   # same size used during training (HxW)
THRESHOLD = 0.5                 # probability threshold for binarization
MIN_AREA = 100                  # minimum contour area in pixels
SIMPLIFY_EPS_RATIO = 0.01       # poly approx epsilon = ratio * arcLength
SAVE_MASK_IMAGES = True         # set True to also save predicted masks 
MASK_SAVE_DIR = "/content/drive/MyDrive/test_predictions_masks"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# create save dir if needed
if SAVE_MASK_IMAGES:
    os.makedirs(MASK_SAVE_DIR, exist_ok=True)


# build same model used in training
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)

# checkpoint loader 
def safe_load_checkpoint(path, map_location):
    """
    Attempts safe loading with weights_only=True first, then allowlists unsafe globals if needed,
    and finally falls back to trusted load (weights_only=False).
    Returns loaded object (either dict or state_dict).
    """
    ckpt = None
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        print("Loaded checkpoint with weights_only=True")
        return ckpt
    except Exception as e:
        print("weights_only=True failed:", e)

    # try to inspect unsafe globals required by checkpoint
    unsafe = []
    try:
        unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
        print("Unsafe globals reported by checkpoint:", unsafe)
    except Exception as e2:
        print("Could not query unsafe globals:", e2)
        unsafe = []

    # try to import and allowlist the reported globals
    safe_objs = []
    for full_name in unsafe:
        try:
            module_name, obj_name = full_name.rsplit(".", 1)
            mod = importlib.import_module(module_name)
            obj = getattr(mod, obj_name)
            safe_objs.append(obj)
            print("Will allowlist:", full_name)
        except Exception as imp_e:
            print(f"Could not import {full_name}: {imp_e}")

    if safe_objs:
        try:
            torch.serialization.add_safe_globals(safe_objs)
            ckpt = torch.load(path, map_location=map_location, weights_only=True)
            print("Loaded checkpoint after allowlisting unsafe globals (weights_only=True).")
            return ckpt
        except Exception as e3:
            print("Failed after allowlisting:", e3)
            traceback.print_exc()
            print("Falling back to trusted load (weights_only=False).")

    # last resort: trusted load.
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        print("Loaded checkpoint with weights_only=False (trusted load).")
        return ckpt
    except Exception as e_final:
        print("Trusted load also failed:", e_final)
        traceback.print_exc()
        raise RuntimeError("Could not load checkpoint: " + str(e_final))

# Choose checkpoint file 
if CHECKPOINT_PATH and os.path.isfile(CHECKPOINT_PATH):
    ckpt_file = CHECKPOINT_PATH
else:
    ckpts = sorted([os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")])
    if len(ckpts) == 0:
        raise FileNotFoundError("No checkpoint provided and no .pth files found in CHECKPOINT_DIR.")
    ckpt_file = ckpts[-1]  # newest
print("Using checkpoint file:", ckpt_file)

# load checkpoint robustly
ckpt = safe_load_checkpoint(ckpt_file, map_location=device)

# resolve model state dict from ckpt
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model_state = ckpt['model_state_dict']
elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
    # appears to be a state_dict itself
    model_state = ckpt
else:
    # fallback to get('model_state_dict', ckpt)
    try:
        model_state = ckpt.get('model_state_dict', ckpt)
    except Exception:
        model_state = ckpt

# load into model
try:
    model.load_state_dict(model_state, strict=False)
    print("Model weights loaded (strict=False).")
except Exception as e:
    print("Model.load_state_dict failed:", e)
    # If loading strict=False failed, still attempt to load as state dict with strict=True if it's a pure state_dict.
    try:
        model.load_state_dict(model_state, strict=True)
        print("Model weights loaded with strict=True (fallback).")
    except Exception as e2:
        print("Model strict load also failed:", e2)
        raise RuntimeError("Failed to load model weights.")

model.to(device)
model.eval()
print("Model ready for inference.")

#  Helpers 
def preprocess_image_for_model(orig_img_rgb, input_size):
    """
    orig_img_rgb: numpy HxWx3 uint8 (RGB)
    returns: tensor 1x3xHxW float (on CPU) normalized to ImageNet
    """
    # resize (cv2 expects (width, height))
    img_resized = cv2.resize(orig_img_rgb, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)
    img_resized = img_resized.astype(np.float32) / 255.0
    # normalize with ImageNet mean/std (same used in training)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_resized = (img_resized - mean) / std
    # HWC -> CHW
    img_chw = img_resized.transpose(2, 0, 1)
    tensor = torch.from_numpy(img_chw).unsqueeze(0).float()
    return tensor

def postprocess_mask_to_polygons(mask_uint8, min_area=MIN_AREA, simplify_eps_ratio=SIMPLIFY_EPS_RATIO):
    """
    mask_uint8: HxW uint8 (0 or 255)
    returns: list_of_polygons, each polygon is a list of (x,y) ints
    """
    polys = []
    # find contours 
    contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        # approximate polygon
        arc_len = cv2.arcLength(cnt, True)
        eps = max(1.0, simplify_eps_ratio * arc_len)
        approx = cv2.approxPolyDP(cnt, epsilon=eps, closed=True)
        coords = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
        if len(coords) >= 3:
            polys.append(coords)
    return polys

def imageid_from_filename(fname):
    # remove extension and leading zeros if numeric
    stem = Path(fname).stem
    try:
        num = int(stem)
        return str(num)  # as string of integer (e.g., "1")
    except Exception:
        return stem

# Inference loop 
test_dir = Path(TEST_DIR)
files = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in (".tif")])

results = []
print(f"Found {len(files)} test images. Running inference...")

with torch.no_grad():
    for p in tqdm(files):
        # read original image (RGB)
        orig = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if orig is None:
            print("Warning: couldn't read", p)
            continue
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = orig.shape[:2]

        # preprocess and run model
        inp = preprocess_image_for_model(orig, MODEL_INPUT_SIZE).to(device)
        # use autocast if CUDA available
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(inp)  # 1x1xH_inxW_in
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()  # H_in x W_in

        # threshold and resize back to original resolution
        bin_small = (probs > THRESHOLD).astype(np.uint8) * 255  # H_in x W_in uint8
        bin_orig = cv2.resize(bin_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)  # orig size

        # optional morphological clean (remove small holes / smooth)
        kernel = np.ones((3,3), np.uint8)
        bin_orig = cv2.morphologyEx(bin_orig, cv2.MORPH_CLOSE, kernel, iterations=1)
        bin_orig = cv2.morphologyEx(bin_orig, cv2.MORPH_OPEN, kernel, iterations=1)

        # save mask image if requested
        if SAVE_MASK_IMAGES:
            save_name = Path(MASK_SAVE_DIR) / (p.stem + "_pred_mask.png")
            cv2.imwrite(str(save_name), bin_orig)

        # extract polygons from the binary mask
        polys = postprocess_mask_to_polygons(bin_orig, min_area=MIN_AREA, simplify_eps_ratio=SIMPLIFY_EPS_RATIO)

        image_id = imageid_from_filename(p.name)
        results.append({
            "ImageID": image_id,
            "Coordinates": polys  # will stringify later
        })

# Create .csv file
rows = []
for r in results:
    coords = r["Coordinates"]
    if not coords:
        coords_str = "[]"
    else:
        poly_strs = []
        for poly in coords:
            pts = ",".join([f"({int(x)},{int(y)})" for (x,y) in poly])
            poly_strs.append(f"[{pts}]")
        coords_str = "[" + ",".join(poly_strs) + "]"
    rows.append({"ImageID": r["ImageID"], "Coordinates": coords_str})

df = pd.DataFrame(rows, columns=["ImageID", "Coordinates"])
df.to_csv(OUTPUT_CSV, index=False)
print("Saved submission CSV to:", OUTPUT_CSV)

