"""Shared helpers: device, Google Drive, checkpoints, model construction and figures."""
import os
import re

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Checkpoints written by train.py / train_baseline.py start with these prefixes.
FINAL_PREFIX = "unet_resnet34_"
BASELINE_PREFIX = "unet_baseline_"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def ensure_drive_mounted(*paths):
    """
    If any path lives on Google Drive (/content/drive/...) and Drive is not mounted,
    try to mount it. Mounting only works from inside a Colab notebook kernel
    (e.g. `%run src/train.py ...`); with `!python ...` mount Drive in a cell first.
    """
    if not any(str(p).startswith("/content/drive") for p in paths if p):
        return
    if os.path.isdir("/content/drive/MyDrive"):
        return
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        raise RuntimeError(
            "Paths point to Google Drive but it is not mounted. In a Colab cell run:\n"
            "    from google.colab import drive; drive.mount('/content/drive')"
        ) from e


# ---------------------------------------------------------------- checkpoints
def _numpy_safe_globals():
    """numpy objects that older checkpoints contain (e.g. a np.float64 metric)."""
    core = getattr(np, "_core", None) or np.core
    objs = [np.dtype, core.multiarray.scalar]
    dtypes = getattr(np, "dtypes", None)
    if dtypes is not None:
        objs += [getattr(dtypes, n) for n in ("Float64DType", "Float32DType", "Int64DType")
                 if hasattr(dtypes, n)]
    return objs


def load_checkpoint(path, map_location="cpu"):
    """
    Load a .pth file with the safe `weights_only=True` loader. If that fails, fall back
    to a full unpickle, which can execute arbitrary code: only use it with checkpoints
    you trust (e.g. your own).
    """
    try:
        with torch.serialization.safe_globals(_numpy_safe_globals()):
            return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        print(f"Safe load failed ({type(e).__name__}); falling back to weights_only=False "
              "(only do this for checkpoints you trust).")
        return torch.load(path, map_location=map_location, weights_only=False)


def extract_state_dict(ckpt):
    """Accept either a full training checkpoint or a bare state_dict."""
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def iou_from_filename(path):
    """Extract the IoU value from '..._IoU0.7833.pth', or None."""
    m = re.search(r"_IoU(\d+(?:\.\d+)?)\.pth$", os.path.basename(path))
    return float(m.group(1)) if m else None


def find_checkpoint(directory, prefix, mode="best"):
    """
    Pick a checkpoint in `directory` whose name starts with `prefix`.
    mode="best":   highest IoU encoded in the file name (falls back to newest)
    mode="latest": most recently written file
    Returns None if there is none.
    """
    if not directory or not os.path.isdir(directory):
        return None
    ckpts = [os.path.join(directory, f) for f in os.listdir(directory)
             if f.startswith(prefix) and f.endswith(".pth")]
    if not ckpts:
        return None
    if mode == "best":
        scored = [(iou_from_filename(c), c) for c in ckpts]
        scored = [s for s in scored if s[0] is not None]
        if scored:
            return max(scored)[1]
    return max(ckpts, key=os.path.getmtime)


# ---------------------------------------------------------------------- model
def build_unet(encoder_weights=None):
    """U-Net with a ResNet-34 encoder (segmentation_models_pytorch), 1 output channel."""
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name="resnet34", encoder_weights=encoder_weights,
                    in_channels=3, classes=1)


def load_model(checkpoint_path, device=None):
    """Build the final model and load weights from a checkpoint (strict)."""
    device = device or get_device()
    model = build_unet(encoder_weights=None)
    state = extract_state_dict(load_checkpoint(checkpoint_path, map_location="cpu"))
    # strict=True: an incompatible checkpoint must fail loudly instead of silently
    # leaving the model with random weights.
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


# -------------------------------------------------------------------- figures
def save_prediction_grid(images, gts, preds, path, normalized=True, max_rows=3):
    """
    Save an Image | Ground truth | Prediction grid.
    images: N x 3 x H x W tensor/array, gts/preds: N x H x W (0/1).
    normalized: images are ImageNet-normalized and must be de-normalized for display.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = np.asarray(images)
    n = min(max_rows, images.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)
    for i in range(n):
        img = images[i].transpose(1, 2, 0)
        if normalized:
            img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        for ax, im, title in zip(axes[i], (img.clip(0, 1), gts[i], preds[i]),
                                 ("Image", "Ground truth", "Prediction")):
            ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print("Saved example predictions to:", path)
