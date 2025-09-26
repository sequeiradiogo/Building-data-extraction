
import os, sys
from time import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score

# Mount Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Paths
DATA_DIR = "/content/drive/MyDrive/data"
TRAIN_IMAGES = f"{DATA_DIR}/train/images"
TRAIN_MASKS  = f"{DATA_DIR}/train/masks"
VAL_IMAGES   = f"{DATA_DIR}/val/images"
VAL_MASKS    = f"{DATA_DIR}/val/masks"
CHECKPOINT_DIR = "/content/drive/MyDrive/models" # to save the models
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Device and seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
print("Device:", device)

# Hyperparameters
IMG_SIZE = (512, 512)
BATCH_SIZE = 4
NUM_EPOCHS = 40
BASE_LR = 5e-4
MAX_LR = 1e-3
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 2
PATIENCE = 7


# Transforms
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

train_aug = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.2),
    A.Rotate(limit=12, p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.OneOf([A.MotionBlur(p=0.3), A.GaussianBlur(p=0.3), A.GaussNoise(p=0.3)], p=0.2),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])

val_aug = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])


# Dataset class

class SegmentationDatasetAlb(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg"))])
        self.masks  = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith((".png"))])

        pairs = []
        mask_set = set(self.masks)
        for img in self.images:
            name = os.path.splitext(img)[0]
            cand = name + ".png"
            if cand in mask_set:
                pairs.append((img, cand))
            else:
                found = next((m for m in self.masks if os.path.splitext(m)[0].startswith(name)), None)
                if found:
                    pairs.append((img, found))
                else:
                    print(f"Warning: No mask found for image {img}")
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 0).astype('uint8')

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            if isinstance(mask, torch.Tensor):
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0).float()
                elif mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask.float()
                else:
                    mask = mask.float()
            else:
                mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)
            return img.float(), mask
        else:
            img = torch.tensor(img.transpose(2,0,1), dtype=torch.float) / 255.0
            mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)
            return img, mask

# DataLoaders
train_dataset = SegmentationDatasetAlb(TRAIN_IMAGES, TRAIN_MASKS, transform=train_aug)
val_dataset   = SegmentationDatasetAlb(VAL_IMAGES, VAL_MASKS, transform=val_aug)

print("Train pairs:", len(train_dataset), "Val pairs:", len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

# Model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
).to(device)

# Loss (BCE)
bce = nn.BCEWithLogitsLoss()

def dice_loss_logits(pred_logits, target, eps=1e-6):
    pred = torch.sigmoid(pred_logits)
    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2. * intersection + eps) / (union + eps)
    return 1 - dice.mean()

def combined_loss(pred_logits, target, alpha=0.3): # alpha = 0.3 -> more weight towards dice
    return alpha * bce(pred_logits, target) + (1 - alpha) * dice_loss_logits(pred_logits, target)


# Evaluation metrics
def pixel_iou_and_dice(pred_logits, target, thresh=0.5):
    pred = (torch.sigmoid(pred_logits) > thresh).cpu().numpy().astype(np.uint8)
    gt   = target.cpu().numpy().astype(np.uint8)
    batch = pred.shape[0]
    ious, dices = [], []
    for i in range(batch):
        p, g = pred[i,0], gt[i,0]
        inter = (p & g).sum()
        union = (p | g).sum()
        iou = inter / union if union > 0 else (1.0 if inter == 0 else 0.0)
        dice = 2*inter / (p.sum() + g.sum()) if (p.sum()+g.sum())>0 else 1.0
        ious.append(iou); dices.append(dice)
    return float(np.mean(ious)), float(np.mean(dices))

def precision_recall_f1(pred_logits, target, thresh=0.5):
    pred = (torch.sigmoid(pred_logits) > thresh).cpu().numpy().astype(np.uint8)
    gt   = target.cpu().numpy().astype(np.uint8)
    precs, recs, f1s = [], [], []
    for i in range(pred.shape[0]):
        p, g = pred[i,0].flatten(), gt[i,0].flatten()
        if g.sum() == 0 and p.sum() == 0:
            precs.append(1.0); recs.append(1.0); f1s.append(1.0)
        else:
            precs.append(precision_score(g, p, zero_division=0))
            recs.append(recall_score(g, p, zero_division=0))
            f1s.append(f1_score(g, p, zero_division=0))
    return np.mean(precs), np.mean(recs), np.mean(f1s)


# Optimizer, scheduler and AMP
optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

steps_per_epoch = max(1, len(train_loader))
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=MAX_LR, steps_per_epoch=steps_per_epoch, epochs=NUM_EPOCHS,
    pct_start=0.3, anneal_strategy='cos', div_factor=10.0, final_div_factor=100.0
)

scaler = torch.amp.GradScaler('cuda')

# Resume checkpoint if available
start_epoch = 1
best_val_iou = -1.0
epochs_no_improve = 0
best_ckpt = None

import os, importlib, traceback, torch

#reformulated logic to load model
ckpts = sorted([os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")])
if len(ckpts) == 0:
    print("No checkpoints found.")
else:
    latest = ckpts[-1]
    print("Found checkpoint:", latest)
    ckpt = None


    try:
        ckpt = torch.load(latest, map_location=device, weights_only=True)
        print("Loaded checkpoint with weights_only=True")
    except Exception as e:
        print("weights_only=True failed:", e)

        try:
            unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(latest)
            print("Unsafe globals reported by checkpoint:", unsafe)
        except Exception as e2:
            print("Could not query unsafe globals:", e2)
            unsafe = []

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
                ckpt = torch.load(latest, map_location=device, weights_only=True)
                print("Loaded checkpoint after allowlisting unsafe globals (weights_only=True).")
            except Exception as e3:
                print("Failed after allowlisting:", e3)
                traceback.print_exc()
                print("Falling back to trusted load (weights_only=False).")
                ckpt = torch.load(latest, map_location=device, weights_only=False)
        else:
            # Nothing to allowlist: last resort (trusted load)
            print("No safe objects added — falling back to trusted load (weights_only=False).")
            ckpt = torch.load(latest, map_location=device, weights_only=False)


    try:
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model_state = ckpt['model_state_dict']
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            # appears to be a state_dict
            model_state = ckpt
        else:
            # fallback: try get('model_state_dict', ckpt)
            model_state = ckpt.get('model_state_dict', ckpt)

        model.load_state_dict(model_state, strict=False)
        print("Model weights loaded (strict=False).")
    except Exception as e:
        print("Model load failed:", e)

    try:
        if isinstance(ckpt, dict) and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print("Optimizer state loaded.")
    except Exception as e:
        print("Optimizer load failed or missing; reinitializing optimizer.", e)

# Training loop
for epoch in range(start_epoch, NUM_EPOCHS + 1):
    model.train()
    running_loss, steps = 0.0, 0
    t0 = time()

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{NUM_EPOCHS}", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = combined_loss(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        steps += images.size(0)
        pbar.set_postfix(loss=float(loss.item()))

    train_loss = running_loss / steps if steps > 0 else 0.0

    # Validation
    model.eval()
    val_loss, vsteps = 0.0, 0
    ious, dices, precs, recs, f1s = [], [], [], [], []
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Val Epoch {epoch}/{NUM_EPOCHS}", leave=False):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = combined_loss(outputs, masks)
            val_loss += loss.item() * images.size(0)
            vsteps += images.size(0)

            iou, dice = pixel_iou_and_dice(outputs, masks)
            prec, rec, f1 = precision_recall_f1(outputs, masks)

            ious.append(iou); dices.append(dice)
            precs.append(prec); recs.append(rec); f1s.append(f1)

    val_loss = val_loss / vsteps if vsteps > 0 else 0.0
    mean_iou = np.mean(ious); mean_dice = np.mean(dices)
    mean_prec, mean_rec, mean_f1 = np.mean(precs), np.mean(recs), np.mean(f1s)
    epoch_time = time() - t0

    print(f"Epoch {epoch} | {epoch_time:.1f}s | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
          f"IoU {mean_iou:.4f} | Precision {mean_prec:.4f} | Recall {mean_rec:.4f} | F1 {mean_f1:.4f}")

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"unet_resnet34_epoch{epoch}_IoU{mean_iou:.4f}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'val_loss': val_loss,
        'val_dice': mean_dice
    }, ckpt_path)
    best_ckpt = ckpt_path

    if mean_iou > best_val_iou:
        best_val_iou = mean_iou
        epochs_no_improve = 0
        print("New best model (saved):", ckpt_path)
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epochs (best {best_val_iou:.4f})")

    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping (no improvement for {PATIENCE} epochs).")
        break

# Visual check
model.eval()
with torch.no_grad():
    imgs, masks = next(iter(val_loader))
    outputs = model(imgs.to(device))
    preds = torch.sigmoid(outputs).cpu().numpy()
    imgs_np = imgs.numpy()
    for i in range(min(3, imgs_np.shape[0])):
        img = imgs_np[i].transpose(1,2,0)
        gt = masks[i,0].numpy()
        pred = (preds[i,0] > 0.5).astype(np.uint8)
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.imshow((img * IMAGENET_STD + IMAGENET_MEAN).clip(0,1)); plt.title("Image"); plt.axis('off')
        plt.subplot(1,3,2); plt.imshow(gt, cmap='gray'); plt.title("GT"); plt.axis('off')
        plt.subplot(1,3,3); plt.imshow(pred, cmap='gray'); plt.title("Pred"); plt.axis('off')
        plt.show()

print("Done. Best checkpoint (most recent save):", best_ckpt)

