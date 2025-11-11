import os, sys
from time import time
from pathlib import Path


# Mount drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# paths
DATA_DIR = "/content/drive/MyDrive/data"
TRAIN_IMAGES = f"{DATA_DIR}/train/images"
TRAIN_MASKS  = f"{DATA_DIR}/train/masks"
VAL_IMAGES   = f"{DATA_DIR}/val/images"
VAL_MASKS    = f"{DATA_DIR}/val/masks"

REPO_DIR = "/content/drive/MyDrive/siemens"
CHECKPOINT_DIR = "/content/drive/MyDrive/models" #path to save models
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
sys.path.append(REPO_DIR)

from dataset import SegmentationDataset #Dataset file


# device and seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
print("Device:", device)


# hyperparameters
IMG_SIZE = (512, 512)
BATCH_SIZE = 4
NUM_EPOCHS = 40
LR = 1e-3
NUM_WORKERS = 2
PIN_MEMORY = True
PATIENCE = 7  


# transforms + datasets
train_transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(degrees=10),
    T.ToTensor()
])
val_transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor()
])

train_dataset = SegmentationDataset(TRAIN_IMAGES, TRAIN_MASKS, transform=train_transform, target_size=IMG_SIZE)
val_dataset   = SegmentationDataset(VAL_IMAGES,   VAL_MASKS,   transform=val_transform,   target_size=IMG_SIZE)

print(f"Train pairs: {len(train_dataset)}  |  Val pairs: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# model (UNet)
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_ch, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up2 = DoubleConv(256+128, 128)
        self.dconv_up1 = DoubleConv(128+64, 64)
        self.conv_last = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        x = self.dconv_down3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out

model = UNet().to(device)

# loss: BCE + Dice combo
bce = nn.BCEWithLogitsLoss()

def dice_loss_logits(pred_logits, target, eps=1e-6):
    pred = torch.sigmoid(pred_logits)
    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2. * intersection + eps) / (union + eps)
    return 1 - dice.mean()

def combined_loss(pred_logits, target, alpha=0.5): #alpha 0.5 equal priority to BCE and dice loss
    return alpha * bce(pred_logits, target) + (1 - alpha) * dice_loss_logits(pred_logits, target)


# metric helpers (IoU & Dice)
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
    return np.mean(ious), np.mean(dices)


# optimizer + scheduler + scaler
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
scaler = torch.amp.GradScaler('cuda')


# resume from best checkpoint
best_val_dice = -1.0
best_ckpt = None
epochs_no_improve = 0
start_epoch = 1

ckpts = [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
if ckpts:
    latest_ckpt = max(ckpts, key=os.path.getctime)  # newest file
    checkpoint = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_val_dice = checkpoint['val_dice']
    start_epoch = checkpoint['epoch'] + 1
    print(f"🔄 Resumed from {latest_ckpt}, epoch {checkpoint['epoch']}, val_dice {best_val_dice:.4f}")


# training loop (with AMP + early stopping)
for epoch in range(start_epoch, NUM_EPOCHS + 1):
    model.train()
    running_loss, steps = 0.0, 0
    t0 = time()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} train", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = combined_loss(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        steps += images.size(0)
        pbar.set_postfix(loss=float(loss.item()))

    train_loss = running_loss / steps

    # validation
    model.eval()
    val_loss, vsteps = 0.0, 0
    iou_vals, dice_vals = [], []
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} val", leave=False):
            images, masks = images.to(device), masks.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = combined_loss(outputs, masks)
            val_loss += loss.item() * images.size(0)
            vsteps += images.size(0)

            iou, dice = pixel_iou_and_dice(outputs, masks)
            iou_vals.append(iou); dice_vals.append(dice)

    val_loss = val_loss / vsteps
    mean_iou, mean_dice = float(np.mean(iou_vals)), float(np.mean(dice_vals))
    epoch_time = time() - t0

    scheduler.step(val_loss)
    print(f"Epoch {epoch} | {epoch_time:.1f}s | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | IoU {mean_iou:.4f} | Dice {mean_dice:.4f}")

    # save best
    if mean_dice > best_val_dice:
        best_val_dice = mean_dice
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"unet_best_epoch{epoch}_dice{mean_dice:.4f}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_dice': mean_dice
        }, ckpt_path)
        best_ckpt = ckpt_path
        print("Saved best checkpoint:", ckpt_path)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epochs")

    # early stopping
    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
        break

# final visual check
model.eval()
with torch.no_grad():
    images, masks = next(iter(val_loader))
    outputs = model(images.to(device))
    preds = torch.sigmoid(outputs).cpu().numpy()
    images = images.numpy()
    for i in range(min(3, images.shape[0])):
        img = images[i].transpose(1,2,0)
        gt = masks[i,0].numpy()
        pred = (preds[i,0] > 0.5).astype(np.uint8)
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.imshow(img); plt.title("Image"); plt.axis('off')
        plt.subplot(1,3,2); plt.imshow(gt, cmap='gray'); plt.title("GT"); plt.axis('off')
        plt.subplot(1,3,3); plt.imshow(pred, cmap='gray'); plt.title("Pred"); plt.axis('off')
        plt.show()

print("Best checkpoint saved at:", best_ckpt)
