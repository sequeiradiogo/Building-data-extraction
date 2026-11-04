"""
Baseline: a small 3-level U-Net trained from scratch (kept for comparison with train.py).

    python src/train_baseline.py --data-dir data --checkpoint-dir models

Expects <data-dir>/{train,val}/{images,masks} as produced by prepare_dataset.py.
The best checkpoint (by validation Dice) is saved, and training resumes from the
most recent baseline checkpoint in --checkpoint-dir (disable with --no-resume).
"""
import argparse
import os
from time import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (BASELINE_PREFIX, ensure_drive_mounted, find_checkpoint, get_device,
                    load_checkpoint, save_prediction_grid, seed_everything)
from dataset import JointRandomAugment, SegmentationDataset
from engine import BCEDiceLoss, evaluate, predict_batch, train_one_epoch


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """Minimal U-Net: 3 encoder levels, 2 decoder levels, bilinear upsampling."""
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_ch, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dconv_up2 = DoubleConv(256 + 128, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)
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
        return self.conv_last(x)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint-dir", default="models")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bce-weight", type=float, default=0.5, help="loss = w*BCE + (1-w)*Dice")
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--examples-path", default="output/val_examples_baseline.png")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ensure_drive_mounted(args.data_dir, args.checkpoint_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    seed_everything(args.seed)
    device = get_device()
    use_amp = device.type == "cuda"
    print("Device:", device)

    size = (args.img_size, args.img_size)
    d = args.data_dir
    # geometric augmentation goes through joint_transform so the mask gets the same flip/rotation
    train_ds = SegmentationDataset(f"{d}/train/images", f"{d}/train/masks", target_size=size,
                                   joint_transform=JointRandomAugment(hflip_p=0.5, vflip_p=0.5, max_rotation=10))
    val_ds = SegmentationDataset(f"{d}/val/images", f"{d}/val/masks", target_size=size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=use_amp)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=use_amp)
    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

    model = UNet().to(device)
    criterion = BCEDiceLoss(bce_weight=args.bce_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val_dice, epochs_no_improve, start_epoch, best_ckpt = -1.0, 0, 1, None
    latest = None if args.no_resume else find_checkpoint(args.checkpoint_dir, BASELINE_PREFIX, mode="latest")
    if latest:
        ckpt = load_checkpoint(latest, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_dice = float(ckpt["val_dice"])
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resumed from {latest}, epoch {ckpt['epoch']}, val_dice {best_val_dice:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp,
                                     desc=f"Train {epoch}/{args.epochs}")
        val_loss, m = evaluate(model, val_loader, criterion, device, use_amp, desc=f"Val {epoch}/{args.epochs}")
        scheduler.step(val_loss)
        print(f"Epoch {epoch} | {time() - t0:.1f}s | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"IoU {m['iou']:.4f} | Dice {m['dice']:.4f}")

        if m["dice"] > best_val_dice:
            best_val_dice, epochs_no_improve = m["dice"], 0
            best_ckpt = os.path.join(args.checkpoint_dir, f"{BASELINE_PREFIX}epoch{epoch}_dice{m['dice']:.4f}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": float(val_loss),
                "val_dice": m["dice"],
            }, best_ckpt)
            print("Saved best checkpoint:", best_ckpt)
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")
        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
            break

    images, gts, preds = predict_batch(model, val_loader, device)
    save_prediction_grid(images, gts, preds, args.examples_path, normalized=False)
    print("Best checkpoint of this run:", best_ckpt)


if __name__ == "__main__":
    main()
