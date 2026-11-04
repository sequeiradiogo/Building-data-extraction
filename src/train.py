"""
Train the final model: U-Net with an ImageNet-pretrained ResNet-34 encoder.

    python src/train.py --data-dir data --checkpoint-dir models

Expects <data-dir>/{train,val}/{images,masks} as produced by prepare_dataset.py.
A checkpoint is written after every epoch and training resumes automatically from
the most recent one in --checkpoint-dir (disable with --no-resume).
"""
import argparse
import os
import warnings
from time import time

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from common import (FINAL_PREFIX, IMAGENET_MEAN, IMAGENET_STD, build_unet, ensure_drive_mounted,
                    find_checkpoint, get_device, iou_from_filename, load_checkpoint,
                    save_prediction_grid, seed_everything)
from dataset import AlbumentationsSegmentationDataset
from engine import BCEDiceLoss, evaluate, predict_batch, train_one_epoch


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint-dir", default="models")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--img-size", type=int, default=512, help="images are resized to img-size x img-size")
    p.add_argument("--lr", type=float, default=5e-4, help="initial AdamW learning rate")
    p.add_argument("--max-lr", type=float, default=1e-3, help="peak learning rate of OneCycleLR")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--bce-weight", type=float, default=0.3, help="loss = w*BCE + (1-w)*Dice")
    p.add_argument("--patience", type=int, default=7, help="early stopping patience (epochs)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--encoder-weights", default="imagenet", help="'imagenet' or 'none'")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-resume", action="store_true", help="start from scratch even if checkpoints exist")
    p.add_argument("--examples-path", default="output/val_examples.png",
                   help="where to save a figure with predictions on validation images")
    return p.parse_args(argv)


def get_augmentations(img_size):
    train_aug = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.2),
        A.Rotate(limit=12, p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.OneOf([A.MotionBlur(p=0.3), A.GaussianBlur(p=0.3), A.GaussNoise(p=0.3)], p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    val_aug = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return train_aug, val_aug


def resume(path, model, optimizer, scheduler, scaler, steps_per_epoch):
    """
    Restore the full training state from a checkpoint.
    Returns (start_epoch, best_val_iou, epochs_no_improve).
    """
    print("Resuming from checkpoint:", path)
    ckpt = load_checkpoint(path, map_location="cpu")
    if not (isinstance(ckpt, dict) and "model_state_dict" in ckpt):
        # weights-only file (e.g. the released model): fine-tune from it, fresh training state
        model.load_state_dict(ckpt, strict=True)
        print("Weights-only checkpoint: loaded model weights, training state starts fresh.")
        return 1, -1.0, 0

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    for name, obj in (("optimizer", optimizer), ("scaler", scaler)):
        key = f"{name}_state_dict"
        try:
            if key in ckpt:
                obj.load_state_dict(ckpt[key])
        except Exception as e:
            print(f"Could not restore {name} state ({e}); reinitializing it.")

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best = ckpt.get("best_val_iou", ckpt.get("val_iou", iou_from_filename(path)))
    best_val_iou = float(best) if best is not None else -1.0
    epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))

    sched_state = ckpt.get("scheduler_state_dict")
    if sched_state is not None and sched_state.get("total_steps") == scheduler.total_steps:
        scheduler.load_state_dict(sched_state)
    else:
        # Older checkpoints did not store the scheduler (or --epochs / dataset size changed,
        # so the saved schedule has a different length): fast-forward OneCycleLR instead.
        n_steps = min((start_epoch - 1) * steps_per_epoch, scheduler.total_steps - 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(n_steps):
                scheduler.step()
        print(f"Scheduler state not restored; fast-forwarded OneCycleLR by {n_steps} steps.")

    print(f"Resuming at epoch {start_epoch} | best IoU so far {best_val_iou:.4f} | "
          f"epochs without improvement {epochs_no_improve}")
    return start_epoch, best_val_iou, epochs_no_improve


def main(argv=None):
    args = parse_args(argv)
    ensure_drive_mounted(args.data_dir, args.checkpoint_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    seed_everything(args.seed)
    device = get_device()
    use_amp = device.type == "cuda"
    print("Device:", device)

    # data
    train_aug, val_aug = get_augmentations(args.img_size)
    d = args.data_dir
    train_ds = AlbumentationsSegmentationDataset(f"{d}/train/images", f"{d}/train/masks", transform=train_aug)
    val_ds = AlbumentationsSegmentationDataset(f"{d}/val/images", f"{d}/val/masks", transform=val_aug)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=use_amp)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=use_amp)
    print("Train pairs:", len(train_ds), "| Val pairs:", len(val_ds))

    # model, loss, optimizer, schedule
    encoder_weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights
    model = build_unet(encoder_weights).to(device)
    criterion = BCEDiceLoss(bce_weight=args.bce_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.max_lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs,
        pct_start=0.3, anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch, best_val_iou, epochs_no_improve = 1, -1.0, 0
    latest = None if args.no_resume else find_checkpoint(args.checkpoint_dir, FINAL_PREFIX, mode="latest")
    if latest:
        model.to("cpu")
        start_epoch, best_val_iou, epochs_no_improve = resume(
            latest, model, optimizer, scheduler, scaler, steps_per_epoch)
        model.to(device)
    else:
        print("No checkpoint to resume from; training from scratch.")

    best_ckpt = None
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp,
                                     batch_scheduler=scheduler, desc=f"Train {epoch}/{args.epochs}")
        val_loss, m = evaluate(model, val_loader, criterion, device, use_amp, desc=f"Val {epoch}/{args.epochs}")
        print(f"Epoch {epoch} | {time() - t0:.1f}s | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"IoU {m['iou']:.4f} | Precision {m['precision']:.4f} | Recall {m['recall']:.4f} | F1 {m['f1']:.4f}")

        # early-stopping bookkeeping first, so the checkpoint stores it
        improved = m["iou"] > best_val_iou
        if improved:
            best_val_iou, epochs_no_improve = m["iou"], 0
        else:
            epochs_no_improve += 1

        ckpt_path = os.path.join(args.checkpoint_dir, f"{FINAL_PREFIX}epoch{epoch}_IoU{m['iou']:.4f}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_loss": float(val_loss),
            "val_iou": m["iou"],
            "best_val_iou": float(best_val_iou),
            "epochs_no_improve": epochs_no_improve,
        }, ckpt_path)

        if improved:
            best_ckpt = ckpt_path
            print("New best model:", ckpt_path)
        else:
            print(f"No improvement for {epochs_no_improve} epochs (best {best_val_iou:.4f})")
        if epochs_no_improve >= args.patience:
            print(f"Early stopping (no improvement for {args.patience} epochs).")
            break

    images, gts, preds = predict_batch(model, val_loader, device)
    save_prediction_grid(images, gts, preds, args.examples_path, normalized=True)
    print("Done. Best checkpoint of this run:", best_ckpt)


if __name__ == "__main__":
    main()
