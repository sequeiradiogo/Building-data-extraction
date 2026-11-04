"""Losses, metrics and the train / validation loops shared by both training scripts."""
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# --------------------------------------------------------------------- losses
def dice_loss_logits(pred_logits, target, eps=1e-6):
    """Soft Dice loss computed per image, averaged over the batch."""
    pred = torch.sigmoid(pred_logits)
    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2. * intersection + eps) / (union + eps)
    return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """bce_weight * BCE + (1 - bce_weight) * Dice. Lower bce_weight -> more weight on Dice."""
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, target):
        return (self.bce_weight * self.bce(pred_logits, target)
                + (1 - self.bce_weight) * dice_loss_logits(pred_logits, target))


# -------------------------------------------------------------------- metrics
def per_image_metrics(pred_logits, target, thresh=0.5):
    """
    Pixel-level metrics for each image of a batch.
    Returns a dict of lists: iou, dice, precision, recall, f1 (dice == f1 for binary masks).
    An image with an empty prediction AND empty ground truth scores 1.0 on everything.
    """
    pred = (torch.sigmoid(pred_logits) > thresh).cpu().numpy().astype(bool)
    gt = target.cpu().numpy() > 0.5
    out = {k: [] for k in ("iou", "dice", "precision", "recall", "f1")}
    for p, g in zip(pred[:, 0], gt[:, 0]):
        tp = np.logical_and(p, g).sum()
        fp = np.logical_and(p, ~g).sum()
        fn = np.logical_and(~p, g).sum()
        if tp + fp + fn == 0:
            vals = dict(iou=1.0, dice=1.0, precision=1.0, recall=1.0, f1=1.0)
        else:
            dice = 2 * tp / (2 * tp + fp + fn)
            vals = dict(iou=tp / (tp + fp + fn),
                        dice=dice,
                        precision=tp / (tp + fp) if tp + fp > 0 else 0.0,
                        recall=tp / (tp + fn) if tp + fn > 0 else 0.0,
                        f1=dice)
        for k, v in vals.items():
            out[k].append(float(v))
    return out


# ---------------------------------------------------------------------- loops
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp,
                    batch_scheduler=None, desc="train"):
    """One training epoch. `batch_scheduler` (e.g. OneCycleLR) is stepped after every batch."""
    model.train()
    running_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if batch_scheduler is not None:
            batch_scheduler.step()

        running_loss += loss.item() * images.size(0)
        n += images.size(0)
        pbar.set_postfix(loss=float(loss.item()))
    return running_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, desc="val"):
    """Returns (mean loss, dict of mean per-image metrics)."""
    model.eval()
    total_loss, n = 0.0, 0
    metrics = {k: [] for k in ("iou", "dice", "precision", "recall", "f1")}
    for images, masks in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        for k, v in per_image_metrics(outputs.float(), masks).items():
            metrics[k].extend(v)
    return total_loss / max(n, 1), {k: float(np.mean(v)) for k, v in metrics.items()}


@torch.no_grad()
def predict_batch(model, loader, device, thresh=0.5):
    """First batch of `loader`: (images, ground-truth masks, binary predictions) as numpy."""
    model.eval()
    images, masks = next(iter(loader))
    probs = torch.sigmoid(model(images.to(device)).float()).cpu().numpy()
    return images.numpy(), masks[:, 0].numpy(), (probs[:, 0] > thresh).astype(np.uint8)
