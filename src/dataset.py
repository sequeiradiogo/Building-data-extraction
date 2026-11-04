"""Datasets that load (image, binary mask) pairs produced by prepare_dataset.py."""
import argparse
import os

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


def find_image_mask_pairs(images_dir, masks_dir, image_exts=(".jpg",), mask_ext=".png"):
    """
    Match every image with the mask of the same name (`123.jpg` <-> `123.png`).
    Images without a mask are skipped with a warning. Returns a sorted list of
    (image_name, mask_name) tuples.
    """
    masks = {f for f in os.listdir(masks_dir) if f.lower().endswith(mask_ext)}
    pairs, missing = [], 0
    for img in sorted(os.listdir(images_dir)):
        if not img.lower().endswith(tuple(image_exts)):
            continue
        mask_name = os.path.splitext(img)[0] + mask_ext
        if mask_name in masks:
            pairs.append((img, mask_name))
        else:
            missing += 1
            print(f"No mask found for image {img}")
    if not pairs:
        raise ValueError(f"No image/mask pairs found in {images_dir} / {masks_dir}")
    print(f"Found {len(pairs)} image/mask pairs" + (f" ({missing} images without mask)" if missing else ""))
    return pairs


def load_binary_mask(path):
    """Load a mask as a uint8 array of 0/1."""
    return (np.array(Image.open(path).convert("L")) > 0).astype(np.uint8)


class JointRandomAugment:
    """
    Applies the SAME random geometric augmentation to an image and its mask
    (both PIL images), so they stay pixel-aligned.
    The image is rotated with bilinear interpolation, the mask with nearest.
    """
    def __init__(self, hflip_p=0.5, vflip_p=0.5, max_rotation=10):
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.max_rotation = max_rotation

    def __call__(self, image, mask):
        if torch.rand(1).item() < self.hflip_p:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if torch.rand(1).item() < self.vflip_p:
            image, mask = TF.vflip(image), TF.vflip(mask)
        if self.max_rotation > 0:
            angle = (torch.rand(1).item() * 2 - 1) * self.max_rotation
            image = TF.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST, fill=0)
        return image, mask


class SegmentationDataset(Dataset):
    """
    torchvision-based dataset (used by the baseline). Images are resized to
    `target_size`, scaled to [0, 1] (no normalization).

    transform:       image-only transform applied after resizing (default ToTensor).
                     Must NOT contain geometric ops, since those would not reach the mask.
    joint_transform: callable(image, mask) -> (image, mask) applied to both PIL images
                     after resizing, e.g. JointRandomAugment.
    """
    def __init__(self, images_dir, masks_dir, transform=None, target_size=(512, 512),
                 joint_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.pairs = find_image_mask_pairs(images_dir, masks_dir)
        self.transform = transform or T.ToTensor()
        self.joint_transform = joint_transform
        self.resize_img = T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR)
        self.resize_mask = T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]
        image = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        mask = Image.fromarray(load_binary_mask(os.path.join(self.masks_dir, mask_name)))

        image = self.resize_img(image)
        mask = self.resize_mask(mask)
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        image = self.transform(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).unsqueeze(0).float()
        return image, mask


class AlbumentationsSegmentationDataset(Dataset):
    """
    Albumentations-based dataset (used by the final model). `transform` receives
    image and mask together, so geometric augmentations stay aligned; it is expected
    to resize, normalize and end with ToTensorV2.
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.pairs = find_image_mask_pairs(images_dir, masks_dir)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]
        img = np.array(Image.open(os.path.join(self.images_dir, img_name)).convert("RGB"))
        mask = load_binary_mask(os.path.join(self.masks_dir, mask_name))

        if self.transform is None:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            return img, torch.from_numpy(mask).unsqueeze(0).float()

        out = self.transform(image=img, mask=mask)
        img, mask = out["image"], out["mask"]
        mask = torch.as_tensor(mask).float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return img.float(), mask


if __name__ == "__main__":
    # Quick sanity check: python src/dataset.py --images data/train/images --masks data/train/masks
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/train/images")
    ap.add_argument("--masks", default="data/train/masks")
    args = ap.parse_args()

    ds = SegmentationDataset(args.images, args.masks, joint_transform=JointRandomAugment())
    img, msk = ds[0]
    print(f"Samples: {len(ds)} | image {tuple(img.shape)} | mask {tuple(msk.shape)} "
          f"| building pixels {msk.mean().item():.1%}")
