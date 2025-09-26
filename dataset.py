import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np

class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for image segmentation.
    Automatically ignores images that don't have a corresponding mask.
    """
    def __init__(self, images_dir, masks_dir, transform=None, target_size=(512, 512)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_size = target_size

        # list all image and mask files
        all_images = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg"))]
        all_masks  = [f for f in os.listdir(masks_dir) if f.lower().endswith(".png")]

        paired_images = []
        paired_masks  = []

        for img in all_images:
            mask_name = os.path.splitext(img)[0] + ".png"
            if mask_name in all_masks:
                paired_images.append(img)
                paired_masks.append(mask_name)
            else:
                print(f"No mask found for image {img}")

        if len(paired_images) == 0:
            raise ValueError("No valid pairs found")

        self.images = sorted(paired_images)
        self.masks  = sorted(paired_masks)

        print(f"Found {len(self.images)} valid image/mask pairs")

        # resize transforms
        self.resize_img  = T.Resize(self.target_size, interpolation=T.InterpolationMode.BILINEAR)
        self.resize_mask = T.Resize(self.target_size, interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load image
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        # load mask
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        mask = Image.open(mask_path)

        # resize
        image = self.resize_img(image)
        mask  = self.resize_mask(mask)

        # convert mask to binary numpy
        mask = np.array(mask, dtype=np.uint8)
        mask = (mask > 0).astype(np.uint8)

        # convert to tensors
        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1,H,W)
        return image, mask



# Quick test (for debugging purposes)

if __name__ == "__main__":
    images_dir = "/content/drive/MyDrive/data/train/images"
    masks_dir  = "/content/drive/MyDrive/data/train/masks"

    transform = T.Compose([
        T.Resize((512,512)),
        T.ToTensor()
    ])

    dataset = SegmentationDataset(images_dir, masks_dir, transform=transform)
    print(f"Number of samples: {len(dataset)}")

    img, msk = dataset[0]
    print(f"Image shape: {img.shape}, Mask shape: {msk.shape}")

    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    images, masks = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}, Batch mask shape: {masks.shape}")
