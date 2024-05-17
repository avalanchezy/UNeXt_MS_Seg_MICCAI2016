import os
import cv2
import numpy as np
import torch
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Initialize the dataset with image and mask information.

        Args:
            img_ids (list): List of image identifiers without file extension.
            img_dir (str): Directory containing images.
            mask_dir (str): Base directory containing mask subdirectories.
            img_ext (str): Extension of image files.
            mask_ext (str): Extension of mask files.
            num_classes (int): Number of mask classes/categories.
            transform (callable, optional): A function/transform that takes in an image and mask
                and returns a transformed version. E.g., data augmentation methods from albumentations.
        
        Directory structure expected:
            <root_dir>
            ├── images
            │   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            └── masks
                ├── 0
                │   ├── 0a7e06.png
                │   ├── 0aab0a.png
                │   ├── 0b1761.png
                │   ├── ...
                ├── 1
                │   ├── 0a7e06.png
                │   ├── 0aab0a.png
                │   ├── 0b1761.png
                │   ├── ...
                ...
        """
        self.img_ids = [os.path.basename(img_id) for img_id in img_ids]  # Only use filenames
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"The image file {img_path} could not be loaded.")

        masks = []
        for i in range(self.num_classes):
            mask_path = os.path.join(self.mask_dir, str(i), img_id + self.mask_ext)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"The mask file {mask_path} could not be loaded.")
            masks.append(mask[..., None])
        
        mask = np.dstack(masks)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)  # From HxWxC to CxHxW for PyTorch compatibility
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)  # Same as above, also ensuring masks are properly shaped

        return img, mask, {'img_id': img_id}

