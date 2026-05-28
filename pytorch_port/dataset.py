"""
Dataset utilities for loading bi-modality medical images.
Ported from the original data.py to use PyTorch Dataset/DataLoader.

Expected data format:
  - Images: grayscale PNG files, 64x64 pixels
  - Name list: text file with one filename per line
  - Data directory: path where images are stored
"""

import os
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MedicalImageDataset(Dataset):
    """
    Dataset for loading grayscale medical images from a directory.
    Used for both ADC and T2w modalities.

    Args:
        data_dir: path to directory containing images
        name_list_path: path to text file listing image filenames (one per line)
        image_size: tuple (H, W) for resizing
    """
    def __init__(self, data_dir, name_list_path, image_size=(64, 64)):
        self.data_dir = data_dir
        self.image_size = image_size
        self.name_list = []

        with open(name_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip().split(' ')[0]  # strip handles \n, \r\n, \r
                if name:
                    self.name_list.append(name)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.name_list[idx])
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)

        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert to tensor: [H, W] -> [1, H, W], int values [0, 255]
        image = torch.from_numpy(image).unsqueeze(0).float()

        # Normalize to [-1, 1]
        image = 2 * (image / 255.0) - 1.0

        return image


class PairedMedicalImageDataset(Dataset):
    """
    Dataset for loading paired bi-modality images (ADC + T2w).
    Assumes both modalities share the same name list (paired by filename).

    Args:
        adc_dir: path to ADC image directory
        t2_dir: path to T2w image directory
        name_list_path: path to text file listing image filenames
        image_size: tuple (H, W) for resizing
    """
    def __init__(self, adc_dir, t2_dir, name_list_path, image_size=(64, 64)):
        self.adc_dir = adc_dir
        self.t2_dir = t2_dir
        self.image_size = image_size
        self.name_list = []

        with open(name_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip().split(' ')[0]
                if name:
                    self.name_list.append(name)

    def __len__(self):
        return len(self.name_list)

    def _load_image(self, directory, filename):
        path = os.path.join(directory, filename)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = torch.from_numpy(image).unsqueeze(0).float()
        image = 2 * (image / 255.0) - 1.0
        return image

    def __getitem__(self, idx):
        name = self.name_list[idx]
        adc_img = self._load_image(self.adc_dir, name)
        t2_img = self._load_image(self.t2_dir, name)
        return adc_img, t2_img


def get_dataloader(data_dir, name_list_path, batch_size=32, image_size=(64, 64),
                   shuffle=True, num_workers=0):
    """
    Create a DataLoader for a single modality.
    num_workers=0 is required on Windows (safe on all platforms).
    """
    dataset = MedicalImageDataset(data_dir, name_list_path, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=True,
                      persistent_workers=False)


def get_paired_dataloader(adc_dir, t2_dir, name_list_path, batch_size=32,
                          image_size=(64, 64), shuffle=True, num_workers=0):
    """
    Create a DataLoader for paired bi-modality images.
    num_workers=0 is required on Windows (safe on all platforms).
    """
    dataset = PairedMedicalImageDataset(adc_dir, t2_dir, name_list_path, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=True,
                      persistent_workers=False)


# ---------------------------------------------------------------------------
# Self-supervised datasets
# ---------------------------------------------------------------------------

class AugmentedPairedDataset(Dataset):
    """
    Paired ADC+T2 dataset with stochastic augmentations for self-supervised
    contrastive pretraining.  Each call to __getitem__ applies independent
    random augmentations to the ADC and T2 images, so two passes over the
    same index produce different views — which is the key property needed by
    the NT-Xent loss.

    Augmentations (conservative for 64x64 medical images):
      - Random horizontal flip  (p=0.5)
      - Random vertical flip    (p=0.5)
      - Random rotation         (±15°, nearest-fill)
      - Random Gaussian noise   (σ ~ U[0, 0.05])
    """

    def __init__(self, adc_dir, t2_dir, name_list_path, image_size=(64, 64)):
        self.adc_dir = adc_dir
        self.t2_dir = t2_dir
        self.image_size = image_size
        self.name_list = []

        with open(name_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip().split(' ')[0]
                if name:
                    self.name_list.append(name)

    def __len__(self):
        return len(self.name_list)

    def _load(self, directory, filename):
        path = os.path.join(directory, filename)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        if img.shape[:2] != self.image_size:
            img = cv2.resize(img, self.image_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tensor = torch.from_numpy(img).unsqueeze(0).float()
        tensor = 2.0 * (tensor / 255.0) - 1.0   # normalise to [-1, 1]
        return tensor

    def _augment(self, tensor):
        """Apply random spatial + noise augmentations to a [1, H, W] tensor."""
        # Horizontal flip
        if random.random() < 0.5:
            tensor = torch.flip(tensor, dims=[2])
        # Vertical flip
        if random.random() < 0.5:
            tensor = torch.flip(tensor, dims=[1])
        # Rotation ±15° via affine grid
        angle = random.uniform(-15.0, 15.0)
        rad = angle * 3.14159265 / 180.0
        cos_a, sin_a = float(torch.cos(torch.tensor(rad))), float(torch.sin(torch.tensor(rad)))
        theta = torch.tensor([[cos_a, -sin_a, 0.0],
                               [sin_a,  cos_a, 0.0]], dtype=torch.float32)
        grid = F.affine_grid(theta.unsqueeze(0),
                             tensor.unsqueeze(0).size(), align_corners=False)
        tensor = F.grid_sample(tensor.unsqueeze(0), grid,
                                mode='bilinear', padding_mode='zeros',
                                align_corners=False).squeeze(0)
        # Additive Gaussian noise
        sigma = random.uniform(0.0, 0.05)
        tensor = (tensor + sigma * torch.randn_like(tensor)).clamp(-1.0, 1.0)
        return tensor

    def __getitem__(self, idx):
        name = self.name_list[idx]
        adc = self._augment(self._load(self.adc_dir, name))
        t2  = self._augment(self._load(self.t2_dir,  name))
        return adc, t2


class LabeledPairedDataset(Dataset):
    """
    Paired ADC+T2 dataset with per-sample labels for downstream evaluation.

    Name-list format (one entry per line):
        filename label
    where label is an integer class index (e.g. 0 = nonCS, 1 = CS).
    """

    def __init__(self, adc_dir, t2_dir, name_list_path, image_size=(64, 64)):
        self.adc_dir = adc_dir
        self.t2_dir = t2_dir
        self.image_size = image_size
        self.name_list = []
        self.label_list = []

        with open(name_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.name_list.append(parts[0])
                    self.label_list.append(int(parts[1]))

    def __len__(self):
        return len(self.name_list)

    def _load(self, directory, filename):
        path = os.path.join(directory, filename)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        if img.shape[:2] != self.image_size:
            img = cv2.resize(img, self.image_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tensor = torch.from_numpy(img).unsqueeze(0).float()
        tensor = 2.0 * (tensor / 255.0) - 1.0
        return tensor

    def __getitem__(self, idx):
        name = self.name_list[idx]
        adc   = self._load(self.adc_dir, name)
        t2    = self._load(self.t2_dir,  name)
        label = torch.tensor(self.label_list[idx], dtype=torch.long)
        return adc, t2, label
