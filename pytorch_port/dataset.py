"""
Dataset utilities for loading bi-modality medical images.
Ported from the original data.py to use PyTorch Dataset/DataLoader.

Expected data format:
  - Images: grayscale PNG files, 64x64 pixels
  - Name list: text file with one filename per line
  - Data directory: path where images are stored
"""

import os
import numpy as np
import cv2
import torch
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

        with open(name_list_path, 'r') as f:
            for line in f:
                line = line.strip('\n').strip('\r')
                name = line.split(' ')[0]  # Take first token (filename)
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

        with open(name_list_path, 'r') as f:
            for line in f:
                line = line.strip('\n').strip('\r')
                name = line.split(' ')[0]
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
    """Create a DataLoader for a single modality."""
    dataset = MedicalImageDataset(data_dir, name_list_path, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=True)


def get_paired_dataloader(adc_dir, t2_dir, name_list_path, batch_size=32,
                          image_size=(64, 64), shuffle=True, num_workers=0):
    """Create a DataLoader for paired bi-modality images."""
    dataset = PairedMedicalImageDataset(adc_dir, t2_dir, name_list_path, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=True)
