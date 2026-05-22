"""
Utility functions for training:
  - WGAN-GP gradient penalty
  - Image saving
  - Results folder creation
  - Infinite data loader wrapper
"""

import os
import datetime
import numpy as np
import cv2
import torch
import torch.autograd as autograd


def compute_gradient_penalty(discriminator, real_data, fake_data, device, lambda_gp=10):
    """
    Compute gradient penalty for WGAN-GP.
    Enforces the 1-Lipschitz constraint on the discriminator.

    Args:
        discriminator: critic network
        real_data: batch of real images [B, 1, H, W]
        fake_data: batch of generated images [B, 1, H, W]
        device: torch device
        lambda_gp: gradient penalty coefficient (default: 10)

    Returns:
        gradient_penalty: scalar tensor
    """
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_data)

    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    d_interpolates = discriminator(interpolates)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def save_generated_images(images, save_dir, prefix=""):
    """
    Save a batch of generated images as PNG files.

    Args:
        images: numpy array [N, 1, H, W] in range [-1, 1]
        save_dir: directory to save images
        prefix: optional filename prefix
    """
    os.makedirs(save_dir, exist_ok=True)
    for i in range(images.shape[0]):
        img = images[i].squeeze()  # [H, W]
        # Convert from [-1, 1] to [0, 255]
        img = ((img + 1) * 255 / 2).clip(0, 255).astype(np.uint8)
        filename = os.path.join(save_dir, f"{prefix}{i}.png")
        cv2.imwrite(filename, img)


def form_results(results_path):
    """
    Create timestamped folders for tensorboard logs, saved models, and log files.

    Returns:
        tuple: (tensorboard_path, saved_model_path, log_path)
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    folder = os.path.join(results_path, timestamp)
    tensorboard_path = os.path.join(folder, 'Tensorboard')
    saved_model_path = os.path.join(folder, 'Saved_models')
    log_path = os.path.join(folder, 'log')

    os.makedirs(tensorboard_path, exist_ok=True)
    os.makedirs(saved_model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    return tensorboard_path, saved_model_path, log_path


def infinite_dataloader(dataloader):
    """
    Wrap a DataLoader to yield batches infinitely (cycle through epochs).
    """
    while True:
        for batch in dataloader:
            yield batch


def get_device():
    """Get the best available device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
