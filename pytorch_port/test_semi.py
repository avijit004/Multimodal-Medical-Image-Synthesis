import argparse
import os
import numpy as np
import torch
import cv2

from models import SharedLayers, Encoder, GeneratorADC, GeneratorT2
from dataset import get_dataloader
from utils import get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True,
                   help='Path to .pt checkpoint file (e.g. ckpt_9500.pt)')
    p.add_argument('--mode', default='both',
                   choices=['random_pairs', 'real_to_fake', 'both'],
                   help='Inference mode')
    p.add_argument('--n_samples', type=int, default=50,
                   help='Number of images to generate')
    p.add_argument('--output_dir', default='./test_output',
                   help='Directory to save output images')
    p.add_argument('--z_dim', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=32)
    # Required only for real_to_fake mode
    p.add_argument('--adc_dir', default=None,
                   help='Directory of real ADC images (required for real_to_fake mode)')
    p.add_argument('--adc_list', default=None,
                   help='Text file listing ADC image names (required for real_to_fake mode)')
    p.add_argument('--use_cpu', action='store_true')
    return p.parse_args()


def load_models(checkpoint_path, z_dim, device):
    """Load all model weights from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    shared  = SharedLayers().to(device)
    encoder = Encoder(z_dim).to(device)
    gen_adc = GeneratorADC(shared, z_dim).to(device)
    gen_t2  = GeneratorT2(shared).to(device)

    shared.load_state_dict(ckpt['shared'])
    encoder.load_state_dict(ckpt['encoder'])
    gen_adc.load_state_dict(ckpt['gen_adc'])
    gen_t2.load_state_dict(ckpt['gen_t2'])

    shared.eval(); encoder.eval(); gen_adc.eval(); gen_t2.eval()
    print(f"Loaded checkpoint from iter {ckpt.get('iter', '?')} → {checkpoint_path}")
    return shared, encoder, gen_adc, gen_t2


def save_images(images, save_dir, prefix=''):
    """
    Save a batch of images as PNG.
    images: numpy array [N, 1, H, W] in range [-1, 1]
    """
    os.makedirs(save_dir, exist_ok=True)
    for i in range(images.shape[0]):
        img = images[i].squeeze()
        img = ((img + 1) * 255 / 2).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f'{prefix}{i}.png'), img)


def generate_random_pairs(gen_adc, gen_t2, n_samples, z_dim, device, output_dir):
    """
    Mode 1: Sample z ~ N(0,I), generate synthetic ADC and T2 pairs.
    Equivalent to Save_pair() in supervise/test.py.
    """
    print(f"\n[Mode: random_pairs]  Generating {n_samples} synthetic ADC+T2 pairs from noise...")
    adc_dir = os.path.join(output_dir, 'random_pairs', 'adc')
    t2_dir  = os.path.join(output_dir, 'random_pairs', 't2')

    all_adc, all_t2 = [], []
    generated = 0
    with torch.no_grad():
        while generated < n_samples:
            batch = min(32, n_samples - generated)
            z = torch.randn(batch, z_dim, device=device)
            fa = gen_adc(z)
            ft = gen_t2(fa)
            all_adc.append(fa.cpu().numpy())
            all_t2.append(ft.cpu().numpy())
            generated += batch

    all_adc = np.concatenate(all_adc, axis=0)
    all_t2  = np.concatenate(all_t2,  axis=0)

    save_images(all_adc, adc_dir)
    save_images(all_t2,  t2_dir)
    print(f"  Saved {n_samples} ADC images → {adc_dir}")
    print(f"  Saved {n_samples} T2  images → {t2_dir}")


def real_to_fake_translation(encoder, gen_adc, gen_t2, adc_dir, adc_list,
                              n_samples, batch_size, device, output_dir):
    """
    Mode 2: Real ADC -> Encoder -> z -> GeneratorADC (reconstructed ADC)
                                     -> GeneratorT2  (synthesized T2)
    Equivalent to Save_real_to_fake_pair() in supervise/test.py.
    This is the clinically meaningful use case: given a real ADC scan,
    produce the corresponding T2-weighted image.
    """
    print(f"\n[Mode: real_to_fake]  Translating {n_samples} real ADC images...")
    recon_adc_dir = os.path.join(output_dir, 'real_to_fake', 'reconstructed_adc')
    synth_t2_dir  = os.path.join(output_dir, 'real_to_fake', 'synthesized_t2')
    real_adc_dir  = os.path.join(output_dir, 'real_to_fake', 'input_adc')

    loader = get_dataloader(adc_dir, adc_list, batch_size=batch_size, shuffle=False)

    all_real, all_recon_adc, all_synth_t2 = [], [], []
    collected = 0

    with torch.no_grad():
        for batch in loader:
            if collected >= n_samples:
                break
            remaining = n_samples - collected
            batch = batch[:remaining].to(device)

            z_enc    = encoder(batch)          # real ADC → latent z
            rec_adc  = gen_adc(z_enc)          # z → reconstructed ADC
            synth_t2 = gen_t2(rec_adc)         # reconstructed ADC → synthesized T2

            all_real.append(batch.cpu().numpy())
            all_recon_adc.append(rec_adc.cpu().numpy())
            all_synth_t2.append(synth_t2.cpu().numpy())
            collected += batch.size(0)

    all_real      = np.concatenate(all_real,      axis=0)
    all_recon_adc = np.concatenate(all_recon_adc, axis=0)
    all_synth_t2  = np.concatenate(all_synth_t2,  axis=0)

    save_images(all_real,      real_adc_dir,  prefix='real_adc_')
    save_images(all_recon_adc, recon_adc_dir, prefix='recon_adc_')
    save_images(all_synth_t2,  synth_t2_dir,  prefix='synth_t2_')
    print(f"  Input real ADC          → {real_adc_dir}")
    print(f"  Reconstructed ADC       → {recon_adc_dir}")
    print(f"  Synthesized T2          → {synth_t2_dir}")


def main():
    args = parse_args()
    device = torch.device('cpu') if args.use_cpu else get_device()
    print(f"Using device: {device}")

    if args.mode in ('real_to_fake', 'both'):
        if not args.adc_dir or not args.adc_list:
            raise ValueError("--adc_dir and --adc_list are required for real_to_fake mode")

    shared, encoder, gen_adc, gen_t2 = load_models(args.checkpoint, args.z_dim, device)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode in ('random_pairs', 'both'):
        generate_random_pairs(gen_adc, gen_t2, args.n_samples,
                              args.z_dim, device, args.output_dir)

    if args.mode in ('real_to_fake', 'both'):
        real_to_fake_translation(encoder, gen_adc, gen_t2,
                                 args.adc_dir, args.adc_list,
                                 args.n_samples, args.batch_size,
                                 device, args.output_dir)

    print(f"\nDone. All outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
