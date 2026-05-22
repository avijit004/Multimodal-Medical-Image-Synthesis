"""
Evaluation Metrics for Semi-Supervised Sequential GAN.
Results are written to TensorBoard AND saved as CSV.

TensorBoard shows:
  Scalars  : MAE, MSE, PSNR, SSIM, FID  (per checkpoint iteration)
  Images   : Real ADC | Recon ADC | Real T2 | Synth T2  (side-by-side grids)
             Random generated ADC + T2 pairs
  Histogram: Per-image PSNR and SSIM distributions

Usage:
    python evaluate.py \
        --checkpoint ./results_semi_real/2026-05-17__01-56-42/Saved_models/ckpt_9500.pt \
        --adc_dir ../data/adc --t2_dir ../data/t2 \
        --paired_list ../data/paired_names.txt \
        --adc_list ../data/adc_names.txt --t2_list ../data/t2_names.txt \
        --output_dir ./eval_output

    Then run:
        tensorboard --logdir ./eval_output/tensorboard
        Open: http://localhost:6006
"""

import argparse, os, csv
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import cv2
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from scipy.linalg import sqrtm
import torchvision.models as tv_models
import torchvision.transforms as T

from models import SharedLayers, Encoder, GeneratorADC, GeneratorT2
from dataset import get_dataloader, get_paired_dataloader
from utils import get_device


# ─── helpers ──────────────────────────────────────────────────────────────────

def tensor_to_uint8(t):
    """[B,1,H,W] in [-1,1]  →  numpy [B,H,W] uint8 [0,255]"""
    arr = t.cpu().numpy()
    arr = ((arr + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return arr[:, 0, :, :]


def to_grid_tensor(t, nrow=8):
    """[B,1,H,W] in [-1,1]  →  display-ready [3,H',W'] in [0,1]"""
    t_rgb = t.repeat(1, 3, 1, 1)                    # grey → RGB
    grid  = vutils.make_grid(t_rgb, nrow=nrow,
                              normalize=True, value_range=(-1, 1))
    return grid                                      # [3,H',W']


def paired_metrics(real_batch, fake_batch):
    mae_l, mse_l, psnr_l, ssim_l = [], [], [], []
    for r, f in zip(real_batch, fake_batch):
        rf, ff = r.astype(np.float32), f.astype(np.float32)
        mae_l.append(float(np.mean(np.abs(rf - ff))))
        mse_l.append(float(np.mean((rf - ff) ** 2)))
        psnr_l.append(float(psnr_fn(r, f, data_range=255)))
        ssim_l.append(float(ssim_fn(r, f, data_range=255)))
    return {'mae': mae_l, 'mse': mse_l, 'psnr': psnr_l, 'ssim': ssim_l}


def summarise(d):
    return {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
            for k, v in d.items()}


# ─── FID ──────────────────────────────────────────────────────────────────────

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        inc = tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.DEFAULT)
        inc.fc = nn.Identity()
        inc.aux_logits = False
        self.model = inc.to(device).eval()
        self.device = device
        self.tfm = T.Compose([
            T.Lambda(lambda x: x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x),
            T.Resize((299, 299), antialias=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, imgs):
        x = (imgs + 1) / 2.0
        x = self.tfm(x)
        return self.model(x).cpu().numpy()


def compute_fid(f_real, f_fake):
    mu1, s1 = f_real.mean(0), np.cov(f_real, rowvar=False)
    mu2, s2 = f_fake.mean(0), np.cov(f_fake, rowvar=False)
    # add small epsilon for numerical stability with small N
    eps = np.eye(s1.shape[0]) * 1e-6
    covmean, _ = sqrtm((s1 + eps) @ (s2 + eps), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(s1 + s2 - 2 * covmean))


def collect_real_features(extractor, loader, max_n):
    feats = []
    n = 0
    for batch in loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        feats.append(extractor.extract(imgs.to(extractor.device)))
        n += feats[-1].shape[0]
        if n >= max_n:
            break
    return np.concatenate(feats)[:max_n]


def collect_gen_features(extractor, gen_fn, n, z_dim, bs, device):
    feats = []
    done = 0
    while done < n:
        b = min(bs, n - done)
        z = torch.randn(b, z_dim, device=device)
        with torch.no_grad():
            imgs = gen_fn(z)
        feats.append(extractor.extract(imgs))
        done += b
    return np.concatenate(feats)[:n]


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--adc_dir',     required=True)
    p.add_argument('--t2_dir',      required=True)
    p.add_argument('--paired_list', required=True)
    p.add_argument('--adc_list',    required=True)
    p.add_argument('--t2_list',     required=True)
    p.add_argument('--output_dir',  default='./eval_output')
    p.add_argument('--z_dim',       type=int, default=128)
    p.add_argument('--batch_size',  type=int, default=32)
    p.add_argument('--fid_samples', type=int, default=200)
    p.add_argument('--vis_images',  type=int, default=16,
                   help='Number of image pairs to show in TensorBoard grids')
    p.add_argument('--use_cpu',     action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cpu') if args.use_cpu else get_device()
    print(f"\nDevice: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    tb_dir = os.path.join(args.output_dir, 'tensorboard')
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"TensorBoard log dir: {tb_dir}")
    print(f"  → run: tensorboard --logdir {tb_dir}")
    print(f"  → open: http://localhost:6006\n")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt    = torch.load(args.checkpoint, map_location=device)
    ckpt_it = ckpt.get('iter', 0)
    shared  = SharedLayers().to(device)
    encoder = Encoder(args.z_dim).to(device)
    gen_adc = GeneratorADC(shared, args.z_dim).to(device)
    gen_t2  = GeneratorT2(shared).to(device)
    shared.load_state_dict(ckpt['shared'])
    encoder.load_state_dict(ckpt['encoder'])
    gen_adc.load_state_dict(ckpt['gen_adc'])
    gen_t2.load_state_dict(ckpt['gen_t2'])
    shared.eval(); encoder.eval(); gen_adc.eval(); gen_t2.eval()
    print(f"Loaded checkpoint — iter {ckpt_it}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1  –  PAIRED METRICS
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("PART 1 — PAIRED METRICS")
    print("=" * 60)

    paired_loader = get_paired_dataloader(
        args.adc_dir, args.t2_dir, args.paired_list,
        batch_size=args.batch_size, shuffle=False
    )

    adc_m = {'mae': [], 'mse': [], 'psnr': [], 'ssim': []}
    t2_m  = {'mae': [], 'mse': [], 'psnr': [], 'ssim': []}

    # Collect a few batches for the image grids
    vis_real_adc, vis_rec_adc, vis_real_t2, vis_syn_t2 = [], [], [], []
    vis_collected = 0

    with torch.no_grad():
        for real_adc, real_t2 in paired_loader:
            real_adc = real_adc.to(device)
            real_t2  = real_t2.to(device)
            z_enc    = encoder(real_adc)
            rec_adc  = gen_adc(z_enc)
            syn_t2   = gen_t2(rec_adc)

            # Compute metrics
            m_a = paired_metrics(tensor_to_uint8(real_adc), tensor_to_uint8(rec_adc))
            m_t = paired_metrics(tensor_to_uint8(real_t2),  tensor_to_uint8(syn_t2))
            for k in adc_m:
                adc_m[k].extend(m_a[k])
                t2_m[k].extend(m_t[k])

            # Collect images for visualisation
            if vis_collected < args.vis_images:
                take = min(args.vis_images - vis_collected, real_adc.size(0))
                vis_real_adc.append(real_adc[:take].cpu())
                vis_rec_adc.append(rec_adc[:take].cpu())
                vis_real_t2.append(real_t2[:take].cpu())
                vis_syn_t2.append(syn_t2[:take].cpu())
                vis_collected += take

    adc_s = summarise(adc_m)
    t2_s  = summarise(t2_m)

    # ── Log scalars ────────────────────────────────────────────────────────────
    for metric in ['mae', 'mse', 'psnr', 'ssim']:
        writer.add_scalar(f'Metrics_ADC_Reconstruction/{metric.upper()}',
                          adc_s[metric]['mean'], ckpt_it)
        writer.add_scalar(f'Metrics_T2_Synthesis/{metric.upper()}',
                          t2_s[metric]['mean'], ckpt_it)

    # ── Log histograms (per-image score distributions) ─────────────────────────
    writer.add_histogram('Distribution/ADC_PSNR', np.array(adc_m['psnr']), ckpt_it)
    writer.add_histogram('Distribution/ADC_SSIM', np.array(adc_m['ssim']), ckpt_it)
    writer.add_histogram('Distribution/T2_PSNR',  np.array(t2_m['psnr']),  ckpt_it)
    writer.add_histogram('Distribution/T2_SSIM',  np.array(t2_m['ssim']),  ckpt_it)

    # ── Log image grids ────────────────────────────────────────────────────────
    nrow = min(8, args.vis_images)

    vis_real_adc = torch.cat(vis_real_adc)
    vis_rec_adc  = torch.cat(vis_rec_adc)
    vis_real_t2  = torch.cat(vis_real_t2)
    vis_syn_t2   = torch.cat(vis_syn_t2)

    # Individual grids per type
    writer.add_image('ADC/1_Real',          to_grid_tensor(vis_real_adc, nrow), ckpt_it)
    writer.add_image('ADC/2_Reconstructed', to_grid_tensor(vis_rec_adc,  nrow), ckpt_it)
    writer.add_image('T2/1_Real',           to_grid_tensor(vis_real_t2,  nrow), ckpt_it)
    writer.add_image('T2/2_Synthesized',    to_grid_tensor(vis_syn_t2,   nrow), ckpt_it)

    # Side-by-side interleaved comparison grids  [real, fake, real, fake, ...]
    interleave_adc = torch.stack(
        [x for pair in zip(vis_real_adc, vis_rec_adc) for x in pair]
    )
    interleave_t2 = torch.stack(
        [x for pair in zip(vis_real_t2, vis_syn_t2) for x in pair]
    )
    writer.add_image('Comparison/ADC_Real_vs_Reconstructed',
                     to_grid_tensor(interleave_adc, nrow=2), ckpt_it)
    writer.add_image('Comparison/T2_Real_vs_Synthesized',
                     to_grid_tensor(interleave_t2,  nrow=2), ckpt_it)

    print(f"\n  ADC Reconstruction — PSNR: {adc_s['psnr']['mean']:.2f} dB  "
          f"SSIM: {adc_s['ssim']['mean']:.4f}")
    print(f"  T2 Synthesis       — PSNR: {t2_s['psnr']['mean']:.2f} dB  "
          f"SSIM: {t2_s['ssim']['mean']:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2  –  RANDOM GENERATION  (visualise only, no metric)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART 2 — RANDOM GENERATION VISUALISATION")
    print("=" * 60)

    with torch.no_grad():
        z_rand   = torch.randn(args.vis_images, args.z_dim, device=device)
        rand_adc = gen_adc(z_rand).cpu()
        rand_t2  = gen_t2(gen_adc(z_rand)).cpu()

    writer.add_image('Generated/ADC_from_noise', to_grid_tensor(rand_adc, nrow), ckpt_it)
    writer.add_image('Generated/T2_from_noise',  to_grid_tensor(rand_t2,  nrow), ckpt_it)

    interleave_rand = torch.stack(
        [x for pair in zip(rand_adc, rand_t2) for x in pair]
    )
    writer.add_image('Generated/ADC_T2_pairs',
                     to_grid_tensor(interleave_rand, nrow=2), ckpt_it)

    print(f"  Logged {args.vis_images} random ADC+T2 pairs.")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3  –  FID
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"PART 3 — FID  ({args.fid_samples} samples each side)")
    print("=" * 60)
    print("  Loading InceptionV3...")

    extractor = InceptionFeatureExtractor(device)

    print("  Real ADC features...")
    f_real_adc = collect_real_features(
        extractor,
        get_dataloader(args.adc_dir, args.adc_list,
                       batch_size=args.batch_size, shuffle=False),
        args.fid_samples
    )
    print("  Generated ADC features...")
    f_fake_adc = collect_gen_features(
        extractor, gen_adc, args.fid_samples, args.z_dim, args.batch_size, device)

    fid_adc = compute_fid(f_real_adc, f_fake_adc)
    print(f"\n  FID (ADC) : {fid_adc:.2f}")

    print("  Real T2 features...")
    f_real_t2 = collect_real_features(
        extractor,
        get_dataloader(args.t2_dir, args.t2_list,
                       batch_size=args.batch_size, shuffle=False),
        args.fid_samples
    )
    print("  Generated T2 features...")
    f_fake_t2 = collect_gen_features(
        extractor,
        lambda z: gen_t2(gen_adc(z)),
        args.fid_samples, args.z_dim, args.batch_size, device
    )

    fid_t2 = compute_fid(f_real_t2, f_fake_t2)
    print(f"  FID (T2)  : {fid_t2:.2f}")

    writer.add_scalar('Metrics_ADC_Reconstruction/FID', fid_adc, ckpt_it)
    writer.add_scalar('Metrics_T2_Synthesis/FID',       fid_t2,  ckpt_it)

    # ══════════════════════════════════════════════════════════════════════════
    # SAVE CSV
    # ══════════════════════════════════════════════════════════════════════════
    csv_path = os.path.join(args.output_dir, 'per_image_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_idx',
                    'adc_mae','adc_mse','adc_psnr','adc_ssim',
                    't2_mae', 't2_mse', 't2_psnr', 't2_ssim'])
        for i in range(len(adc_m['mae'])):
            w.writerow([i,
                        adc_m['mae'][i], adc_m['mse'][i],
                        adc_m['psnr'][i], adc_m['ssim'][i],
                        t2_m['mae'][i],  t2_m['mse'][i],
                        t2_m['psnr'][i], t2_m['ssim'][i]])

    summary_path = os.path.join(args.output_dir, 'summary_metrics.csv')
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric','adc_mean','adc_std','t2_mean','t2_std'])
        for k in ['mae','mse','psnr','ssim']:
            w.writerow([k.upper(),
                        f"{adc_s[k]['mean']:.4f}", f"{adc_s[k]['std']:.4f}",
                        f"{t2_s[k]['mean']:.4f}",  f"{t2_s[k]['std']:.4f}"])
        w.writerow(['FID', f"{fid_adc:.2f}", '-', f"{fid_t2:.2f}", '-'])

    writer.close()

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"\n  {'Metric':<8}  {'ADC Reconstruction':<26}  T2 Synthesis")
    print(f"  {'-'*8}  {'-'*26}  {'-'*26}")
    for k, unit in [('mae',''), ('mse',''), ('psnr',' dB'), ('ssim','')]:
        a, t = adc_s[k], t2_s[k]
        print(f"  {k.upper():<8}  {a['mean']:.4f} ± {a['std']:.4f}{unit:<12}"
              f"  {t['mean']:.4f} ± {t['std']:.4f}{unit}")
    print(f"  {'FID':<8}  {fid_adc:<26.2f}  {fid_t2:.2f}")
    print(f"\n  CSV  → {summary_path}")
    print(f"  CSV  → {csv_path}")
    print(f"\n  TensorBoard → tensorboard --logdir {tb_dir}")
    print(           "              http://localhost:6006")


if __name__ == '__main__':
    main()
