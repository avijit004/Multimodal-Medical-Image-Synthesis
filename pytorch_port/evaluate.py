"""
Unified evaluation: Semi-supervised vs Self-supervised models.

Runs each model on the same paired data, reports individual results, then
produces a side-by-side comparison and writes everything to a single
TensorBoard log dir.

Semi-supervised outputs
  Scalars   : MAE / MSE / PSNR / SSIM / FID  (ADC reconstruction + T2 synthesis)
  Images    : real vs reconstructed ADC, real vs synthesised T2, random pairs
  Histograms: per-image PSNR / SSIM distributions

Self-supervised outputs
  Scalars   : Top-1 / Top-3 / Top-5 retrieval accuracy, Mean Rank
  Projector : ADC + T2 encoder embeddings (colour-coded by modality)

Comparison
  Figure    : matplotlib side-by-side bar chart
  Text      : markdown summary table
  CSV       : eval_summary.csv

Usage:
    python evaluate.py \\
        --semi_checkpoint ./results_semi_real/…/semi_ckpt_final.pt \\
        --self_checkpoint ./results_self/…/self_ckpt_final.pt \\
        --adc_dir ../data/adc --t2_dir ../data/t2 \\
        --paired_list ../data/paired_names.txt \\
        --adc_list ../data/adc_names.txt --t2_list ../data/t2_names.txt \\
        --output_dir ./eval_output

    tensorboard --logdir ./eval_output/tensorboard
    open http://localhost:6006
"""

import argparse
import os
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import cv2
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from scipy.linalg import sqrtm
import torchvision.models as tv_models
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models import SharedLayers, Encoder, GeneratorADC, GeneratorT2, ProjectionHead
from dataset import get_dataloader, get_paired_dataloader, PairedMedicalImageDataset
from utils import get_device


# ─── shared helpers ───────────────────────────────────────────────────────────

def tensor_to_uint8(t):
    """[B,1,H,W] in [-1,1]  →  numpy [B,H,W] uint8"""
    arr = t.cpu().numpy()
    return ((arr + 1) * 127.5).clip(0, 255).astype(np.uint8)[:, 0]


def to_grid(t, nrow=8):
    """[B,1,H,W] in [-1,1]  →  display [3,H',W'] in [0,1]"""
    return vutils.make_grid(t.repeat(1, 3, 1, 1), nrow=nrow,
                            normalize=True, value_range=(-1, 1))


def paired_metrics(real_np, fake_np):
    mae_l, mse_l, psnr_l, ssim_l = [], [], [], []
    for r, f in zip(real_np, fake_np):
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

class InceptionExtractor(nn.Module):
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
        return self.model(self.tfm((imgs + 1) / 2.0)).cpu().numpy()


def compute_fid(f_real, f_fake):
    mu1, s1 = f_real.mean(0), np.cov(f_real, rowvar=False)
    mu2, s2 = f_fake.mean(0), np.cov(f_fake, rowvar=False)
    eps = np.eye(s1.shape[0]) * 1e-6
    covmean, _ = sqrtm((s1 + eps) @ (s2 + eps), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(s1 + s2 - 2 * covmean))


def real_features(extractor, loader, max_n):
    feats, n = [], 0
    for batch in loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        feats.append(extractor.extract(imgs.to(extractor.device)))
        n += feats[-1].shape[0]
        if n >= max_n:
            break
    return np.concatenate(feats)[:max_n]


def gen_features(extractor, gen_fn, n, z_dim, bs, device):
    feats, done = [], 0
    while done < n:
        b = min(bs, n - done)
        z = torch.randn(b, z_dim, device=device)
        with torch.no_grad():
            imgs = gen_fn(z)
        feats.append(extractor.extract(imgs))
        done += b
    return np.concatenate(feats)[:n]


# ══════════════════════════════════════════════════════════════════════════════
# PART A — SEMI-SUPERVISED EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_semi(args, device, writer):
    print("\n" + "=" * 60)
    print("SEMI-SUPERVISED EVALUATION")
    print("=" * 60)

    # ── load checkpoint ────────────────────────────────────────────────────────
    ckpt    = torch.load(args.semi_checkpoint, map_location=device, weights_only=False)
    ckpt_it = ckpt.get('iter', 0)
    shared  = SharedLayers().to(device)
    encoder = Encoder(args.z_dim).to(device)
    gen_adc = GeneratorADC(shared, args.z_dim).to(device)
    gen_t2  = GeneratorT2(shared).to(device)
    shared.load_state_dict(ckpt['shared'])
    encoder.load_state_dict(ckpt['encoder'])
    gen_adc.load_state_dict(ckpt['gen_adc'])
    gen_t2.load_state_dict(ckpt['gen_t2'])
    for m in (shared, encoder, gen_adc, gen_t2):
        m.eval()
    print(f"  Checkpoint loaded — iter {ckpt_it}")

    # ── A1: paired image-quality metrics ──────────────────────────────────────
    print("\n  [A1] Paired image quality metrics …")
    paired_loader = get_paired_dataloader(
        args.adc_dir, args.t2_dir, args.paired_list,
        batch_size=args.batch_size, shuffle=False)

    adc_m = {'mae': [], 'mse': [], 'psnr': [], 'ssim': []}
    t2_m  = {'mae': [], 'mse': [], 'psnr': [], 'ssim': []}
    vis_radc, vis_fadc, vis_rt2, vis_ft2 = [], [], [], []
    vis_n = 0

    with torch.no_grad():
        for real_adc, real_t2 in paired_loader:
            real_adc, real_t2 = real_adc.to(device), real_t2.to(device)
            z   = encoder(real_adc)
            rec = gen_adc(z)
            syn = gen_t2(rec)

            m_a = paired_metrics(tensor_to_uint8(real_adc), tensor_to_uint8(rec))
            m_t = paired_metrics(tensor_to_uint8(real_t2),  tensor_to_uint8(syn))
            for k in adc_m:
                adc_m[k].extend(m_a[k])
                t2_m[k].extend(m_t[k])

            if vis_n < args.vis_images:
                take = min(args.vis_images - vis_n, real_adc.size(0))
                vis_radc.append(real_adc[:take].cpu())
                vis_fadc.append(rec[:take].cpu())
                vis_rt2.append(real_t2[:take].cpu())
                vis_ft2.append(syn[:take].cpu())
                vis_n += take

    adc_s = summarise(adc_m)
    t2_s  = summarise(t2_m)

    # log scalars
    step = 0
    for metric in ('mae', 'mse', 'psnr', 'ssim'):
        writer.add_scalar(f'Semi/ADC_Recon/{metric.upper()}', adc_s[metric]['mean'], step)
        writer.add_scalar(f'Semi/T2_Synth/{metric.upper()}',  t2_s[metric]['mean'],  step)

    # log histograms
    writer.add_histogram('Semi/Distribution/ADC_PSNR', np.array(adc_m['psnr']), step)
    writer.add_histogram('Semi/Distribution/ADC_SSIM', np.array(adc_m['ssim']), step)
    writer.add_histogram('Semi/Distribution/T2_PSNR',  np.array(t2_m['psnr']),  step)
    writer.add_histogram('Semi/Distribution/T2_SSIM',  np.array(t2_m['ssim']),  step)

    # log image grids
    nrow = min(8, args.vis_images)
    radc = torch.cat(vis_radc); fadc = torch.cat(vis_fadc)
    rt2  = torch.cat(vis_rt2);  ft2  = torch.cat(vis_ft2)

    writer.add_image('Semi/ADC/1_Real',          to_grid(radc, nrow), step)
    writer.add_image('Semi/ADC/2_Reconstructed', to_grid(fadc, nrow), step)
    writer.add_image('Semi/T2/1_Real',           to_grid(rt2,  nrow), step)
    writer.add_image('Semi/T2/2_Synthesized',    to_grid(ft2,  nrow), step)

    il_adc = torch.stack([x for p in zip(radc, fadc) for x in p])
    il_t2  = torch.stack([x for p in zip(rt2, ft2)   for x in p])
    writer.add_image('Semi/Comparison/ADC_Real_vs_Recon', to_grid(il_adc, nrow=2), step)
    writer.add_image('Semi/Comparison/T2_Real_vs_Synth',  to_grid(il_t2,  nrow=2), step)

    # ── A2: random generation visualisation ───────────────────────────────────
    print("  [A2] Random generation …")
    with torch.no_grad():
        zr     = torch.randn(args.vis_images, args.z_dim, device=device)
        rnd_a  = gen_adc(zr).cpu()
        rnd_t  = gen_t2(gen_adc(zr)).cpu()
    writer.add_image('Semi/Generated/ADC_from_noise', to_grid(rnd_a, nrow), step)
    writer.add_image('Semi/Generated/T2_from_noise',  to_grid(rnd_t, nrow), step)
    il_rnd = torch.stack([x for p in zip(rnd_a, rnd_t) for x in p])
    writer.add_image('Semi/Generated/ADC_T2_pairs', to_grid(il_rnd, nrow=2), step)

    # ── A3: FID ───────────────────────────────────────────────────────────────
    print(f"  [A3] FID ({args.fid_samples} samples) …")
    ext = InceptionExtractor(device)

    f_real_adc = real_features(ext, get_dataloader(
        args.adc_dir, args.adc_list, batch_size=args.batch_size, shuffle=False),
        args.fid_samples)
    f_fake_adc = gen_features(ext, gen_adc, args.fid_samples, args.z_dim, args.batch_size, device)
    fid_adc = compute_fid(f_real_adc, f_fake_adc)

    f_real_t2  = real_features(ext, get_dataloader(
        args.t2_dir, args.t2_list, batch_size=args.batch_size, shuffle=False),
        args.fid_samples)
    f_fake_t2  = gen_features(ext, lambda z: gen_t2(gen_adc(z)),
                               args.fid_samples, args.z_dim, args.batch_size, device)
    fid_t2 = compute_fid(f_real_t2, f_fake_t2)

    writer.add_scalar('Semi/ADC_Recon/FID', fid_adc, step)
    writer.add_scalar('Semi/T2_Synth/FID',  fid_t2,  step)

    # ── print ─────────────────────────────────────────────────────────────────
    print(f"\n  ADC reconstruction — PSNR {adc_s['psnr']['mean']:.2f} dB  "
          f"SSIM {adc_s['ssim']['mean']:.4f}  FID {fid_adc:.2f}")
    print(f"  T2  synthesis      — PSNR {t2_s['psnr']['mean']:.2f} dB  "
          f"SSIM {t2_s['ssim']['mean']:.4f}  FID {fid_t2:.2f}")

    return {
        'adc_s': adc_s, 't2_s': t2_s,
        'fid_adc': fid_adc, 'fid_t2': fid_t2,
        'adc_m': adc_m, 't2_m': t2_m,
        'ckpt_it': ckpt_it,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART B — SELF-SUPERVISED EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_self(args, device, writer):
    print("\n" + "=" * 60)
    print("SELF-SUPERVISED EVALUATION")
    print("=" * 60)

    # ── load checkpoint ────────────────────────────────────────────────────────
    ckpt    = torch.load(args.self_checkpoint, map_location=device, weights_only=False)
    ckpt_it = ckpt.get('iter', 0)

    # Recover proj_dim from saved args (defaults to 64)
    saved_args = ckpt.get('args', {})
    proj_dim   = saved_args.get('proj_dim', 64)

    enc_adc  = Encoder(args.z_dim).to(device)
    enc_t2   = Encoder(args.z_dim).to(device)
    proj_adc = ProjectionHead(args.z_dim, proj_dim=proj_dim).to(device)
    proj_t2  = ProjectionHead(args.z_dim, proj_dim=proj_dim).to(device)
    enc_adc.load_state_dict(ckpt['encoder_adc'])
    enc_t2.load_state_dict(ckpt['encoder_t2'])
    proj_adc.load_state_dict(ckpt['proj_adc'])
    proj_t2.load_state_dict(ckpt['proj_t2'])
    for m in (enc_adc, enc_t2, proj_adc, proj_t2):
        m.eval()
    print(f"  Checkpoint loaded — iter {ckpt_it}  (proj_dim={proj_dim})")

    # ── B1: cross-modal retrieval ──────────────────────────────────────────────
    # IMPORTANT: retrieval must use the PROJECTED features (the space where
    # NT-Xent aligned ADC↔T2), NOT the raw encoder outputs.  The encoders were
    # never directly constrained to share a common embedding space; only the
    # projection-head outputs were trained to be cross-modally aligned.
    print("\n  [B1] Cross-modal retrieval (ADC query → find paired T2) …")
    dataset = PairedMedicalImageDataset(args.adc_dir, args.t2_dir, args.paired_list)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=0, drop_last=False)
    N = len(dataset)
    print(f"  {N} paired samples")

    all_z_adc, all_z_t2 = [], []
    with torch.no_grad():
        for adc, t2 in loader:
            # Use projection-head outputs — same space used during training
            h_adc = F.normalize(proj_adc(enc_adc(adc.to(device))), dim=1)
            h_t2  = F.normalize(proj_t2(enc_t2(t2.to(device))),   dim=1)
            all_z_adc.append(h_adc.cpu())
            all_z_t2.append(h_t2.cpu())

    z_adc = torch.cat(all_z_adc)   # already L2-normalised  [N, proj_dim]
    z_t2  = torch.cat(all_z_t2)    # already L2-normalised  [N, proj_dim]

    sim    = torch.mm(z_adc, z_t2.t())                 # [N, N]
    labels = torch.arange(N)
    ranks  = sim.argsort(dim=1, descending=True)        # [N, N]

    top1 = (ranks[:, 0] == labels).float().mean().item()
    top3 = (ranks[:, :3] == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    top5 = (ranks[:, :5] == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    correct_rank = (ranks == labels.unsqueeze(1)).nonzero(as_tuple=False)[:, 1].float()
    mean_rank    = correct_rank.mean().item() + 1   # 1-indexed

    step = 0
    writer.add_scalar('Self/Retrieval/Top1_Acc_%',  top1 * 100,  step)
    writer.add_scalar('Self/Retrieval/Top3_Acc_%',  top3 * 100,  step)
    writer.add_scalar('Self/Retrieval/Top5_Acc_%',  top5 * 100,  step)
    writer.add_scalar('Self/Retrieval/Mean_Rank',   mean_rank,   step)

    print(f"\n  Top-1 : {top1*100:.1f}%")
    print(f"  Top-3 : {top3*100:.1f}%")
    print(f"  Top-5 : {top5*100:.1f}%")
    print(f"  Mean Rank : {mean_rank:.1f} / {N}")

    # ── B2: embedding projector ────────────────────────────────────────────────
    # Use the projected features (proj_dim) for the projector — they are the
    # ones that were explicitly aligned, so the ADC/T2 clusters should overlap.
    print("\n  [B2] Embedding projector …")
    max_proj = min(N, 512)   # TensorBoard projector works best with ≤ 512 pts
    n_adc = min(max_proj, z_adc.size(0))
    n_t2  = min(max_proj, z_t2.size(0))
    emb_all     = torch.cat([z_adc[:n_adc], z_t2[:n_t2]], dim=0)   # [n_adc+n_t2, proj_dim]
    labels_proj = ['ADC'] * n_adc + ['T2'] * n_t2

    # Collect exactly n_adc/n_t2 images so sprite shape matches emb_all
    def grey_to_rgb(t):
        return ((t.repeat(1, 3, 1, 1) + 1) / 2.0).clamp(0, 1)

    imgs_adc_proj, imgs_t2_proj = [], []
    n_collected = 0
    for adc, t2 in loader:
        take = min(max_proj - n_collected, adc.size(0))
        imgs_adc_proj.append(adc[:take])
        imgs_t2_proj.append(t2[:take])
        n_collected += take
        if n_collected >= max_proj:
            break
    imgs_adc_proj = torch.cat(imgs_adc_proj)[:n_adc]
    imgs_t2_proj  = torch.cat(imgs_t2_proj)[:n_t2]

    sprite = torch.cat([grey_to_rgb(imgs_adc_proj),
                        grey_to_rgb(imgs_t2_proj)], dim=0)   # [n_adc+n_t2, 3, H, W]

    writer.add_embedding(
        emb_all,
        metadata=labels_proj,
        label_img=sprite,
        global_step=step,
        tag='Self/Embeddings_ADC_vs_T2',
    )

    return {
        'top1': top1, 'top3': top3, 'top5': top5,
        'mean_rank': mean_rank, 'N': N,
        'ckpt_it': ckpt_it,
        'z_adc': z_adc, 'z_t2': z_t2,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART C — COMPARISON + VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def compare_and_visualise(semi_res, self_res, writer, output_dir):
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    s  = semi_res
    sl = self_res

    # ── C1: matplotlib comparison figure ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Semi-supervised vs Self-supervised — Evaluation Comparison', fontsize=13)

    # Panel 1 — semi: PSNR and SSIM (ADC recon vs T2 synth)
    ax = axes[0]
    metrics  = ['ADC PSNR\n(dB)', 'T2 PSNR\n(dB)', 'ADC SSIM', 'T2 SSIM']
    means    = [s['adc_s']['psnr']['mean'], s['t2_s']['psnr']['mean'],
                s['adc_s']['ssim']['mean'], s['t2_s']['ssim']['mean']]
    stds     = [s['adc_s']['psnr']['std'],  s['t2_s']['psnr']['std'],
                s['adc_s']['ssim']['std'],   s['t2_s']['ssim']['std']]
    x = np.arange(len(metrics))
    ax.bar(x, means, yerr=stds, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'],
           capsize=4, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title('Semi-supervised\nImage Quality')
    ax.set_ylabel('Score')

    # Panel 2 — semi: FID
    ax2 = axes[1]
    fid_vals = [s['fid_adc'], s['fid_t2']]
    ax2.bar(['ADC FID', 'T2 FID'], fid_vals,
            color=['#4C72B0', '#DD8452'], alpha=0.85)
    for i, v in enumerate(fid_vals):
        ax2.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=9)
    ax2.set_title('Semi-supervised\nFID (lower = better)')
    ax2.set_ylabel('FID')

    # Panel 3 — self: retrieval accuracy
    ax3 = axes[2]
    ret_labels = ['Top-1', 'Top-3', 'Top-5']
    ret_vals   = [sl['top1'] * 100, sl['top3'] * 100, sl['top5'] * 100]
    bars = ax3.bar(ret_labels, ret_vals,
                   color=['#8172B2', '#937860', '#56B4E9'], alpha=0.85)
    for bar, v in zip(bars, ret_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
    ax3.set_ylim(0, 110)
    ax3.set_title('Self-supervised\nCross-modal Retrieval Accuracy')
    ax3.set_ylabel('Accuracy (%)')
    ax3.axhline(100, ls='--', c='grey', lw=0.8)

    plt.tight_layout()
    writer.add_figure('Comparison/Summary_Chart', fig, global_step=0)

    fig_path = os.path.join(output_dir, 'comparison_chart.png')
    fig.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart saved → {fig_path}")

    # ── C2: text table in TensorBoard ─────────────────────────────────────────
    md = (
        "## Evaluation Summary\n\n"
        "### Semi-supervised model\n\n"
        "| Metric | ADC Reconstruction | T2 Synthesis |\n"
        "|--------|-------------------|--------------|\n"
        f"| MAE    | {s['adc_s']['mae']['mean']:.4f} ± {s['adc_s']['mae']['std']:.4f} "
        f"| {s['t2_s']['mae']['mean']:.4f} ± {s['t2_s']['mae']['std']:.4f} |\n"
        f"| MSE    | {s['adc_s']['mse']['mean']:.4f} ± {s['adc_s']['mse']['std']:.4f} "
        f"| {s['t2_s']['mse']['mean']:.4f} ± {s['t2_s']['mse']['std']:.4f} |\n"
        f"| PSNR   | {s['adc_s']['psnr']['mean']:.2f} ± {s['adc_s']['psnr']['std']:.2f} dB "
        f"| {s['t2_s']['psnr']['mean']:.2f} ± {s['t2_s']['psnr']['std']:.2f} dB |\n"
        f"| SSIM   | {s['adc_s']['ssim']['mean']:.4f} ± {s['adc_s']['ssim']['std']:.4f} "
        f"| {s['t2_s']['ssim']['mean']:.4f} ± {s['t2_s']['ssim']['std']:.4f} |\n"
        f"| FID    | {s['fid_adc']:.2f} | {s['fid_t2']:.2f} |\n\n"
        "### Self-supervised model\n\n"
        "| Metric | Value |\n"
        "|--------|-------|\n"
        f"| Top-1 Retrieval | {sl['top1']*100:.1f}% |\n"
        f"| Top-3 Retrieval | {sl['top3']*100:.1f}% |\n"
        f"| Top-5 Retrieval | {sl['top5']*100:.1f}% |\n"
        f"| Mean Rank       | {sl['mean_rank']:.1f} / {sl['N']} |\n\n"
        "> Semi-supervised: image synthesis quality (PSNR/SSIM/FID).  \n"
        "> Self-supervised: cross-modal embedding alignment (retrieval accuracy).  \n"
        f"> Semi checkpoint iter: {s['ckpt_it']}  |  "
        f"Self checkpoint iter: {sl['ckpt_it']}\n"
    )
    writer.add_text('Comparison/Summary_Table', md, global_step=0)

    # ── C3: print to console ───────────────────────────────────────────────────
    print("\n  ┌─────────────────────────────────────────────────────────┐")
    print(  "  │          SEMI-SUPERVISED  (image quality)               │")
    print(  "  ├──────────────┬────────────────────┬────────────────────┤")
    print(  "  │ Metric       │ ADC Reconstruction │ T2 Synthesis       │")
    print(  "  ├──────────────┼────────────────────┼────────────────────┤")
    for k, unit in [('mae',''), ('mse',''), ('psnr',' dB'), ('ssim','')]:
        a, t = s['adc_s'][k], s['t2_s'][k]
        print(f"  │ {k.upper():<12} │ {a['mean']:.4f} ± {a['std']:.4f}{unit:<6} │ "
              f"{t['mean']:.4f} ± {t['std']:.4f}{unit:<6} │")
    print(f"  │ {'FID':<12} │ {s['fid_adc']:<18.2f} │ {s['fid_t2']:<18.2f} │")
    print(  "  └──────────────┴────────────────────┴────────────────────┘")

    print("\n  ┌─────────────────────────────────────────────────────────┐")
    print(  "  │        SELF-SUPERVISED  (retrieval accuracy)            │")
    print(  "  ├─────────────────────┬───────────────────────────────────┤")
    print(f"  │ Top-1 Accuracy      │ {sl['top1']*100:.1f}%"
          f"{'':30}│")
    print(f"  │ Top-3 Accuracy      │ {sl['top3']*100:.1f}%"
          f"{'':30}│")
    print(f"  │ Top-5 Accuracy      │ {sl['top5']*100:.1f}%"
          f"{'':30}│")
    print(f"  │ Mean Rank           │ {sl['mean_rank']:.1f} / {sl['N']}"
          f"{'':28}│")
    print(  "  └─────────────────────┴───────────────────────────────────┘")


# ══════════════════════════════════════════════════════════════════════════════
# CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_csvs(semi_res, self_res, output_dir):
    # per-image metrics (semi only)
    s = semi_res
    per_img_path = os.path.join(output_dir, 'semi_per_image_metrics.csv')
    with open(per_img_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_idx',
                    'adc_mae', 'adc_mse', 'adc_psnr', 'adc_ssim',
                    't2_mae',  't2_mse',  't2_psnr',  't2_ssim'])
        for i in range(len(s['adc_m']['mae'])):
            w.writerow([i,
                        s['adc_m']['mae'][i], s['adc_m']['mse'][i],
                        s['adc_m']['psnr'][i], s['adc_m']['ssim'][i],
                        s['t2_m']['mae'][i],  s['t2_m']['mse'][i],
                        s['t2_m']['psnr'][i], s['t2_m']['ssim'][i]])

    # summary (both models)
    summary_path = os.path.join(output_dir, 'eval_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'metric', 'value_mean', 'value_std', 'notes'])
        for k in ('mae', 'mse', 'psnr', 'ssim'):
            w.writerow(['semi', f'adc_{k}',
                        f"{s['adc_s'][k]['mean']:.4f}", f"{s['adc_s'][k]['std']:.4f}",
                        'ADC reconstruction'])
            w.writerow(['semi', f't2_{k}',
                        f"{s['t2_s'][k]['mean']:.4f}", f"{s['t2_s'][k]['std']:.4f}",
                        'T2 synthesis'])
        w.writerow(['semi', 'adc_fid', f"{s['fid_adc']:.2f}", '-', 'ADC FID'])
        w.writerow(['semi', 't2_fid',  f"{s['fid_t2']:.2f}",  '-', 'T2 FID'])
        sl = self_res
        w.writerow(['self', 'top1_acc_%', f"{sl['top1']*100:.2f}", '-', 'retrieval'])
        w.writerow(['self', 'top3_acc_%', f"{sl['top3']*100:.2f}", '-', 'retrieval'])
        w.writerow(['self', 'top5_acc_%', f"{sl['top5']*100:.2f}", '-', 'retrieval'])
        w.writerow(['self', 'mean_rank',  f"{sl['mean_rank']:.2f}", '-',
                    f'out of {sl["N"]}'])

    print(f"\n  CSV (per-image) → {per_img_path}")
    print(f"  CSV (summary)   → {summary_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI + MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Unified semi + self evaluation with TensorBoard comparison.')
    p.add_argument('--semi_checkpoint', required=True,
                   help='Path to semi-supervised .pt checkpoint')
    p.add_argument('--self_checkpoint', required=True,
                   help='Path to self-supervised .pt checkpoint')
    p.add_argument('--adc_dir',     required=True)
    p.add_argument('--t2_dir',      required=True)
    p.add_argument('--paired_list', required=True)
    p.add_argument('--adc_list',    required=True)
    p.add_argument('--t2_list',     required=True)
    p.add_argument('--output_dir',  default='./eval_output')
    p.add_argument('--z_dim',       type=int, default=128)
    p.add_argument('--batch_size',  type=int, default=32)
    p.add_argument('--fid_samples', type=int, default=200)
    p.add_argument('--vis_images',  type=int, default=16)
    p.add_argument('--use_cpu',     action='store_true')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cpu') if args.use_cpu else get_device()
    print(f"\nDevice : {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    tb_dir = os.path.join(args.output_dir, 'tensorboard')
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"TensorBoard dir : {tb_dir}")
    print(f"  → run  : tensorboard --logdir {tb_dir}")
    print(f"  → open : http://localhost:6006\n")

    # ── run evaluations ───────────────────────────────────────────────────────
    semi_res = evaluate_semi(args, device, writer)
    self_res = evaluate_self(args, device, writer)

    # ── comparison + visualisation ────────────────────────────────────────────
    compare_and_visualise(semi_res, self_res, writer, args.output_dir)

    # ── save CSVs ─────────────────────────────────────────────────────────────
    save_csvs(semi_res, self_res, args.output_dir)

    writer.close()
    print(f"\nDone.  TensorBoard → tensorboard --logdir {tb_dir}")
    print(        "                   http://localhost:6006")


if __name__ == '__main__':
    main()
