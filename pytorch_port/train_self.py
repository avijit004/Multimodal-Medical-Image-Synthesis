"""
Self-supervised cross-modal contrastive pretraining.

Training scale matches the semi-supervised baseline exactly:
  - 40,000 iterations  (same as train_semi.py)
  - batch size 32      (same)
  - Adam lr=1e-4       (same)
  - log every 50 iters (same)
  - save every 1,000 iters after iter 20,000 (same)

Method: NT-Xent (InfoNCE) cross-modal contrastive learning.
  - Positive pair : (ADC_i, T2_i)  — same patient, independently augmented
  - Negative pairs: all other cross-modal combinations in the batch
  - No labels used at any point (purely self-supervised)

Comparison table (all trained on the same PROSTATEx data):
  Supervised      -> L1 reconstruction on paired images only
  Semi-supervised -> L1 (paired) + WGAN-GP (unpaired), alternating
  Self-supervised -> NT-Xent on paired images, no generators, no labels
"""

import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models import Encoder, ProjectionHead
from dataset import AugmentedPairedDataset
from utils import form_results, get_device, infinite_dataloader


# ---------------------------------------------------------------------------
# CLI  — mirrors train_semi.py argument names wherever possible
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Self-supervised contrastive pretraining (40K iters, matches semi-supervised scale).'
    )
    p.add_argument('--adc_dir',       required=True,  help='Directory of ADC PNG images')
    p.add_argument('--t2_dir',        required=True,  help='Directory of T2w PNG images')
    p.add_argument('--paired_list',   required=True,  help='Text file listing paired filenames')
    p.add_argument('--results_path',  default='./results_self')
    p.add_argument('--save_image_path', default='./generated_self',
                   help='(Unused for self-supervised; kept for API parity with train_semi.py)')
    p.add_argument('--batch_size',    type=int,   default=32)
    p.add_argument('--iters',         type=int,   default=10000,  help='Training iterations (default: 10,000, matches semi-supervised run)')
    p.add_argument('--z_dim',         type=int,   default=128)
    p.add_argument('--proj_dim',      type=int,   default=64)
    p.add_argument('--lr',            type=float, default=1e-4)   # matches semi-supervised
    p.add_argument('--temperature',   type=float, default=0.07)
    p.add_argument('--log_interval',  type=int,   default=50)     # matches semi-supervised
    p.add_argument('--save_interval', type=int,   default=1000)   # matches semi-supervised
    p.add_argument('--save_after',    type=int,   default=0,
                   help='Only save checkpoints after this iteration (default: 0 = save from start)')
    p.add_argument('--use_cpu',       action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# NT-Xent (InfoNCE) loss
# ---------------------------------------------------------------------------

def nt_xent_loss(h_adc, h_t2, temperature):
    """
    Cross-modal NT-Xent loss.

    h_adc, h_t2 : [N, proj_dim]  L2-normalised projection vectors
    Diagonal of the N×N similarity matrix = positive pairs.
    Both directions (ADC→T2, T2→ADC) are averaged.
    """
    N      = h_adc.size(0)
    sim    = torch.mm(h_adc, h_t2.t()) / temperature   # [N, N]
    labels = torch.arange(N, device=h_adc.device)
    loss   = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2.0
    return loss


# ---------------------------------------------------------------------------
# Top-1 retrieval accuracy  (monitoring only, not a training signal)
# ---------------------------------------------------------------------------

@torch.no_grad()
def top1_accuracy(h_adc, h_t2, temperature):
    N      = h_adc.size(0)
    sim    = torch.mm(h_adc, h_t2.t()) / temperature
    labels = torch.arange(N, device=h_adc.device)
    preds  = sim.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device('cpu') if args.use_cpu else get_device()
    print(f"Using device: {device}")

    # ---- Data ---------------------------------------------------------------
    from torch.utils.data import DataLoader
    dataset = AugmentedPairedDataset(args.adc_dir, args.t2_dir, args.paired_list)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=0, drop_last=True)
    data_iter = infinite_dataloader(loader)     # same helper as train_semi.py

    print(f"Dataset  : {len(dataset)} paired samples  |  "
          f"{len(loader)} batches/epoch  |  batch={args.batch_size}")

    # ---- Models -------------------------------------------------------------
    encoder_adc = Encoder(args.z_dim).to(device)
    encoder_t2  = Encoder(args.z_dim).to(device)
    proj_adc    = ProjectionHead(args.z_dim, proj_dim=args.proj_dim).to(device)
    proj_t2     = ProjectionHead(args.z_dim, proj_dim=args.proj_dim).to(device)

    params     = (list(encoder_adc.parameters()) + list(encoder_t2.parameters()) +
                  list(proj_adc.parameters())    + list(proj_t2.parameters()))
    optimizer  = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))

    # ---- Logging ------------------------------------------------------------
    tb_path, model_path, log_path = form_results(args.results_path)
    writer = SummaryWriter(log_dir=tb_path)
    os.makedirs(model_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SELF-SUPERVISED CROSS-MODAL CONTRASTIVE TRAINING")
    print(f"{'='*60}")
    print(f"Iterations : {args.iters}  |  Batch : {args.batch_size}  |  "
          f"LR : {args.lr}  |  τ : {args.temperature}  |  Device : {device}\n")

    # ---- Training loop  (mirrors train_semi.py structure exactly) -----------
    for i in range(args.iters):

        aug_adc, aug_t2 = next(data_iter)
        aug_adc = aug_adc.to(device)
        aug_t2  = aug_t2.to(device)

        # Forward
        z_adc = encoder_adc(aug_adc)
        z_t2  = encoder_t2(aug_t2)
        h_adc = F.normalize(proj_adc(z_adc), dim=1)
        h_t2  = F.normalize(proj_t2(z_t2),  dim=1)

        loss = nt_xent_loss(h_adc, h_t2, args.temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- Logging (every log_interval, same as semi-supervised) ----------
        if i % args.log_interval == 0:
            acc = top1_accuracy(h_adc.detach(), h_t2.detach(), args.temperature)

            print(f"iteration:{i}")
            print(f"NT-Xent Loss  = {loss.item():.4f}")
            print(f"Top-1 Acc     = {acc*100:.1f}%")

            writer.add_scalar("NT-Xent Loss",  loss.item(), i)
            writer.add_scalar("Top-1 Acc (%)", acc * 100,   i)

        # ---- Checkpoint (same policy as train_semi.py) ----------------------
        if i > args.save_after and i % args.save_interval == 0:
            ckpt_path = os.path.join(model_path, f'self_ckpt_{i}.pt')
            # Explicitly move to CPU before saving — fixes MPS serialisation
            # bug on Apple Silicon (ios_base::clear / unexpected pos error).
            torch.save({
                'iter':        i,
                'encoder_adc': {k: v.cpu() for k, v in encoder_adc.state_dict().items()},
                'encoder_t2':  {k: v.cpu() for k, v in encoder_t2.state_dict().items()},
                'proj_adc':    {k: v.cpu() for k, v in proj_adc.state_dict().items()},
                'proj_t2':     {k: v.cpu() for k, v in proj_t2.state_dict().items()},
                'args':        vars(args),
            }, ckpt_path)
            print(f"  [Checkpoint saved → {ckpt_path}]")

    # Save final checkpoint
    final_path = os.path.join(model_path, 'self_ckpt_final.pt')
    torch.save({
        'iter':        args.iters,
        'encoder_adc': {k: v.cpu() for k, v in encoder_adc.state_dict().items()},
        'encoder_t2':  {k: v.cpu() for k, v in encoder_t2.state_dict().items()},
        'proj_adc':    {k: v.cpu() for k, v in proj_adc.state_dict().items()},
        'proj_t2':     {k: v.cpu() for k, v in proj_t2.state_dict().items()},
        'args':        vars(args),
    }, final_path)

    writer.close()
    print(f"\nTraining complete! Results: {args.results_path}")
    print(f"Final checkpoint : {final_path}")
    print(f"\nNext step → run test_self.py with --checkpoint {final_path}")


if __name__ == '__main__':
    main()
