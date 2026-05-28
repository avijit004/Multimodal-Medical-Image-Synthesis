"""
Evaluation for self-supervised pretrained encoders.

Two modes  (--mode):

  retrieval      [DEFAULT — no labels needed]
                 Given each ADC image, rank all T2 images by cosine similarity
                 and check whether the correct paired T2 is ranked #1, top-3,
                 or top-5.  Uses paired_names.txt — no CS/nonCS labels required.
                 This is the standard self-supervised evaluation.

  linear_probe   [needs labeled train_list / test_list]
                 Freeze encoders, train a linear classifier on [z_adc || z_t2],
                 report classification accuracy — same benchmark as the paper.

Results are saved to --output_dir  (default: ./test_output_self)
so they stay separate from test_semi.py output (./test_output_semi).
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import Encoder
from dataset import PairedMedicalImageDataset, LabeledPairedDataset
from utils import get_device


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  required=True,
                   help='Path to self_ckpt_final.pt')
    p.add_argument('--adc_dir',     required=True)
    p.add_argument('--t2_dir',      required=True)
    p.add_argument('--paired_list', default=None,
                   help='Name list for retrieval mode  (no labels, format: filename)')
    p.add_argument('--train_list',  default=None,
                   help='Labeled name list for linear_probe  (format: filename label)')
    p.add_argument('--test_list',   default=None,
                   help='Labeled name list for linear_probe  (format: filename label)')
    p.add_argument('--mode',        default='retrieval',
                   choices=['retrieval', 'linear_probe'])
    p.add_argument('--z_dim',       type=int,   default=128)
    p.add_argument('--iters',       type=int,   default=10000)
    p.add_argument('--batch_size',  type=int,   default=64)
    p.add_argument('--lr',          type=float, default=0.005)
    p.add_argument('--output_dir',  default='./test_output_self')
    p.add_argument('--use_cpu',     action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load encoders
# ---------------------------------------------------------------------------

def load_encoders(checkpoint_path, z_dim, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    enc_adc = Encoder(z_dim).to(device)
    enc_t2  = Encoder(z_dim).to(device)
    enc_adc.load_state_dict(ckpt['encoder_adc'])
    enc_t2.load_state_dict(ckpt['encoder_t2'])
    enc_adc.eval(); enc_t2.eval()
    print(f"Loaded checkpoint  iter={ckpt.get('iter','?')}  →  {checkpoint_path}")
    return enc_adc, enc_t2


# ---------------------------------------------------------------------------
# MODE 1 — Cross-modal retrieval  (no labels)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_retrieval(enc_adc, enc_t2, adc_dir, t2_dir, paired_list, batch_size, device, output_dir):
    """
    For every ADC image, compute cosine similarity to all T2 images.
    The correct T2 is the one at the same index (paired).
    Report Top-1, Top-3, Top-5 retrieval accuracy.
    """
    print("\n[Mode: retrieval]  No labels needed — using paired structure as ground truth.")

    dataset = PairedMedicalImageDataset(adc_dir, t2_dir, paired_list)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, drop_last=False)
    N = len(dataset)
    print(f"  {N} paired samples")

    # Extract all ADC and T2 embeddings
    all_z_adc, all_z_t2 = [], []
    for adc, t2 in loader:
        all_z_adc.append(enc_adc(adc.to(device)).cpu())
        all_z_t2.append(enc_t2(t2.to(device)).cpu())
    z_adc = F.normalize(torch.cat(all_z_adc), dim=1)   # [N, z_dim]
    z_t2  = F.normalize(torch.cat(all_z_t2),  dim=1)   # [N, z_dim]

    # Pairwise cosine similarity matrix [N, N]
    sim = torch.mm(z_adc, z_t2.t())   # sim[i,j] = cos(adc_i, t2_j)

    # Ground truth: diagonal (index i matches index i)
    labels = torch.arange(N)
    ranks  = sim.argsort(dim=1, descending=True)   # [N, N]  sorted by similarity

    top1 = (ranks[:, 0] == labels).float().mean().item()
    top3 = (ranks[:, :3] == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    top5 = (ranks[:, :5] == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    # Also compute ADC→T2 and T2→ADC mean rank
    correct_rank = (ranks == labels.unsqueeze(1)).nonzero(as_tuple=False)[:, 1].float()
    mean_rank = correct_rank.mean().item() + 1   # 1-indexed

    print(f"\n  Cross-modal retrieval  (ADC query → find correct T2):")
    print(f"  Top-1 Acc : {top1*100:.1f}%")
    print(f"  Top-3 Acc : {top3*100:.1f}%")
    print(f"  Top-5 Acc : {top5*100:.1f}%")
    print(f"  Mean Rank : {mean_rank:.1f} / {N}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, 'retrieval_results.txt')
    with open(result_path, 'w') as f:
        f.write('********* self-supervised retrieval ************\n')
        f.write(f'N samples  : {N}\n')
        f.write(f'Top-1 Acc  : {top1*100:.2f}%\n')
        f.write(f'Top-3 Acc  : {top3*100:.2f}%\n')
        f.write(f'Top-5 Acc  : {top5*100:.2f}%\n')
        f.write(f'Mean Rank  : {mean_rank:.2f} / {N}\n')

    print(f"\n  Results saved → {result_path}")
    return top1, top3, top5, mean_rank


# ---------------------------------------------------------------------------
# MODE 2 — Linear probe  (needs labels)
# ---------------------------------------------------------------------------

class BimodalClassifier(nn.Module):
    def __init__(self, in_dim, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, z):
        return self.out(F.relu(self.fc2(F.relu(self.fc1(z)))))


@torch.no_grad()
def extract_features(enc_adc, enc_t2, loader, device):
    feats, labels = [], []
    for adc, t2, label in loader:
        z = torch.cat([enc_adc(adc.to(device)), enc_t2(t2.to(device))], dim=1)
        feats.append(z.cpu()); labels.append(label)
    return torch.cat(feats), torch.cat(labels)


def run_linear_probe(enc_adc, enc_t2, adc_dir, t2_dir, train_list, test_list,
                     args, device, output_dir):
    print("\n[Mode: linear_probe]  Encoders FROZEN.")

    train_ds = LabeledPairedDataset(adc_dir, t2_dir, train_list)
    test_ds  = LabeledPairedDataset(adc_dir, t2_dir, test_list)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"  Extracting features  (train={len(train_ds)}, test={len(test_ds)}) ...")
    train_feats, train_labels = extract_features(enc_adc, enc_t2, train_loader, device)
    test_feats,  test_labels  = extract_features(enc_adc, enc_t2, test_loader,  device)

    classifier = BimodalClassifier(train_feats.size(1)).to(device)
    optimizer  = torch.optim.SGD(classifier.parameters(), lr=args.lr,
                                 weight_decay=1e-4, momentum=0.9)
    scheduler  = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    feat_iter = _infinite(DataLoader(TensorDataset(train_feats, train_labels),
                                     batch_size=args.batch_size, shuffle=True))

    import heapq
    best_acc = 0.0; all_acc = []
    print(f"  Training for {args.iters} iterations ...")

    for step in range(args.iters):
        classifier.train()
        feat, label = next(feat_iter)
        loss = F.cross_entropy(classifier(feat.to(device)), label.to(device))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 30 == 0: scheduler.step()

        if step % 20 == 0:
            classifier.eval()
            with torch.no_grad():
                preds = classifier(test_feats.to(device)).argmax(dim=1).cpu()
                acc   = (preds == test_labels).float().mean().item()
            all_acc.append(acc); best_acc = max(best_acc, acc)
            if acc > 0.8:
                print(f"  step {step:5d}  accuracy: {acc*100:.2f}%")
            else:
                print(f"  step {step:5d}")

    top10     = heapq.nlargest(10, all_acc)
    avg_top10 = float(np.mean(top10))
    std_top10 = float(np.std(top10))
    print(f"\n  Top-10 avg accuracy : {avg_top10*100:.2f}%  ±{std_top10*100:.2f}%")
    print(f"  Best accuracy       : {best_acc*100:.2f}%")

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, 'classification_results.txt')
    with open(result_path, 'w') as f:
        f.write('********* self-supervised linear_probe ************\n')
        f.write(f'mean acc (top-10): {avg_top10*100:.2f}%\n')
        f.write(f'std  acc (top-10): {std_top10*100:.2f}%\n')
        f.write(f'best acc         : {best_acc*100:.2f}%\n')
        f.write(f'\nComparison (paper Table III):\n')
        f.write(f'  Supervised       : 59.50 +/- 2.84 %\n')
        f.write(f'  Unsupervised     : 90.20 +/- 0.40 %\n')
        f.write(f'  Semi-supervised  : 93.00 +/- 0.45 %\n')
        f.write(f'  Self-supervised  : {avg_top10*100:.2f} +/- {std_top10*100:.2f} %\n')
    print(f"  Results saved → {result_path}")
    return avg_top10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infinite(loader):
    while True:
        for b in loader: yield b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device('cpu') if args.use_cpu else get_device()
    print(f"Using device: {device}")

    enc_adc, enc_t2 = load_encoders(args.checkpoint, args.z_dim, device)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SELF-SUPERVISED EVALUATION  [{args.mode}]")
    print(f"Output → {args.output_dir}")
    print(f"{'='*60}")

    if args.mode == 'retrieval':
        if not args.paired_list:
            raise ValueError("--paired_list is required for retrieval mode")
        top1, top3, top5, mean_rank = run_retrieval(
            enc_adc, enc_t2,
            args.adc_dir, args.t2_dir, args.paired_list,
            args.batch_size, device, args.output_dir)

        print(f"\n{'='*60}")
        print(f"  Top-1 retrieval accuracy : {top1*100:.1f}%")
        print(f"  Top-3 retrieval accuracy : {top3*100:.1f}%")
        print(f"  Top-5 retrieval accuracy : {top5*100:.1f}%")
        print(f"  Mean rank                : {mean_rank:.1f} / {len(PairedMedicalImageDataset(args.adc_dir, args.t2_dir, args.paired_list))}")
        print(f"{'='*60}")

    else:
        if not args.train_list or not args.test_list:
            raise ValueError("--train_list and --test_list are required for linear_probe mode")
        run_linear_probe(enc_adc, enc_t2,
                         args.adc_dir, args.t2_dir,
                         args.train_list, args.test_list,
                         args, device, args.output_dir)


if __name__ == '__main__':
    main()
