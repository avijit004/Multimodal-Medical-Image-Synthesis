import argparse, os, numpy as np, torch
from torch.utils.tensorboard import SummaryWriter
from models import SharedLayers, Encoder, GeneratorADC, GeneratorT2, DiscriminatorADC, DiscriminatorT2
from dataset import get_dataloader, get_paired_dataloader
from utils import compute_gradient_penalty, save_generated_images, form_results, infinite_dataloader, get_device

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--adc_dir', required=True); p.add_argument('--t2_dir', required=True)
    p.add_argument('--paired_list', required=True); p.add_argument('--adc_list', required=True)
    p.add_argument('--t2_list', required=True)
    p.add_argument('--results_path', default='./results_semi')
    p.add_argument('--save_image_path', default='./generated_semi')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--iters', type=int, default=5000)
    p.add_argument('--z_dim', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--n_critic', type=int, default=3)
    p.add_argument('--save_interval', type=int, default=500)
    p.add_argument('--log_interval', type=int, default=50)
    p.add_argument('--use_cpu', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device('cpu') if args.use_cpu else get_device()
    print(f"Using device: {device}")

    paired_loader = get_paired_dataloader(args.adc_dir, args.t2_dir, args.paired_list, batch_size=args.batch_size)
    paired_iter = infinite_dataloader(paired_loader)
    adc_iter = infinite_dataloader(get_dataloader(args.adc_dir, args.adc_list, batch_size=args.batch_size))
    t2_iter = infinite_dataloader(get_dataloader(args.t2_dir, args.t2_list, batch_size=args.batch_size))

    shared = SharedLayers().to(device)
    encoder = Encoder(args.z_dim).to(device)
    gen_adc = GeneratorADC(shared, args.z_dim).to(device)
    gen_t2 = GeneratorT2(shared).to(device)
    disc_adc = DiscriminatorADC().to(device)
    disc_t2 = DiscriminatorT2().to(device)

    g1_params = list(shared.parameters()) + list(gen_adc.parameters()) + list(encoder.parameters())
    g2_params = list(shared.parameters()) + list(gen_t2.parameters())
    opt_d1 = torch.optim.Adam(disc_adc.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_d2 = torch.optim.Adam(disc_t2.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_g1 = torch.optim.Adam(g1_params, lr=args.lr, betas=(0.0, 0.9))
    opt_g2 = torch.optim.Adam(g2_params, lr=args.lr, betas=(0.0, 0.9))
    opt_l1_adc = torch.optim.Adam(g1_params, lr=args.lr, betas=(0.0, 0.9))
    opt_l1_t2 = torch.optim.Adam(g2_params, lr=args.lr, betas=(0.0, 0.9))

    tb_path, model_path, log_path = form_results(args.results_path)
    os.makedirs(args.save_image_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_path)

    print(f"\n{'='*60}\nSEMI-SUPERVISED SEQUENTIAL GAN TRAINING\n{'='*60}")
    print(f"Iterations: {args.iters} | Batch: {args.batch_size} | LR: {args.lr} | Device: {device}\n")

    for i in range(args.iters):
        z = torch.randn(args.batch_size, args.z_dim, device=device)
        real_adc = next(adc_iter).to(device)
        real_t2 = next(t2_iter).to(device)

        # Train Discriminators
        with torch.no_grad():
            fa = gen_adc(z); ft = gen_t2(fa)

        disc_adc.zero_grad()
        d1_loss = disc_adc(fa.detach()).mean() - disc_adc(real_adc).mean() + compute_gradient_penalty(disc_adc, real_adc, fa.detach(), device)
        d1_loss.backward(); opt_d1.step()

        disc_t2.zero_grad()
        d2_loss = disc_t2(ft.detach()).mean() - disc_t2(real_t2).mean() + compute_gradient_penalty(disc_t2, real_t2, ft.detach(), device)
        d2_loss.backward(); opt_d2.step()

        # Train Generators (every n_critic steps)
        if i % args.n_critic == 0:
            # Unsupervised: adversarial loss
            z2 = torch.randn(args.batch_size, args.z_dim, device=device)
            fa2 = gen_adc(z2)
            opt_g1.zero_grad()
            (-disc_adc(fa2).mean()).backward(); opt_g1.step()

            for _ in range(2):
                z3 = torch.randn(args.batch_size, args.z_dim, device=device)
                opt_g2.zero_grad()
                (-disc_t2(gen_t2(gen_adc(z3))).mean()).backward(); opt_g2.step()

            # Supervised: L1 reconstruction loss
            pa, pt = next(paired_iter)
            pa, pt = pa.to(device), pt.to(device)
            rz = encoder(pa); ra = gen_adc(rz); rt = gen_t2(ra)

            opt_l1_adc.zero_grad()
            l1a = torch.mean(torch.abs(pa - ra))
            l1a.backward(retain_graph=True); opt_l1_adc.step()

            rz = encoder(pa); ra = gen_adc(rz); rt = gen_t2(ra)
            opt_l1_t2.zero_grad()
            l1t = torch.mean(torch.abs(pt - rt))
            l1t.backward(); opt_l1_t2.step()

        if i % args.log_interval == 0:
            with torch.no_grad():
                zl = torch.randn(4, args.z_dim, device=device)
                fal = gen_adc(zl); ftl = gen_t2(fal)
            print(f"iter:{i:5d} | D1={d1_loss.item():.4f} D2={d2_loss.item():.4f}")
            writer.add_scalar('Loss/D1', d1_loss.item(), i)
            writer.add_scalar('Loss/D2', d2_loss.item(), i)
            writer.add_images('Gen/ADC', (fal+1)/2, i)
            writer.add_images('Gen/T2', (ftl+1)/2, i)

        if i > 0 and i % args.save_interval == 0:
            torch.save({'iter': i, 'shared': shared.state_dict(), 'encoder': encoder.state_dict(),
                        'gen_adc': gen_adc.state_dict(), 'gen_t2': gen_t2.state_dict()},
                       os.path.join(model_path, f'semi_ckpt_{i}.pt'))
            with torch.no_grad():
                gz = torch.randn(50, args.z_dim, device=device)
                ga = gen_adc(gz).cpu().numpy(); gt = gen_t2(gen_adc(gz)).cpu().numpy()
            sd = os.path.join(args.save_image_path, str(i))
            save_generated_images(ga, os.path.join(sd, 'adc'))
            save_generated_images(gt, os.path.join(sd, 't2'))
            print(f"  [Saved checkpoint + 50 image pairs at iter {i}]")

    writer.close()
    print(f"\nTraining complete! Results: {args.results_path}")

if __name__ == '__main__':
    main()
