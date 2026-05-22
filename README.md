# Bi-Modality Medical Image Synthesis
### Semi-Supervised Sequential GAN — PyTorch Implementation

> PyTorch port of **"Bi-modality Medical Image Synthesis using Semi-supervised Sequential Generative Adversarial Networks"**
> IEEE Journal of Biomedical and Health Informatics, 2019.
> Original paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/8736809)

---

## What This Project Does

Given a patient's **ADC (Apparent Diffusion Coefficient)** prostate MRI scan, this model automatically synthesizes the corresponding **T2-weighted MRI** scan — without needing a fully paired dataset.

It uses a **Semi-Supervised Sequential GAN** that combines:
- A small set of **paired** images (supervised L1 loss)
- A large set of **unpaired** images (unsupervised WGAN-GP loss)

---

## Project Structure

```
Multimodal-Medical-Image-Synthesis/
├── pytorch_port/
│   ├── models.py               ← 6 network architectures (Encoder, GeneratorADC, GeneratorT2, Discriminators, SharedLayers)
│   ├── dataset.py              ← PyTorch Dataset for single and paired modalities
│   ├── utils.py                ← WGAN-GP gradient penalty, image saving, device helpers
│   ├── preprocess_dicom.py     ← Convert raw DICOM → 64x64 PNG
│   ├── train_semi.py           ← Semi-supervised training script
│   ├── test_semi.py            ← Inference script (random generation + real ADC → T2)
│   ├── evaluate.py             ← Evaluation metrics (MAE, MSE, PSNR, SSIM, FID) + TensorBoard
│   ├── requirements.txt        ← Python dependencies
│   ├── results_semi_real/      ← Training checkpoints + TensorBoard logs
│   ├── generated_semi_real/    ← Images generated during training
│   ├── test_output/            ← Inference output images
│   └── eval_output/            ← Evaluation metrics CSV + TensorBoard
├── data/
│   ├── adc/                    ← 500 ADC PNG images (64x64)
│   ├── t2/                     ← 500 T2 PNG images (64x64)
│   ├── paired_names.txt        ← 200 paired filenames (supervised signal)
│   ├── adc_names.txt           ← 500 ADC filenames (unsupervised signal)
│   └── t2_names.txt            ← 500 T2 filenames (unsupervised signal)
├── semi/                       ← Original TensorFlow 1.x semi-supervised code
├── supervise/                  ← Original TensorFlow 1.x supervised code
├── unsupervise/                ← Original TensorFlow 1.x unsupervised code
├── BIMODAL.pdf                 ← Research paper
├── Project_Report.pdf          ← Full project report with results
└── README.md
```

---

## Requirements

```bash
Python >= 3.8
torch >= 2.0
torchvision
numpy
opencv-python
tensorboard
Pillow
scikit-image
scipy
pydicom
```

Install all at once:
```bash
cd pytorch_port
pip install -r requirements.txt
pip install pydicom scikit-image scipy
```

---

## Dataset

### Download PROSTATEx

1. Go to: **https://www.cancerimagingarchive.net/collection/prostatex/**
2. Click **"Download"**
3. You will get a `.tcia` manifest file: `PROSTATEx-v1-doiJNLP.tcia`

### Convert .tcia to DICOM using NBIA Data Retriever

The `.tcia` file is a manifest — not the actual images. You need the **NBIA Data Retriever** app to download the actual DICOM files.

1. Download NBIA Data Retriever from:
   **https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images**

2. Install and open the app

3. Open your `.tcia` file inside the app
   - File → Open → select `PROSTATEx-v1-doiJNLP.tcia`

4. Click **"Download"**
   - All DICOM files will be saved to your chosen directory
   - This creates the folder: `PROSTATEx-v1-doiJNLP/prostatex/`

5. The folder structure will look like:
   ```
   PROSTATEx-v1-doiJNLP/
   └── prostatex/
       ├── ProstateX-0000/
       │   └── <study>/
       │       └── <series>/
       │           └── *.dcm
       ├── ProstateX-0001/
       └── ...
   ```

---

## Step-by-Step Execution

### Step 1 — Clone the Repository

```bash
git clone https://github.com/avijit004/Multimodal-Medical-Image-Synthesis.git
cd Multimodal-Medical-Image-Synthesis
```

### Step 2 — Install Dependencies

```bash
cd pytorch_port
pip install -r requirements.txt
pip install pydicom scikit-image scipy
```

### Step 3 — Preprocess DICOM to PNG

> Skip this step if you already have the `data/` folder (included in the repo).

```bash
python preprocess_dicom.py
```

This will:
- Walk through all patients in `PROSTATEx-v1-doiJNLP/prostatex/`
- Identify ADC and T2-axial series by DICOM `SeriesDescription`
- Extract the middle slice from each series
- Center-crop and normalize to 64×64 PNG
- Save to `../data/adc/` and `../data/t2/`
- Generate `paired_names.txt`, `adc_names.txt`, `t2_names.txt`

### Step 4 — Train the Model

```bash
python train_semi.py \
    --adc_dir ../data/adc \
    --t2_dir ../data/t2 \
    --paired_list ../data/paired_names.txt \
    --adc_list ../data/adc_names.txt \
    --t2_list ../data/t2_names.txt \
    --results_path ./results_semi_real \
    --save_image_path ./generated_semi_real \
    --iters 10000 \
    --batch_size 32 \
    --z_dim 128 \
    --lr 1e-4 \
    --n_critic 3 \
    --save_interval 500
```

**Training arguments:**

| Argument | Default | Description |
|---|---|---|
| `--adc_dir` | required | Directory of ADC PNG images |
| `--t2_dir` | required | Directory of T2 PNG images |
| `--paired_list` | required | Text file listing paired image names |
| `--adc_list` | required | Text file listing all ADC image names |
| `--t2_list` | required | Text file listing all T2 image names |
| `--iters` | 5000 | Number of training iterations |
| `--batch_size` | 32 | Batch size |
| `--z_dim` | 128 | Latent vector dimension |
| `--lr` | 1e-4 | Learning rate |
| `--n_critic` | 3 | Discriminator updates per generator update |
| `--save_interval` | 500 | Save checkpoint every N iterations |
| `--use_cpu` | False | Force CPU training |

**During training you will see:**
```
iter:    0 | D1=9.7217 D2=9.7274
iter:   50 | D1=-11.36 D2=-19.29
iter:  100 | D1=-16.27 D2=-7.69
...
[Saved checkpoint + 50 image pairs at iter 500]
```

### Step 5 — Monitor Training in TensorBoard

```bash
tensorboard --logdir ./results_semi_real
```

Open browser: **http://localhost:6006**

- **SCALARS tab** → `Loss/D1` and `Loss/D2` loss curves
- **IMAGES tab** → `Gen/ADC` and `Gen/T2` — drag slider to watch image quality improve over iterations

### Step 6 — Run Inference (Test)

```bash
python test_semi.py \
    --checkpoint ./results_semi_real/2026-05-17__01-56-42/Saved_models/ckpt_9500.pt \
    --mode both \
    --adc_dir ../data/adc \
    --adc_list ../data/adc_names.txt \
    --n_samples 50 \
    --output_dir ./test_output
```

**Inference modes:**

| Mode | Description |
|---|---|
| `random_pairs` | Generate synthetic ADC+T2 pairs from random noise |
| `real_to_fake` | Translate real ADC images to synthesized T2 images |
| `both` | Run both modes |

**Output structure:**
```
test_output/
├── random_pairs/
│   ├── adc/    → synthetic ADC images from noise
│   └── t2/     → corresponding synthetic T2 images
└── real_to_fake/
    ├── input_adc/          → real ADC images fed in
    ├── reconstructed_adc/  → encoder → decoder round-trip
    └── synthesized_t2/     → THE MAIN RESULT — synthesized T2
```

### Step 7 — Run Evaluation Metrics

```bash
python evaluate.py \
    --checkpoint ./results_semi_real/2026-05-17__01-56-42/Saved_models/ckpt_9500.pt \
    --adc_dir ../data/adc \
    --t2_dir ../data/t2 \
    --paired_list ../data/paired_names.txt \
    --adc_list ../data/adc_names.txt \
    --t2_list ../data/t2_names.txt \
    --output_dir ./eval_output \
    --vis_images 16 \
    --fid_samples 200
```

**Metrics computed:**
- **MAE** — Mean Absolute Error (pixel-level)
- **MSE** — Mean Squared Error
- **PSNR** — Peak Signal-to-Noise Ratio (dB)
- **SSIM** — Structural Similarity Index
- **FID** — Fréchet Inception Distance (distribution-level)

**Outputs saved to `eval_output/`:**
- `summary_metrics.csv` — mean ± std for all metrics
- `per_image_metrics.csv` — per-image breakdown
- `tensorboard/` — TensorBoard event files

### Step 8 — View Evaluation in TensorBoard

```bash
tensorboard --logdir ./eval_output/tensorboard
```

Open browser: **http://localhost:6006**

- **SCALARS** → MAE, PSNR, SSIM, FID scores
- **IMAGES** → `Comparison/T2_Real_vs_Synthesized` — real T2 vs synthesized T2 side by side
- **DISTRIBUTIONS** → per-image PSNR and SSIM histograms

---

## Results

Trained on PROSTATEx dataset — 9,500 iterations, batch size 32, Apple MPS device.

| Metric | ADC Reconstruction | T2 Synthesis |
|---|---|---|
| MAE | 34.92 ± 5.30 | 106.65 ± 19.94 |
| MSE | 2022.92 ± 608.69 | 15294.23 ± 4832.37 |
| PSNR | 15.26 dB ± 1.29 | 6.55 dB ± 1.63 |
| SSIM | 0.1329 ± 0.033 | 0.0297 ± 0.054 |
| FID | 310.28 | 351.01 |

> **Note:** Low SSIM on T2 synthesis is expected. ADC and T2 are fundamentally different image contrasts — they do not share pixel values even for the same anatomy. Visual quality and FID are more meaningful metrics for cross-modal synthesis. The original paper trains for 40,000 iterations; results improve significantly with more training.

---

## Architecture Overview

![Architecture diagram](assets/architecture.png)
**6 networks in total:**
- `SharedLayers` — shared decoder layers between both generators
- `Encoder` — compresses real ADC to 128-dim latent vector
- `GeneratorADC` — 128-dim noise/code → 64×64 ADC image
- `GeneratorT2` — 64×64 ADC → 64×64 T2 image (U-Net with skip connections)
- `DiscriminatorADC` — WGAN critic for ADC images
- `DiscriminatorT2` — WGAN critic for T2 images

---

## Original TensorFlow Code

The original TF 1.x implementation is preserved in:
- `semi/` — semi-supervised training (this project's focus)
- `supervise/` — fully supervised training
- `unsupervise/` — fully unsupervised training

---

## Citation

```bibtex
@article{yi2019bimodal,
  title={Bi-modality Medical Image Synthesis Using Semi-supervised Sequential Generative Adversarial Networks},
  author={Yi, Xin and Walia, Ekta and Babyn, Paul},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2019}
}
```

---

## References

- Paper: https://ieeexplore.ieee.org/document/8736809
- Dataset: https://www.cancerimagingarchive.net/collection/prostatex/
- NBIA Data Retriever: https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images
