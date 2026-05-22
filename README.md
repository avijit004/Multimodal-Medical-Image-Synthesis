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
<svg width="100%" viewBox="0 0 680 760" role="img" style="" xmlns="http://www.w3.org/2000/svg"><title style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">Self-supervised paired ADC and T2 image synthesis architecture</title><desc style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">Two paths: a generation path where random noise feeds shared layers driving GeneratorADC and GeneratorT2 to produce spatially aligned synthetic ADC and T2 images, and a reconstruction path where a real ADC image is encoded to a latent vector then decoded back into a reconstructed ADC and a synthesized T2, supervised by L1 loss against real paired images.</desc>
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker>
</defs>

<text x="40" y="34" style="fill:rgb(250, 249, 245);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:start;dominant-baseline:auto">Path A — generation from noise</text>

<g onclick="sendPrompt('Why sample noise from a standard normal distribution?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="40" y="50" width="180" height="56" rx="8" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="130" y="72" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Random noise z</text>
<text x="130" y="90" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">z ~ N(0, I)</text>
</g>

<line x1="130" y1="106" x2="130" y2="146" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<g onclick="sendPrompt('What are the shared layers and how do they guarantee alignment?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="40" y="146" width="180" height="56" rx="8" stroke-width="0.5" style="fill:rgb(60, 52, 137);stroke:rgb(175, 169, 236);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="130" y="168" text-anchor="middle" dominant-baseline="central" style="fill:rgb(206, 203, 246);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Shared layers</text>
<text x="130" y="186" text-anchor="middle" dominant-baseline="central" style="fill:rgb(175, 169, 236);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">same weights, both gens</text>
</g>

<line x1="220" y1="174" x2="290" y2="146" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<line x1="220" y1="186" x2="290" y2="214" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<g onclick="sendPrompt('What does GeneratorADC do?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="290" y="118" width="170" height="56" rx="8" stroke-width="0.5" style="fill:rgb(8, 80, 65);stroke:rgb(93, 202, 165);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="375" y="140" text-anchor="middle" dominant-baseline="central" style="fill:rgb(159, 225, 203);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">GeneratorADC</text>
<text x="375" y="158" text-anchor="middle" dominant-baseline="central" style="fill:rgb(93, 202, 165);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">z to 64x64</text>
</g>

<g onclick="sendPrompt('What does GeneratorT2 do?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="290" y="186" width="170" height="56" rx="8" stroke-width="0.5" style="fill:rgb(8, 80, 65);stroke:rgb(93, 202, 165);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="375" y="208" text-anchor="middle" dominant-baseline="central" style="fill:rgb(159, 225, 203);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">GeneratorT2</text>
<text x="375" y="226" text-anchor="middle" dominant-baseline="central" style="fill:rgb(93, 202, 165);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">to 64x64</text>
</g>

<line x1="460" y1="146" x2="510" y2="146" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<line x1="460" y1="214" x2="510" y2="214" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<g onclick="sendPrompt('What is a synthetic ADC image?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="510" y="120" width="140" height="52" rx="8" stroke-width="0.5" style="fill:rgb(99, 56, 6);stroke:rgb(239, 159, 39);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="580" y="146" text-anchor="middle" dominant-baseline="central" style="fill:rgb(250, 199, 117);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Synthetic ADC</text>
</g>

<g onclick="sendPrompt('What is a synthetic T2 image?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="510" y="188" width="140" height="52" rx="8" stroke-width="0.5" style="fill:rgb(99, 56, 6);stroke:rgb(239, 159, 39);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="580" y="214" text-anchor="middle" dominant-baseline="central" style="fill:rgb(250, 199, 117);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Synthetic T2</text>
</g>

<text x="290" y="270" dominant-baseline="central" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:central">Shared weights make the two outputs spatially aligned</text>

<line x1="40" y1="298" x2="640" y2="298" stroke="var(--t)" stroke-width="0.5" stroke-dasharray="4 4" style="fill:rgb(0, 0, 0);stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-dasharray:4px, 4px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<text x="40" y="334" style="fill:rgb(250, 249, 245);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:start;dominant-baseline:auto">Path B — reconstruction from a real ADC</text>

<g onclick="sendPrompt('What is the real paired ADC input?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="40" y="350" width="150" height="52" rx="8" stroke-width="0.5" style="fill:rgb(12, 68, 124);stroke:rgb(133, 183, 235);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="115" y="376" text-anchor="middle" dominant-baseline="central" style="fill:rgb(181, 212, 244);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Real ADC</text>
</g>

<line x1="190" y1="376" x2="240" y2="376" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<g onclick="sendPrompt('What does the encoder learn to compress?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="240" y="350" width="150" height="52" rx="8" stroke-width="0.5" style="fill:rgb(60, 52, 137);stroke:rgb(175, 169, 236);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="315" y="376" text-anchor="middle" dominant-baseline="central" style="fill:rgb(206, 203, 246);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Encoder</text>
</g>

<line x1="390" y1="376" x2="440" y2="376" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<g onclick="sendPrompt('What is z_encoded and how does it relate to the noise z?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="440" y="350" width="170" height="56" rx="8" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="525" y="372" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">z_encoded</text>
<text x="525" y="390" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">latent vector</text>
</g>

<line x1="525" y1="406" x2="525" y2="446" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<g onclick="sendPrompt('How is the same GeneratorADC reused on the encoded latent?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="290" y="446" width="170" height="56" rx="8" stroke-width="0.5" style="fill:rgb(8, 80, 65);stroke:rgb(93, 202, 165);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="375" y="468" text-anchor="middle" dominant-baseline="central" style="fill:rgb(159, 225, 203);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">GeneratorADC</text>
<text x="375" y="486" text-anchor="middle" dominant-baseline="central" style="fill:rgb(93, 202, 165);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">shared weights</text>
</g>

<path d="M525 446 L525 426 L460 426 L460 474 L460 474" fill="none" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<line x1="525" y1="446" x2="525" y2="500" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<g onclick="sendPrompt('How is the same GeneratorT2 reused on the encoded latent?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="290" y="546" width="170" height="56" rx="8" stroke-width="0.5" style="fill:rgb(8, 80, 65);stroke:rgb(93, 202, 165);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="375" y="568" text-anchor="middle" dominant-baseline="central" style="fill:rgb(159, 225, 203);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">GeneratorT2</text>
<text x="375" y="586" text-anchor="middle" dominant-baseline="central" style="fill:rgb(93, 202, 165);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">shared weights</text>
</g>

<line x1="290" y1="474" x2="150" y2="474" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<g onclick="sendPrompt('What is the reconstructed ADC compared against?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="40" y="448" width="110" height="52" rx="8" stroke-width="0.5" style="fill:rgb(99, 56, 6);stroke:rgb(239, 159, 39);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="95" y="466" text-anchor="middle" dominant-baseline="central" style="fill:rgb(250, 199, 117);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Reconst.</text>
<text x="95" y="484" text-anchor="middle" dominant-baseline="central" style="fill:rgb(250, 199, 117);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">ADC</text>
</g>

<line x1="290" y1="574" x2="150" y2="574" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<g onclick="sendPrompt('What is the synthesized T2 compared against?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="40" y="548" width="110" height="52" rx="8" stroke-width="0.5" style="fill:rgb(99, 56, 6);stroke:rgb(239, 159, 39);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="95" y="566" text-anchor="middle" dominant-baseline="central" style="fill:rgb(250, 199, 117);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Synth.</text>
<text x="95" y="584" text-anchor="middle" dominant-baseline="central" style="fill:rgb(250, 199, 117);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">T2</text>
</g>

<line x1="95" y1="500" x2="95" y2="640" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<line x1="95" y1="600" x2="95" y2="640" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<g onclick="sendPrompt('How does the L1 loss against real paired images train this network?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="40" y="640" width="300" height="56" rx="8" stroke-width="0.5" style="fill:rgb(121, 31, 31);stroke:rgb(240, 149, 149);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="190" y="662" text-anchor="middle" dominant-baseline="central" style="fill:rgb(247, 193, 193);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">L1 loss vs real paired images</text>
<text x="190" y="680" text-anchor="middle" dominant-baseline="central" style="fill:rgb(240, 149, 149);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">supervises both reconstruction and synthesis</text>
</g>

<line x1="525" y1="502" x2="525" y2="546" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<line x1="525" y1="546" x2="525" y2="574 " stroke="none" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<path d="M525 502 L525 660 L340 660" fill="none" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
</svg>
<img width="134" height="150" alt="paired_adc_t2_generator_encoder_architecture" src="https://github.com/user-attachments/assets/31ae2624-c8d5-4dfc-b763-f1afaa308541" />


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
