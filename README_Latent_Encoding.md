# Latent Encoding Workflow

This directory contains tools and scripts for preprocessing images into latent representations using Stable Diffusion VAE, which significantly speeds up training.

## Quick Start

### 1. Encode Images to Latents

```bash
# Encode all datasets at once
./latent-encoding/encode_all_datasets.sh

# Or encode individual datasets
python encode_latents.py -i data/recon -o latents/recon
python encode_latents.py -i data/scand -o latents/scand
python encode_latents.py -i data/tartan -o latents/tartan_drive
python encode_latents.py -i data/sacson -o latents/sacson
```

### 2. Train with Latents

Use the latent-optimized config files:

```bash
# For CDiT-B model
python train.py --config config/nwm_cdit_b_latents.yaml

# For CDiT-XL model  
python train.py --config config/nwm_cdit_xl_latents.yaml
```

## How It Works

### Preprocessing (`encode_latents.py`)
- Loads images from dataset directories
- Encodes them using Stable Diffusion VAE (`stabilityai/sd-vae-ft-ema`)
- Saves latent representations as `.pt` files
- Copies trajectory data (`.pkl` files) to latent directories
- Supports incremental encoding (skips already processed files)

### Training (`latent_dataset.py`)
- `LatentTrainingDataset`: Loads pre-encoded latents instead of raw images
- `LatentEvalDataset`: Evaluation dataset for latent-based models
- Automatic format detection and conversion via `_extract_latent()` method

### Configuration
- Use `latent_folder` instead of `data_folder` in config files
- Models automatically detect and use latent-based datasets
- Fallback to image-based datasets if `latent_folder` not specified

## Performance Benefits

- **Training Speed**: 3-5x faster due to eliminated real-time VAE encoding
- **Memory Efficiency**: Reduced GPU memory usage during training
- **Storage**: Latents are smaller than original images
- **Reproducibility**: Same latents used across training runs

## Directory Structure

```
latents/
├── scand/
│   ├── trajectory_001/
│   │   ├── 0.pt          # Encoded latent
│   │   ├── 1.pt
│   │   └── traj_data.pkl # Copied trajectory data
│   └── trajectory_002/
├── tartan_drive/
├── sacson/
└── recon/
```

## Requirements

- `diffusers` library for VAE model
- CUDA-compatible GPU (recommended)
- Sufficient storage for latent files (~20% of original image size)
