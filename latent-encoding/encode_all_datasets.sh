#!/bin/bash

# Script to encode images to latents for all datasets
# Usage: ./encode_all_datasets.sh

echo "Starting latent encoding for all datasets..."

# Create latents directory if it doesn't exist
mkdir -p latents

# Encode each dataset
echo "Encoding scand dataset..."
python encode_latents.py -i data/scand -o latents/scand --vae-model stabilityai/sd-vae-ft-ema

echo "Encoding tartan_drive dataset..."
python encode_latents.py -i data/tartan -o latents/tartan_drive --vae-model stabilityai/sd-vae-ft-ema

echo "Encoding sacson dataset..."
python encode_latents.py -i data/sacson -o latents/sacson --vae-model stabilityai/sd-vae-ft-ema

echo "Encoding recon dataset..."
python encode_latents.py -i data/recon -o latents/recon --vae-model stabilityai/sd-vae-ft-ema

echo "All datasets encoded successfully!"
echo "Latent files saved in latents/ directory"
