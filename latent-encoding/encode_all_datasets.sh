#!/bin/bash

# Script to encode images to latents for all datasets with batch processing
# Usage: ./encode_all_datasets.sh [batch_size]
# Default batch size: 32

BATCH_SIZE=${1:-32}  # Use first argument or default to 32
VAE_MODEL="stabilityai/sd-vae-ft-ema"

echo "Starting latent encoding for all datasets with batch size: $BATCH_SIZE"

# Create latents directory if it doesn't exist
mkdir -p latents

# Encode each dataset with enhanced error handling and batch processing
echo "Encoding recon dataset..."
python encode_latents.py -i data/recon -o latents/recon --vae-model $VAE_MODEL --batch-size $BATCH_SIZE
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to encode recon dataset"
    exit 1
fi

echo "Encoding scand dataset..."
python encode_latents.py -i data/scand -o latents/scand --vae-model $VAE_MODEL --batch-size $BATCH_SIZE
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to encode scand dataset"
    exit 1
fi

echo "Encoding tartan_drive dataset..."
python encode_latents.py -i data/tartan -o latents/tartan_drive --vae-model $VAE_MODEL --batch-size $BATCH_SIZE
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to encode tartan_drive dataset"
    exit 1
fi

echo "Encoding sacson dataset..."
python encode_latents.py -i data/sacson -o latents/sacson --vae-model $VAE_MODEL --batch-size $BATCH_SIZE
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to encode sacson dataset"
    exit 1
fi

echo "All datasets encoded successfully with batch processing!"
echo "Latent files saved in latents/ directory"
echo "Used batch size: $BATCH_SIZE"
