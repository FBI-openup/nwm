#!/bin/bash
#SBATCH --job-name=nwm_hybrid_l40s_training
#SBATCH --partition=ENSTA-l40s
#SBATCH --gpus=1
#SBATCH --time=23:50:00
#SBATCH --nodelist=ensta-l40s01.r2.enst.fr
#SBATCH --output=logs/slurm_outputs/l40s_job_%j.out
#SBATCH --error=logs/slurm_outputs/l40s_job_%j.err

# =============================================================================
# SLURM Training Script for NWM on L40S Cluster
# =============================================================================
# Description: Training script for Navigation World Models with Hybrid CDiT
# Target: ENSTA L40S cluster with GPU support
# Author: NWM Team
# Date: August 2025
# =============================================================================

echo "🚀 =============================================="
echo "🚀 NWM Hybrid CDiT Training on L40S Cluster"
echo "🚀 =============================================="
echo "✅ Job started on node: $(hostname)"
echo "🕒 Start time: $(date)"
echo "🎯 Job ID: $SLURM_JOB_ID"
echo "📊 Allocated GPUs: $SLURM_GPUS"

# =============================================================================
# Environment Setup
# =============================================================================
echo "🔧 Setting up environment..."

# Load conda environment
echo "📦 Loading conda environment..."
# Try multiple common conda locations
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    # If conda is already in PATH, try to use it directly
    echo "✅ Conda found in PATH"
else
    echo "❌ Could not find conda installation"
    echo "💡 Please ensure conda is installed and accessible"
    exit 1
fi

# Initialize conda/mamba shell environment
echo "🔄 Initializing conda shell environment..."
eval "$(conda shell.bash hook)" || {
    echo "⚠️  Failed to initialize conda shell hook, trying manual initialization..."
    # Fallback: try to initialize mamba if available
    if command -v mamba &> /dev/null; then
        eval "$(mamba shell.bash hook)"
    fi
}

echo "🐍 Activating nwm-env conda environment..."
conda activate nwm-env || { 
    echo "❌ Failed to activate conda environment 'nwm-env'"; 
    echo "💡 Make sure you have run: ./scripts/setup_nwm_env.sh"; 
    exit 1; 
}

# Change to project directory
echo "📂 Changing to project directory..."
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Go to the parent directory (nwm root)
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR" || { 
    echo "❌ Failed to change to project directory: $PROJECT_DIR"; 
    echo "💡 Please ensure this script is in the scripts/ subdirectory of your NWM repository"; 
    exit 1; 
}
echo "✅ Working in project directory: $(pwd)"

# Create log directories if they don't exist
mkdir -p logs/slurm_outputs
mkdir -p logs/nwm_cdit_xl/checkpoints
mkdir -p logs/nwm_cdit_xl/hybrid_logs  # Additional logs for hybrid model

# =============================================================================
# Pre-training Checks
# =============================================================================
echo "🔍 Running pre-training checks..."

# Check if config file exists
CONFIG_FILE="config/hybrid_l40s_inference_memory.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    echo "📋 Available config files:"
    ls -la config/*.yaml
    exit 1
else
    echo "✅ Using hybrid config: $CONFIG_FILE"
fi

# Check GPU availability
echo "🎮 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

# Check data directory
echo "📁 Checking data directory..."
if [ -d "data" ]; then
    echo "✅ Data directory found. Available datasets:"
    ls -la data/
else
    echo "⚠️  Data directory not found. Please ensure datasets are in ./data/"
fi

# Check if pretrained models exist
echo "🏗️  Checking for pretrained models..."
if [ -d "logs/nwm_cdit_xl/checkpoints" ] && [ "$(ls -A logs/nwm_cdit_xl/checkpoints)" ]; then
    echo "✅ Pretrained models found:"
    ls -la logs/nwm_cdit_xl/checkpoints/
else
    echo "⚠️  No pretrained models found in logs/nwm_cdit_xl/checkpoints/"
    echo "💡 Download models from: https://huggingface.co/facebook/nwm"
fi

# =============================================================================
# Training Configuration
# =============================================================================
echo "⚙️  Training Configuration:"
echo "├── Config: $CONFIG_FILE"
echo "├── Node: $(hostname)"
echo "├── GPU: $SLURM_GPUS"
echo "├── Time limit: 23:50:00"
echo "└── Output logs: logs/slurm_outputs/"

# =============================================================================
# Start Training
# =============================================================================
echo "🚀 Starting Hybrid CDiT model training..."
echo "⏰ Training started at: $(date)"

# Main training command - Hybrid CDiT with WorldMem
python train.py --config "$CONFIG_FILE" \
    --device cuda \
    --ckpt-every 2000 \
    --eval-every 10000 \
    --bfloat16 1 \
    --epochs 300 \
    --torch-compile 0 \
    --hybrid-mode \
    --memory-enabled

# Capture training exit code
TRAINING_EXIT_CODE=$?

# =============================================================================
# Post-training Summary
# =============================================================================
echo "🏁 =============================================="
echo "🏁 Training Session Complete"
echo "🏁 =============================================="
echo "🕓 End time: $(date)"
echo "⏱️  Total GPU time used: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📊 Final model checkpoints:"
    ls -la logs/nwm_cdit_xl/checkpoints/ | tail -5
else
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "🔍 Check error logs: logs/slurm_outputs/l40s_job_${SLURM_JOB_ID}.err"
fi

echo "📋 Job Summary:"
echo "├── Job ID: $SLURM_JOB_ID"
echo "├── Node: $(hostname)"
echo "├── Exit Code: $TRAINING_EXIT_CODE"
echo "└── Log files: logs/slurm_outputs/l40s_job_${SLURM_JOB_ID}.{out,err}"

exit $TRAINING_EXIT_CODE
