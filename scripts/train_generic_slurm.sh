#!/bin/bash
#SBATCH --job-name=nwm_training
#SBATCH --partition=<YOUR_PARTITION>        # Replace with your cluster partition
#SBATCH --gpus=1                           # Number of GPUs (adjust as needed)
#SBATCH --time=23:50:00                    # Time limit (adjust as needed)
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks-per-node=1                # Tasks per node
#SBATCH --cpus-per-task=8                  # CPU cores per task
#SBATCH --output=logs/slurm_outputs/job_%j.out
#SBATCH --error=logs/slurm_outputs/job_%j.err

# =============================================================================
# Generic SLURM Training Script for NWM
# =============================================================================
# Description: Template for training Navigation World Models on SLURM clusters
# Instructions: 
#   1. Replace <YOUR_PARTITION> with your cluster partition name
#   2. Replace <YOUR_CONDA_PATH> with your conda installation path
#   3. Replace <YOUR_PROJECT_PATH> with your NWM project directory
#   4. Adjust SLURM parameters above as needed for your cluster
#   5. Run with: sbatch scripts/train_generic_slurm.sh
# =============================================================================

echo "🚀 =============================================="
echo "🚀 NWM Training on SLURM Cluster"
echo "🚀 =============================================="
echo "✅ Job started on node: $(hostname)"
echo "🕒 Start time: $(date)"
echo "🎯 Job ID: $SLURM_JOB_ID"

# =============================================================================
# User Configuration - MODIFY THESE PATHS
# =============================================================================
CONDA_PATH="~/miniconda3"                  # Update with your conda path
PROJECT_PATH="${HOME}/boyuan/nwm"          # Update with your project path  
CONDA_ENV="nwm-env"                        # Your conda environment name
CONFIG_FILE="config/nwm_cdit_xl.yaml"     # Training config file

# =============================================================================
# Environment Setup
# =============================================================================
echo "🔧 Setting up environment..."

# Load conda
source ${CONDA_PATH}/etc/profile.d/conda.sh || {
    echo "❌ Failed to load conda from: ${CONDA_PATH}"
    echo "💡 Please update CONDA_PATH in this script"
    exit 1
}

# Activate environment
conda activate ${CONDA_ENV} || {
    echo "❌ Failed to activate conda environment: ${CONDA_ENV}"
    echo "💡 Make sure you have run: ./setup_nwm_env.sh"
    exit 1
}

# Change to project directory
cd ${PROJECT_PATH} || {
    echo "❌ Project directory not found: ${PROJECT_PATH}"
    echo "💡 Please update PROJECT_PATH in this script"
    exit 1
}

# Create necessary directories
mkdir -p logs/slurm_outputs

# =============================================================================
# Pre-training Checks
# =============================================================================
echo "🔍 Running pre-training checks..."

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  nvidia-smi not available"
fi

# Check config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    echo "📋 Available configs:"
    ls config/*.yaml
    exit 1
fi

# =============================================================================
# Start Training
# =============================================================================
echo "🚀 Starting NWM training..."
echo "⚙️  Configuration: $CONFIG_FILE"

# Training command - modify parameters as needed
python train.py \
    --config "$CONFIG_FILE" \
    --device cuda \
    --ckpt-every 2000 \
    --eval-every 10000 \
    --bfloat16 1 \
    --epochs 300 \
    --torch-compile 0

TRAINING_EXIT_CODE=$?

# =============================================================================
# Post-training Summary
# =============================================================================
echo "🏁 Training complete at: $(date)"
echo "📊 Exit code: $TRAINING_EXIT_CODE"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training successful!"
else
    echo "❌ Training failed!"
fi

exit $TRAINING_EXIT_CODE
