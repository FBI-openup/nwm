#!/bin/bash

################################################################################
# Usage Instructions:
# This script is intended to create and configure a Conda environment 
# tailored for the NWM project (Python 3.10).
#
# - Please ensure that conda and Mamba is installed on your system.
# - The script will automatically create a Conda environment named "nwm-env" 
#   and install all required dependencies then activate it
#
# Usage (run in terminal on Linux or via WSL2 on Windows):
# 1. Save this file as setup_nwm_env.sh
# 2. Grant execution permission: chmod +x setup_nwm_env.sh
# 3. Execute the script: ./setup_nwm_env.sh  or bash setup_nwm_env.sh
#
# ⚠️ Note: This script installs the PyTorch nightly build with CUDA 12.6 support.
################################################################################


#Stop after any error
set -e

echo "Step 1: Create new conda environment (nwm-env, Python 3.10)"
mamba create -y -n nwm-env python=3.10

echo "Step 2: Activate the environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
eval "$(mamba shell hook --shell bash)"
mamba activate nwm-env

echo "Step 3: Install PyTorch, torchvision, torchaudio with CUDA 12.6 using pip"
pip3 install pyyaml typeguard --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu126

echo "Step 4: Install core scientific libraries using pip"
pip3 install numpy scipy pandas matplotlib scikit-learn 

echo "Step 5: Install IPython and JupyterLab using pip"
pip3 install ipython jupyterlab

echo "Step 6: Install world model project libraries (einops, transformers) using pip"
pip3 install einops transformers

echo "Step 7: Install additional pip-only libraries"
pip3 install decord diffusers tqdm timm torcheval lpips notebook dreamsim ipywidgets ffmpeg

echo "Step 8: Install additional conda libraries" ffmpegfor video processing
conda install -c conda-forge ffmpeg

echo "✅ Environment 'nwm-env' setup complete. You can now activate it using:"
echo "   mamba activate nwm-env"
eval "$(mamba shell hook --shell bash)"
mamba activate nwm-env
