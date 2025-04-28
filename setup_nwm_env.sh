#!/bin/bash

# 脚本一出错就停止
set -e

echo "Step 1: Create a new conda environment: nwm-env (Python 3.10)"
#mamba create -y -n nwm-Venv python=3.10

echo "Step 2: Activate the environment"
# 注意：脚本里激活 conda 需要重新 source 一下
source $(conda info --base)/etc/profile.d/conda.sh
#conda activate nwm-Venv

echo "Step 3: Install deep learning libraries (torch, torchvision, torchaudio with CUDA 11.8)"
#mamba install -y numpy scipy pandas matplotlib scikit-learn

echo "Step 5: Install IPython and JupyterLab"
mamba install -y ipython jupyterlab

echo "Step 6: Install world model project libraries (einops, decord, transformers)"
mamba install -c conda-forge einops decord transformers

echo "Step 7: Install additional pip-only libraries (dreamsim)"
pip install dreamsim

echo "✅ All done! You can now work in the 'nwm-env' environment."
