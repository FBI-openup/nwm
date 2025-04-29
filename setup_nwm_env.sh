#!/bin/bash

################################################################################
# 用法说明:
# 此脚本用于创建并配置适用于 NWM 项目的 Conda 环境（Python 3.10）。
# - 请确保系统已安装 mamba。
# - 脚本会自动创建名为 "nwm-env" 的环境，并安装必要依赖。
#
# 使用方式：
# 1. 保存本文件为 setup_nwm_env.sh
# 2. 给予执行权限：chmod +x setup_nwm_env.sh
# 3. 运行脚本：./setup_nwm_env.sh
#
# ⚠️ 注意：此脚本使用 PyTorch nightly 版本，CUDA 12.6。
################################################################################

# 出错立即终止
set -e

echo "Step 1: Create new conda environment (nwm-env, Python 3.10)"
mamba create -y -n nwm-env python=3.10

echo "Step 2: Activate the environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
mamba activate nwm-env

echo "Step 3: Install PyTorch, torchvision, torchaudio with CUDA 12.6 using pip"
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

echo "Step 4: Install core scientific libraries using pip"
pip3 install numpy scipy pandas matplotlib scikit-learn

echo "Step 5: Install IPython and JupyterLab using pip"
pip3 install ipython jupyterlab

echo "Step 6: Install world model project libraries (einops, transformers) using pip"
pip3 install einops transformers

echo "Step 7: Install additional pip-only libraries"
pip3 install decord diffusers tqdm timm torcheval lpips notebook dreamsim

echo "✅ Environment 'nwm-env' setup complete. You can now activate it using:"
echo "   mamba activate nwm-env"
