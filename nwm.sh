#!/bin/bash
#SBATCH --job-name=my_l40s_training
#SBATCH --partition=ENSTA-l40s
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodelist=ensta-l40s01.r2.enst.fr
#SBATCH --output=l40s_job_%j.out
#SBATCH --error=l40s_job_%j.err

echo "任务开始，运行在节点: $(hostname)"
echo "当前时间: $(date)"

# 正确加载 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh || { echo "❌ 无法加载 conda.sh"; exit 1; }
conda activate nwm-env || { echo "❌ conda activate 失败"; exit 1; }


# 进入代码目录（使用绝对路径）
cd ${HOME}/boyuan/nwm || { echo "❌ 路径不存在"; exit 1; }


# 测试命令
echo "即将运行 Python hello world"
python -c "print('Hello from Python inside SLURM')"

echo "任务结束"
