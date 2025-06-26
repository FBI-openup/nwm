#!/bin/bash

# --- SLURM 资源申请指令 ---
#SBATCH --job-name=test_hello_nwm
#SBATCH --partition=ENSTA-l40s
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --nodelist=ensta-l40s01.r2.enst.fr
#SBATCH --output=hello_job_%j.out
#SBATCH --error=hello_job_%j.err

echo "任务开始，运行在节点: $(hostname)"
echo "SLURM 任务ID: $SLURM_JOB_ID"

# 获取当前环境名（适用于 conda >= 4.6）
CURRENT_ENV=$(conda info --envs | awk '{if ($1 == "*") print $2}')

# 只在不处于 nwm-env 时激活它
if [[ "$CURRENT_ENV" != "nwm-env" ]]; then
    echo "当前环境为 '$CURRENT_ENV'，正在激活 nwm-env"
    source /home/ensta-boyuan.zhang/miniconda3/bin/activate nwm-env
else
    echo "已经在 nwm-env 中，无需重新激活"
fi

# 跳转到你的工作目录（绝对路径更保险）
cd /home/ensta-boyuan.zhang/boyuan/nwm

# 简单 hello world 测试
echo "Hello from SLURM job!"
python -c "print('Hello from Python in SLURM job.')"

# 可选：真正运行训练命令
# python train.py --data ./log --epochs 100

echo "任务结束"