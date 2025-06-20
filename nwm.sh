#!/bin/bash

# --- SLURM 资源申请指令 ---
#SBATCH --job-name=my_l40s_training      # 任务名，方便在 squeue 中辨识
#SBATCH --partition=ENSTA-l40s           # 必须！指定使用 L40s 分区
#SBATCH --gpus=1                         # 申请1个GPU（每个L40s节点有4个）
#SBATCH --time=04:00:00                  # 运行时长上限 (根据幻灯片，学生最多4小时)
#SBATCH --nodelist=ensta-l40s01.r2.enst.fr # 关键！直接指定使用当前可用的节点
#SBATCH --output=l40s_job_%j.out         # 标准输出日志，%j 会被替换为JOBID
#SBATCH --error=l40s_job_%j.err          # 错误输出日志

# --- 实际要执行的命令 ---
echo "任务开始，运行在节点: $(hostname)"
echo "SLURM 任务ID: $SLURM_JOB_ID"

# 激活您的 Python 环境 (如果使用了 conda 或 venv)
# 例如: source /path/to/your/conda/bin/activate your_env_name
# 例如: source ~/myenv/bin/activate
# 在 SLURM 脚本中
source /home/ensta-boyuan.zhang/miniconda3/bin/activate nwm-env

# 切换到您存放代码的目录 (通常是提交任务的目录)
cd $boyuan/nwm

# 运行您的 Python 脚本
echo "start training Python script...of NWM"
#TODO: 替换为实际的训练命令
python train.py --data ./log --epochs 100

echo "任务结束"