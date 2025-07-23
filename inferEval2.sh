#!/bin/bash
#SBATCH --job-name=nwm_training_job
#SBATCH --partition=ENSTA-l40s
#SBATCH --gpus=1
#SBATCH --time=5:00:00
#SBATCH --nodelist=ensta-l40s01.r2.enst.fr
#SBATCH --output=l40s_job_%j.out
#SBATCH --error=l40s_job_%j.err

echo "✅ Job started on node: $(hostname) / 任务运行节点：$(hostname)"
echo "🕒 Start time: $(date) / 开始时间：$(date)"

#load conda environment
source ~/miniconda3/etc/profile.d/conda.sh || { echo "❌ Failed to load conda.sh / 无法加载 conda.sh"; exit 1; }
conda activate nwm-env || { echo "❌ Failed to activate conda env / 激活 Conda 环境失败"; exit 1; }

# switch to the directory containing the code
cd ${HOME}/boyuan/nwm || { echo "❌ Directory not found: ${HOME}/boyuan/nwm / 找不到代码目录"; exit 1; }

echo "🚀 Starting model inference and evaluation... / 开始模型推理和评估..."

# replace with your inference and evaluation command
python inferEval.py --exp_config config/nwm_cdit_l_latents_L40S.yaml

#example command for inference and evaluation with specific parameters
# python infer.py --config config/nwm_cdit_xl.yaml --device cuda --batch_size 8 --num_workers 4

echo "✅ Inference and evaluation finished. / 推理和评估完成。"
echo "🕓 End time: $(date) / 结束时间：$(date)"