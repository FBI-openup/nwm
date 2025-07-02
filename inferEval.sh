# please use the inferEval.py script to run the evaluation which is more advanced
# This script contain no saving eval result as a file  and some of the output path is not defined



#!/bin/bash

################################################################################
#                     CONFIGURATION SECTION — MODIFY HERE ONLY                #
################################################################################

# Dataset to evaluate. Use comma-separated values if evaluating multiple datasets.
DATASETS="scand"  # Possible values: scand, recon, huron, tartan_drive, navwset (your own dataset)

# Checkpoint & experiment
CKPT="0100000"  # Checkpoint ID to evaluate. Must exist in checkpoints dir.
EXP_CONFIG="config/nomad_eval.yaml"  # Path to your experiment config file (.yaml)

# Output location for results (both GT and prediction)
OUTPUT_DIR="/path/to/output_dir"  # Change to desired storage path

# Input FPS of dataset (controls time scale granularity)
INPUT_FPS=4  # Normally 4, matching the input video frame rate

# Rollout FPS values to evaluate (must be <= INPUT_FPS)
ROLLOUT_FPS_VALUES="1,4"  # e.g., 1 frame/sec and 4 frame/sec

# Number of seconds to evaluate into the future (i.e., 1s, 2s, 4s, etc.)
NUM_SEC_EVAL=5  # Will evaluate at seconds = [1, 2, 4, 8, 16]

# Evaluation batch size
BATCH_SIZE=12  # Modify depending on GPU capacity

# Num of CPU workers for dataloader
NUM_WORKERS=8  # Increase if CPU load permits

################################################################################
#                                Derived Variables                             #
################################################################################

EXP_NAME=$(basename $EXP_CONFIG .yaml)_${CKPT}  # e.g., nomad_eval_0100000

################################################################################
#                                Pipeline Start                                #
################################################################################

echo "========================================="
echo "Step 1: Generate Ground Truth (GT) Images"
echo "========================================="

python isolated_nwm_infer.py \
  --output_dir ${OUTPUT_DIR} \
  --exp dummy_exp_gt \
  --ckp ${CKPT} \
  --eval_type time \
  --datasets ${DATASETS} \  # Example values: scand, recon, huron, tartan_drive, navwset
  --input_fps ${INPUT_FPS} \
  --gt 1 \
  --num_sec_eval ${NUM_SEC_EVAL} \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS}

python isolated_nwm_infer.py \
  --output_dir ${OUTPUT_DIR} \
  --exp dummy_exp_gt \
  --ckp ${CKPT} \
  --eval_type rollout \
  --datasets ${DATASETS} \
  --input_fps ${INPUT_FPS} \
  --rollout_fps_values ${ROLLOUT_FPS_VALUES} \
  --gt 1 \
  --num_sec_eval ${NUM_SEC_EVAL} \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS}

echo "==========================================="
echo "Step 2: Generate Predicted Images via Model"
echo "==========================================="

python isolated_nwm_infer.py \
  --output_dir ${OUTPUT_DIR} \
  --exp ${EXP_CONFIG} \
  --ckp ${CKPT} \
  --eval_type time \
  --datasets ${DATASETS} \
  --input_fps ${INPUT_FPS} \
  --gt 0 \
  --num_sec_eval ${NUM_SEC_EVAL} \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS}

python isolated_nwm_infer.py \
  --output_dir ${OUTPUT_DIR} \
  --exp ${EXP_CONFIG} \
  --ckp ${CKPT} \
  --eval_type rollout \
  --datasets ${DATASETS} \
  --input_fps ${INPUT_FPS} \
  --rollout_fps_values ${ROLLOUT_FPS_VALUES} \
  --gt 0 \
  --num_sec_eval ${NUM_SEC_EVAL} \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS}

echo "==============================="
echo "Step 3: Compute Evaluation Score"
echo "==============================="

python isolated_nwm_eval.py \
  --datasets ${DATASETS} \
  --gt_dir ${OUTPUT_DIR}/dummy_exp_gt \
  --exp_dir ${OUTPUT_DIR}/${EXP_NAME} \
  --eval_types time,rollout \  # Types of evaluation: "time" for fixed sec, "rollout" for sequential prediction
  --rollout_fps_values ${ROLLOUT_FPS_VALUES} \
  --batch_size ${BATCH_SIZE} \
  --num_sec_eval ${NUM_SEC_EVAL}

echo "✅ Evaluation completed. Results saved under:"
echo "   ${OUTPUT_DIR}/${EXP_NAME}/*.json"