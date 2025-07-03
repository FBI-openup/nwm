#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Boyuan Zhang

This script automates the full evaluation pipeline for NWM models:
    1. Generate Ground Truth images
    2. Generate predicted images using pretrained models
    3. Run LPIPS / DreamSim / FID evaluation and output scores

The script also creates/maintains a central table (CSV) to store results, for easy comparison across experiments.

Usage Notes:
- You need to configure dataset, config path, checkpoint, and optional fps settings.
- All generated files are placed under clearly named subfolders of: output_GT, output_pred, and eval_table.
"""

import os
import subprocess
import argparse
import csv
from datetime import datetime

# ======================= Argument Parsing ========================
parser = argparse.ArgumentParser(description="Run full NWM evaluation pipeline")

parser.add_argument('--datasets', type=str, default="scand", help="Comma-separated dataset list (e.g., 'scand,recon,huron,tartan')")
parser.add_argument('--exp_config', type=str, required=True, help="Path to model inference config (e.g., config/nwm_cdit_xl.yaml)")
parser.add_argument('--ckpt', type=str, default='0100000', help="Checkpoint ID (e.g., '0100000')")
parser.add_argument('--input_fps', type=int, default=4, help="Input video FPS (default=4)")
parser.add_argument('--rollout_fps', type=str, default='1,4', help="Rollout FPS values (comma-separated)")
parser.add_argument('--num_sec_eval', type=int, default=5, help="How many seconds to evaluate (powers of 2)")
parser.add_argument('--batch_size', type=int, default=12, help="Batch size for eval and infer")
parser.add_argument('--num_workers', type=int, default=8, help="Num workers for DataLoader")

args = parser.parse_args()

# ======================= Derived Paths ========================
root_dir = os.getcwd()  # assumes script is in NWM root dir

gt_dir = os.path.join(root_dir, "output_GT")
pred_dir = os.path.join(root_dir, "output_pred")
eval_dir = os.path.join(root_dir, "eval_table")

os.makedirs(gt_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

exp_name = os.path.splitext(os.path.basename(args.exp_config))[0]
pred_exp_dir = os.path.join(pred_dir, exp_name)
if args.ckpt != '0100000':
    pred_exp_dir += f"_{args.ckpt}"


# For eval script
#gt_exp_dir = gt_dir + "/dummy_exp_gt"
gt_exp_dir = os.path.join(gt_dir, "gt")
pred_exp_dir = os.path.join(pred_dir, exp_name)

def run(cmd_list):
    print(f"\n>>> Running: {' '.join(cmd_list)}\n")
    subprocess.run(cmd_list, check=True)

# ======================= Step 1: GT generation ========================
for eval_type in ["time", "rollout"]:
    cmd = [
        "python", "isolated_nwm_infer.py",
        "--output_dir", gt_dir,
        "--exp", "dummy_exp_gt",
        "--ckp", args.ckpt,
        "--datasets", args.datasets,
        "--eval_type", eval_type,
        "--input_fps", str(args.input_fps),
        "--gt", "1",
        "--num_sec_eval", str(args.num_sec_eval),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
    ]
    if eval_type == "rollout":
        cmd += ["--rollout_fps_values", args.rollout_fps]
    run(cmd)

# ======================= Step 2: Prediction generation ========================
for eval_type in ["time", "rollout"]:
    cmd = [
        "python", "isolated_nwm_infer.py",
        "--output_dir", pred_dir,
        "--exp", args.exp_config,
        "--ckp", args.ckpt,
        "--datasets", args.datasets,
        "--eval_type", eval_type,
        "--input_fps", str(args.input_fps),
        "--gt", "0",
        "--num_sec_eval", str(args.num_sec_eval),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
    ]
    if eval_type == "rollout":
        cmd += ["--rollout_fps_values", args.rollout_fps]
    run(cmd)

# ======================= Step 3: Run Evaluation ========================
eval_cmd = [
    "python", "isolated_nwm_eval.py",
    "--datasets", args.datasets,
    "--gt_dir", gt_exp_dir,
    "--exp_dir", pred_exp_dir,
    "--eval_types", "time,rollout",
    "--rollout_fps_values", args.rollout_fps,
    "--batch_size", str(args.batch_size),
    "--num_sec_eval", str(args.num_sec_eval),
]
run(eval_cmd)

# ======================= Step 4: Save to Eval Table ========================
print("\n>>> Saving evaluation summary to table")
from glob import glob
import json

json_files = sorted(glob(os.path.join(pred_exp_dir, "*.json")))
eval_table_path = os.path.join(eval_dir, "nwm_eval_results.csv")

is_new = not os.path.exists(eval_table_path)
with open(eval_table_path, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    if is_new:
        writer.writerow(["timestamp", "exp_name", "dataset", "eval_type", "metric", "sec", "score"])

    for jf in json_files:
        with open(jf, 'r') as f:
            stats = json.load(f)
        for key, score in stats.items():
            parts = key.split("_")  # e.g., scand_time_lpips_1s
            if len(parts) < 4:
                continue
            dataset, etype, metric, sec = parts[0], parts[1], parts[2], parts[3]
            writer.writerow([datetime.now().isoformat(), exp_name, dataset, etype, metric, sec, score])

print(f"\n✅ All done. Scores written to: {eval_table_path}")
