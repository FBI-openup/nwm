# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

"""
SLURM Multi-node Training Script
===============================

This script enables distributed training across multiple nodes using SLURM job scheduler.
It is OPTIONAL and only needed for large-scale cluster training.

FUNCTIONALITY:
- Submits training jobs to SLURM clusters
- Manages multi-node distributed training coordination
- Handles job scheduling, resource allocation, and fault tolerance
- Supports automatic job requeuing and checkpoint resumption

OPTIONAL DEPENDENCY:
This script requires the submitit library which is NOT installed by default.

To install the required dependency:
pip install submitit

WHEN TO USE:
- You have access to a SLURM cluster (HPC/research clusters)
- You want to train on multiple nodes (8+ GPUs across multiple machines)
- You need automated job management and resource scheduling

WHEN NOT NEEDED:
- Single GPU training: Use train.py directly
- Single node multi-GPU: Use torchrun with train.py
- Local development: Use train.py with appropriate config

USAGE EXAMPLES:
python submitit_train_cw.py --nodes 8 --ngpus 8 --config config/nwm_cdit_xl.yaml
python submitit_train_cw.py --nodes 4 --ngpus 8 --partition gpu --qos high

NOTE: This script is designed for Meta's internal cluster configuration.
Modify SLURM parameters (partition, account, etc.) for your specific cluster.
"""

import argparse
import os
import uuid
from pathlib import Path

import train as trainer
import submitit


def parse_args():
    trainer_parser = trainer.get_args_parser()
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument("--partition", default="learn", type=str, help="Partition where to submit")
    parser.add_argument("--qos", default="low", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Request 32G V100 GPUs")
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(f"/checkpoint/amaia/video/{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import train as trainer
        self._setup_gpu_args()
        trainer.main(self.args)

    def checkpoint(self):
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.log_dir = self.args.output_dir
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'

    executor.update_parameters(
        mem_gb=55 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,
        cpus_per_task=12,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        qos=args.qos,
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        slurm_account='fair_amaia_cw_video',
        **kwargs
    )

    executor.update_parameters(name="nwm")

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir
    args.log_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)
    print(job.job_id)


if __name__ == "__main__":
    main()