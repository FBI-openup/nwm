## Navigation World Models, CVPR 2025 (Oral) <br><sub>Official PyTorch Implementation</sub>

### [Paper](https://arxiv.org/abs/2412.03572) | [Project Page](https://www.amirbar.net/nwm/) | [Notebook Demo](interactive_model.ipynb) | [Models](https://huggingface.co/facebook/nwm)

This repository contains the official PyTorch implementation of **Navigation World Models** - the Conditional Diffusion Transformer (CDiT) model training code, along with an integrated **WorldMem** system for long-term consistent world simulation with memory. See the [project page](https://www.amirbar.net/nwm) for additional results.

## Key Features

🚀 **Navigation World Models (CDiT)**
- Conditional Diffusion Transformer for navigation tasks
- Support for multiple datasets (RECON, SCAND, SACSon, Tartan Drive)
- Distributed training with PyTorch and torchrun
- Comprehensive evaluation on single-step and trajectory prediction

🧠 **WorldMem Integration**
- Long-term consistent world simulation with memory
- Hybrid CDiT architecture with intelligent memory management
- Interactive Gradio interface for real-time exploration
- Advanced memory retrieval and storage mechanisms

⚡ **Performance Optimizations**
- Latent encoding workflow for faster training
- Mixed precision training support
- Torch compile optimization
- Automated setup and testing scripts

> [**Navigation World Models**](https://www.amirbar.net/nwm)<br>
> [Amir Bar](https://www.amirbar.net), [Gaoyue "Kathy" Zhou](https://gaoyuezhou.github.io/), [Danny Tran](https://dannytran123.github.io/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Yann LeCun](http://yann.lecun.com/)
> <br>AI at Meta, UC Berkeley, New York University<br>

## Setup & Installation

### Quick Start (Recommended)

1. **Clone the repository:**
```bash
git clone https://github.com/FBI-openup/nwm
cd nwm
```

2. **Setup base NWM environment:**
```bash
chmod +x scripts/setup_nwm_env.sh
./scripts/setup_nwm_env.sh
```

3. **Activate environment and setup WorldMem:**
```bash
mamba activate nwm-env
python scripts/worldmem_setup_and_test.py
```

4. **Download and setup data & models:**
```bash
# Create necessary directories
mkdir -p logs/nwm_cdit_xl/checkpoints
mkdir -p data

# Download pretrained models (choose one of the following):
# Option 1: Download from Hugging Face
# Visit https://huggingface.co/facebook/nwm and download the model checkpoint
# Then place it in: ./logs/nwm_cdit_xl/checkpoints/

# Option 2: If you have a specific checkpoint file, place it directly:
# cp /path/to/your/checkpoint.pth.tar ./logs/nwm_cdit_xl/checkpoints/

# Setup datasets (follow the data preparation steps below)
# Processed datasets should be placed in ./data/<dataset_name>/
```

5. **Ready to train or run inference!**

### Manual Installation

For manual setup or troubleshooting, follow these steps:

#### 1. Create conda environment:
```bash
mamba create -n nwm-env python=3.10
mamba activate nwm-env
```

#### 2. Install PyTorch with CUDA support:
```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
```

#### 3. Install dependencies:
```bash
# Core scientific libraries
pip3 install numpy scipy pandas matplotlib scikit-learn ipython jupyterlab

# World model libraries
pip3 install einops transformers decord diffusers tqdm timm torcheval lpips notebook dreamsim ipywidgets

# Video processing
conda install -c conda-forge ffmpeg

# WorldMem specific dependencies (if using WorldMem)
cd WorldMem
pip3 install -r requirements.txt
```

### Automated Setup Scripts

The repository includes two automated setup scripts:

- **`scripts/setup_nwm_env.sh`**: Sets up the base NWM environment with PyTorch and core dependencies
- **`scripts/worldmem_setup_and_test.py`**: Comprehensive WorldMem installation and testing suite

For detailed information about the automated setup, see [WorldMem Setup Guide](docs/WorldMem_Setup_README.md).

## Project Structure

```
nwm/
├── 📁 Core NWM Files
│   ├── train.py                    # Main training script
│   ├── models.py                   # CDiT model implementations
│   ├── hybrid_models.py            # Hybrid CDiT with memory system
│   ├── datasets.py                 # Dataset loading and preprocessing
│   └── latent_dataset.py           # Latent dataset handling
│
├── 📁 Scripts & Tools
│   ├── scripts/
│   │   ├── setup_nwm_env.sh        # Base environment setup
│   │   ├── worldmem_setup_and_test.py # WorldMem setup & testing
│   │   ├── train_L40S_slurm.sh     # SLURM training script
│   │   ├── isolated_nwm_infer.py   # Model inference
│   │   ├── isolated_nwm_eval.py    # Model evaluation
│   │   ├── planning_eval.py        # Planning evaluation
│   │   ├── inferEval.py            # Inference evaluation
│   │   ├── encode_latents.py       # Latent encoding
│   │   └── encode_all_datasets.sh  # Batch latent encoding
│
├── 📁 Configuration & Data
│   ├── config/                     # Model and training configurations
│   ├── data/                       # Training datasets
│   └── data_splits/                # Dataset split configurations
│
├── 📁 WorldMem Integration
│   └── WorldMem/                   # Complete WorldMem project
│       ├── app.py                  # Gradio interface
│       ├── main.py                 # WorldMem training script
│       ├── algorithms/             # Core WorldMem algorithms
│       ├── experiments/            # Experiment configurations
│       ├── datasets/               # WorldMem-specific datasets
│       └── utils/                  # Utility functions
│
└── 📁 Documentation
    ├── README.md                   # Main project documentation (this file)
    └── docs/
        ├── WorldMem_Setup_README.md    # Comprehensive setup guide
        ├── README_Hybrid_Memory.md     # Memory system architecture
        └── README_Latent_Encoding.md   # Latent encoding workflow
```

## Data

### Data Preparation

To download and preprocess data, please follow the steps from [NoMaD](https://github.com/robodhruv/visualnav-transformer?tab=readme-ov-file#data-wrangling), specifically:
- Download the datasets
- Change the [preprocessing resolution](https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/data/data_utils.py#L13) from (160, 120) to (320, 240) for higher resolution 
- run `process_bags.py` and `process_recon.py` to save each processed dataset to `path/to/nwm_repo/data/<dataset_name>`.

For [SACSon/HuRoN](https://sites.google.com/view/sacson-review/huron-dataset), we used a private version which contains higher resolution images. Please contact the authors for access.

### Required Directory Structure

After data preparation and model download, you should have the following structure:

```
nwm/
├── data/                           # Processed datasets directory
│   ├── recon/                      # RECON dataset
│   ├── scand/                      # SCAND dataset  
│   ├── sacson/                     # SACSon dataset
│   ├── tartan_drive/               # Tartan Drive dataset
│   └── navware/                    # (Optional) NavWare dataset
│
├── logs/                           # Model checkpoints directory
│   └── nwm_cdit_xl/               # CDiT-XL model directory
│       └── checkpoints/            # Place pretrained models here
│           ├── 0100000.pth.tar     # Example checkpoint
│           └── latest.pth.tar      # Latest checkpoint
│
└── latents/                        # (Optional) Pre-encoded latents for faster training
    ├── recon/                      # Latent encodings for RECON
    ├── scand/                      # Latent encodings for SCAND
    └── ...
```

### Dataset Structure Details

Each dataset should follow this structure:

### Dataset Structure Details

Each dataset should follow this structure:

```
data/<dataset_name>/
├── <name_of_traj1>/
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── ...
│   ├── T_1.jpg
│   └── traj_data.pkl
├── <name_of_traj2>/
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── ...
│   ├── T_2.jpg
│   └── traj_data.pkl
├── ...
└── <name_of_trajN>/
    ├── 0.jpg
    ├── 1.jpg
    ├── ...
    ├── T_N.jpg
    └── traj_data.pkl
```

### Important Notes
- **Processed datasets** must be placed in `./data/` directory
- **Pretrained models** must be placed in `./logs/nwm_cdit_xl/checkpoints/` directory
- For faster training, consider using [latent encoding](README_Latent_Encoding.md) to pre-encode images  

## Training

### NWM CDiT Training

#### Distributed Training (Multi-node)

Using torchrun:
```bash
export NUM_NODES=8
export HOST_NODE_ADDR=<HOST_ADDR>
export CURR_NODE_RANK=<NODE_RANK>

torchrun \
  --nnodes=${NUM_NODES} \
  --nproc-per-node=8 \
  --node-rank=${CURR_NODE_RANK} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${HOST_NODE_ADDR}:29500 \
  train.py --config config/nwm_cdit_xl.yaml --ckpt-every 2000 --eval-every 10000 --bfloat16 1 --epochs 300 --torch-compile 0
```

Or using submitit and slurm (8 machines of 8 gpus):
```bash
python submitit_train_cw.py --nodes 8 --partition <partition_name> --qos <qos> --config config/nwm_cdit_xl.yaml --ckpt-every 2000 --eval-every 10000 --bfloat16 1 --epochs 300  --torch-compile 0
```

#### Single GPU Training

For local development or single GPU training:
```bash
python train.py --config config/nwm_cdit_xl.yaml --ckpt-every 2000 --eval-every 10000 --bfloat16 1 --epochs 300  --torch-compile 0
```

#### Cluster Training with SLURM

For training on SLURM clusters (like L40S), use the provided script:
```bash
# Modify the script for your cluster configuration
sbatch scripts/train_L40S_slurm.sh
```

See [scripts/train_L40S_slurm.sh](scripts/train_L40S_slurm.sh) for SLURM job configuration and training parameters.

### WorldMem Training

For WorldMem-specific training with memory components:

```bash
cd WorldMem
python main.py +name=worldmem_experiment

# Or use training scripts for different stages
bash train_stage_1.sh  # Initial training
bash train_stage_2.sh  # Memory integration
bash train_stage_3.sh  # Full system training
```

### Performance Notes
- Torch compile can lead to ~40% faster training speed but might cause instabilities
- Use latent encoding (see [Latent Encoding Guide](docs/README_Latent_Encoding.md)) for significant speedup
- For memory-enabled training, see [Hybrid Memory System](docs/README_Hybrid_Memory.md)

## Pretrained Models
To use a pretrained CDiT/XL model:
- Download a pretrained model from [Hugging Face](https://huggingface.co/facebook/nwm)
- Place the checkpoint in ./logs/nwm_cdit_xl/checkpoints

# Evaluation

directory to save evaluation results:
`export RESULTS_FOLDER=/path/to/res_folder/`

## Evaluate on single time step prediction 

### 1. Prepare ground truth frames for evaluation (one-time)

```bash
python scripts/isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --datasets recon,scand,sacson,tartan_drive \
    --batch_size 96 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1
```
### 2. Predict future state given action

```bash    
python scripts/isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --ckp 0100000 \
    --datasets <dataset_name> \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER}
```
### 3. Report metrics compared to GT (LPIPS, DreamSim, FID)

```bash    
python scripts/isolated_nwm_eval.py \
    --datasets <dataset_name> \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/nwm_cdit_xl \
    --eval_types time
```
Results are saved in ${RESULTS_FOLDER}/nwm_cdit_xl/<dataset_name>

## Evaluate on following ground truth trajectories

### 1. Prepare ground truth frames for evaluation (one-time)

```bash
python scripts/isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --datasets recon,scand,sacson,tartan_drive \
    --batch_size 96 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1 \
    --rollout_fps_values 1,4
```
### 2. Simulate a GT trajectory using NWM
```bash
python scripts/isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --ckp 0100000 \
    --datasets <dataset_name> \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER} \
    --rollout_fps_values 1,4
```

### 3. Report metrics compared to GT trajectories (LPIPS, DreamSim, FID)
```bash
    python scripts/isolated_nwm_eval.py \
        --datasets recon \
        --gt_dir ${RESULTS_FOLDER}/gt \
        --exp_dir ${RESULTS_FOLDER}/nwm_cdit_xl \
        --eval_types rollout
```
Results are saved in ${RESULTS_FOLDER}/nwm_cdit_xl/<dataset_name>

### Trajectory Evaluation - Planning

Using 1-step Cross Entropy Method planning on 8 gpus (sampling 120 trajectories):
```bash
torchrun --nproc-per-node=8 scripts/planning_eval.py \
    --exp config/nwm_cdit_xl.yaml   \
    --datasets recon   \
    --rollout_stride 1   \
    --batch_size 1   \
    --num_samples 120   \
    --topk 5   \
    --num_workers 12   \
    --output_dir ${RESULTS_FOLDER}   \
    --save_preds   \
    --ckp 0100000   \
    --opt_steps 1   \
    --num_repeat_eval 3
```
Results are saved in ${RESULTS_FOLDER}/nwm_cdit_xl/<dataset_name>

## BibTeX

```bibtex
@article{bar2024navigation,
  title={Navigation world models},
  author={Bar, Amir and Zhou, Gaoyue and Tran, Danny and Darrell, Trevor and LeCun, Yann},
  journal={arXiv preprint arXiv:2412.03572},
  year={2024}
}
```

## Documentation Guide

This repository contains several specialized README files for different components:

### Core Components
- **[WorldMem Setup Guide](docs/WorldMem_Setup_README.md)**: Comprehensive installation and testing guide for WorldMem components
- **[Hybrid Memory System](docs/README_Hybrid_Memory.md)**: Design documentation for the Hybrid CDiT Memory System, including dual-standard memory architecture and adaptive optimization
- **[Latent Encoding Workflow](docs/README_Latent_Encoding.md)**: Guide for preprocessing images into latent representations using Stable Diffusion VAE for faster training

### WorldMem Subdirectories
- **[WorldMem/README.md](WorldMem/README.md)**: Main WorldMem project documentation with installation and quick start
- **[WorldMem/algorithms/README.md](WorldMem/algorithms/README.md)**: Core algorithm implementations and usage
- **[WorldMem/experiments/README.md](WorldMem/experiments/README.md)**: Experiment configurations and training setups
- **[WorldMem/datasets/README.md](WorldMem/datasets/README.md)**: Dataset handling and preprocessing utilities
- **[WorldMem/utils/README.md](WorldMem/utils/README.md)**: Utility functions and helper modules

### Quick Navigation
- **Setup & Installation**: See [WorldMem Setup Guide](docs/WorldMem_Setup_README.md) for automated installation
- **Memory System Design**: Read [Hybrid Memory System](docs/README_Hybrid_Memory.md) for architecture details
- **Performance Optimization**: Check [Latent Encoding Workflow](docs/README_Latent_Encoding.md) for training speedup
- **WorldMem Usage**: Visit [WorldMem/README.md](WorldMem/README.md) for the main WorldMem documentation

### Getting Started Checklist
1. ✅ Follow [Setup & Installation](#setup--installation) above
2. ✅ Read [WorldMem Setup Guide](docs/WorldMem_Setup_README.md) for detailed setup
3. ✅ (Optional) Use [Latent Encoding](docs/README_Latent_Encoding.md) for faster training
4. ✅ (Advanced) Explore [Hybrid Memory System](docs/README_Hybrid_Memory.md) for custom implementations

## Acknowledgments
We thank Noriaki Hirose for his help with the HuRoN dataset and for sharing his insights, and to Manan Tomar, David Fan, Sonia Joseph, Angjoo Kanazawa, Ethan Weber, Nicolas Ballas, and the anonymous reviewers for their helpful discussions and feedback.

## License
The code and model weights are licensed under Creative Commons Attribution-NonCommercial 4.0 International. See [`LICENSE.txt`](LICENSE.txt) for details.