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

## Project Structure & File Guide

```
nwm/
├── Core Training & Model Files
│   ├── train.py                    # Main training script for NWM CDiT models
│   ├── models.py                   # CDiT model implementations (Transformer architectures)
│   ├── hybrid_models.py            # Hybrid CDiT with integrated memory system
│   ├── datasets.py                 # Legacy dataset handling (used by evaluation scripts)
│   ├── latent_dataset.py           # Current dataset pipeline - latent-based training
│   ├── distributed.py              # Distributed training utilities and synchronization
│   ├── misc.py                     # Utility functions (transforms, metrics, helpers)
│   └── submitit_train_cw.py        # SLURM cluster training script (requires: pip install submitit)
│
├── Evaluation & Inference Scripts
│   ├── scripts/
│   │   ├── isolated_nwm_infer.py   # Model inference for single-step and rollout prediction
│   │   ├── isolated_nwm_eval.py    # Image quality evaluation (LPIPS, FID, DreamSim)
│   │   ├── inferEval.py            # Automated evaluation pipeline for image generation
│   │   ├── planning_eval.py        # NOMAD path planning evaluation (trajectory analysis)
│   │   ├── encode_latents.py       # VAE latent encoding for individual datasets
│   │   └── encode_all_datasets.sh  # Batch script for encoding multiple datasets
│
├── Setup & Environment Scripts  
│   ├── scripts/
│   │   ├── setup_nwm_env.sh        # Base environment setup (conda, PyTorch, dependencies)
│   │   ├── worldmem_setup_and_test.py # Comprehensive WorldMem installation & testing
│   │   └── train_L40S_slurm.sh     # SLURM job script for L40S cluster training
│
├── Configuration Files
│   ├── config/
│   │   ├── data_config.yaml        # Dataset paths and preprocessing settings
│   │   ├── eval_config.yaml        # Evaluation parameters and metrics configuration
│   │   ├── memory_config.yaml      # Memory system configuration for WorldMem
│   │   └── nwm_cdit_xl.yaml        # CDiT-XL model configuration (training hyperparams)
│
├── Data & Datasets
│   ├── data/                       # Training datasets directory
│   │   ├── recon/                  # RECON indoor navigation dataset
│   │   ├── scand/                  # SCAND outdoor navigation dataset  
│   │   ├── sacson/                 # SACSon/HuRoN dataset
│   │   └── tartan_drive/           # Tartan Drive dataset
│   │
│   ├── data_splits/                # Dataset split configurations
│   └── latents/                    # (Optional) Pre-encoded latents for faster training
│
├── Model Outputs & Results
│   ├── logs/nwm_cdit_xl/checkpoints/ # Model checkpoint files (.pth.tar)
│   ├── output_GT/                  # Ground truth evaluation images
│   ├── output_pred/                # Model prediction outputs
│   └── eval_table/                 # Evaluation results and metrics tables
│
├── WorldMem Integration
│   └── WorldMem/                   # Complete WorldMem project
│       ├── app.py                  # Interactive Gradio interface
│       ├── main.py                 # WorldMem training script
│       ├── requirements.txt        # WorldMem-specific dependencies
│       ├── algorithms/             # Core WorldMem algorithms
│       └── datasets/               # WorldMem-specific datasets
│
├── Advanced Features
│   ├── diffusion/                  # Diffusion model utilities and schedulers
│   ├── interactive_model.ipynb     # Jupyter notebook for interactive model exploration
│   └── requirements-eval.txt       # Optional evaluation dependencies (evo, dreamsim)
│
└── Documentation
    ├── README.md                   # Main project documentation (this file)
    └── docs/                       # Detailed documentation guides
```

### Data Processing Pipeline Evolution

**Current Pipeline (Recommended):**
- 🚀 **`latent_dataset.py`**: Main dataset handler for training
  - Uses pre-encoded VAE latents (`.pt` files) for faster training
  - Implemented in `LatentTrainingDataset` and `LatentEvalDataset`
  - Significant performance improvement over raw image processing

**Legacy Pipeline (Evaluation Only):**
- 📁 **`datasets.py`**: Legacy dataset handling
  - Still used by evaluation scripts: `isolated_nwm_infer.py`, `planning_eval.py`
  - Processes raw images on-the-fly (slower but needed for specific evaluations)
  - Classes: `EvalDataset`, `TrajectoryEvalDataset`

**Pipeline Workflow:**
1. **Data Encoding**: Use `scripts/encode_latents.py` to pre-process datasets
2. **Training**: `train.py` automatically uses `latent_dataset.py`
3. **Evaluation**: Some scripts still require `datasets.py` for compatibility

> **Note**: Both files are maintained for compatibility. The training pipeline has fully migrated to latent-based processing, while evaluation scripts may still require the legacy image processing pipeline.

### Key File Categories

**Core Training Files**
- `train.py`: Main entry point for training NWM models with distributed support
- `models.py`: Contains CDiT (Conditional Diffusion Transformer) model architectures
- `hybrid_models.py`: Advanced models with integrated memory systems for long-term consistency
- `datasets.py`: Data loading pipeline supporting multiple navigation datasets

**Inference & Evaluation**
- `isolated_nwm_infer.py`: Generate predictions for evaluation (both single-step and trajectory)
- `isolated_nwm_eval.py`: Compute image quality metrics (LPIPS, FID, DreamSim)
- `inferEval.py`: End-to-end evaluation automation for image generation quality
- `planning_eval.py`: Specialized evaluation for navigation planning using trajectory metrics

**Configuration Files**
- `data_config.yaml`: Dataset paths, preprocessing settings, and data pipeline config
- `eval_config.yaml`: Evaluation metrics, batch sizes, and testing parameters
- `memory_config.yaml`: Memory system settings for WorldMem integration

**Setup & Dependencies**
- `worldmem_setup_and_test.py`: Automated installation and testing for all components
- `setup_nwm_env.sh`: Base environment setup script
- `requirements-eval.txt`: Optional heavy dependencies for trajectory evaluation (evo library)
- `submitit_train_cw.py`: Optional SLURM cluster training (requires: `pip install submitit`)

### Usage Workflow

1. **Setup**: Use `setup_nwm_env.sh` and `worldmem_setup_and_test.py`
2. **Data Prep**: Use `encode_latents.py` for faster training (optional)
3. **Training Options**:
   - **Single GPU**: `python train.py` (recommended for development)
   - **Multi-GPU (single node)**: `torchrun` with `train.py` (most common)
   - **Multi-node cluster**: `submitit_train_cw.py` (requires `pip install submitit`)
4. **Evaluation**: Use `inferEval.py` for images or `planning_eval.py` for trajectories
5. **Interactive**: Explore with `interactive_model.ipynb` or WorldMem's `app.py`

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

**⚠️ Prerequisites**: This requires the `submitit` library for SLURM job management:
```bash
pip install submitit
```

```bash
python submitit_train_cw.py --nodes 8 --partition <partition_name> --qos <qos> --config config/nwm_cdit_xl.yaml --ckpt-every 2000 --eval-every 10000 --bfloat16 1 --epochs 300  --torch-compile 0
```

> **Note**: The `submitit` library is only needed for SLURM cluster training. For single GPU or local multi-GPU training, use the torchrun method above or single GPU training below.

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

The repository provides comprehensive evaluation tools for different aspects of the Navigation World Models:

## Evaluation Types Overview

### **Image Generation Quality** (`inferEval.py`)
- **Purpose**: Evaluate visual quality of generated images
- **Metrics**: LPIPS, DreamSim, FID
- **Use Case**: Assess how realistic and accurate the generated images are
- **Target**: General model performance evaluation

### **Navigation Planning Quality** (`planning_eval.py`)  
- **Purpose**: Evaluate trajectory and path planning accuracy
- **Metrics**: APE (Absolute Pose Error), RPE (Relative Pose Error)
- **Use Case**: Assess navigation and planning capabilities
- **Target**: NOMAD path planning evaluation
- **Dependencies**: Requires `evo` library (install via `requirements-eval.txt`)

### **Low-level Evaluation Tools**
- **`isolated_nwm_infer.py`**: Generate predictions for both evaluation types
- **`isolated_nwm_eval.py`**: Compute specific image quality metrics

## Quick Evaluation Setup

```bash
# Set up evaluation environment
export RESULTS_FOLDER=/path/to/results

# For image generation evaluation (lightweight)
python scripts/inferEval.py --exp_config config/nwm_cdit_xl.yaml --datasets recon

# For navigation planning evaluation (requires evo library)
pip install -r requirements-eval.txt  # Install evaluation dependencies
python scripts/planning_eval.py --exp config/nwm_cdit_xl.yaml --datasets recon
```

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

### Trajectory Evaluation - Navigation Planning

**⚠️ Important**: This evaluation requires specialized dependencies for trajectory analysis.

#### Prerequisites
```bash
# Install trajectory evaluation dependencies (heavyweight packages ~100MB+)
pip install -r requirements-eval.txt

# Or use the automated setup
python scripts/worldmem_setup_and_test.py
```

#### Planning Evaluation with Cross Entropy Method (CEM)
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

#### What This Evaluates
- **Trajectory Accuracy**: How well the model follows intended paths
- **Pose Estimation**: Accuracy of position and orientation predictions  
- **Planning Quality**: Effectiveness of the Cross Entropy Method for navigation
- **Navigation Metrics**: APE (Absolute Pose Error) and RPE (Relative Pose Error)

Results are saved in ${RESULTS_FOLDER}/nwm_cdit_xl/<dataset_name>

#### Optional Dependencies Note
The `planning_eval.py` script uses specialized libraries:
- **`evo`**: Trajectory evaluation with ROS support
- **`dreamsim`**: Advanced similarity metrics

These are not required for basic model training or image generation evaluation.

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
1. Follow [Setup & Installation](#setup--installation) above
2. Review [Project Structure & File Guide](#project-structure--file-guide) to understand the codebase
3. Read [WorldMem Setup Guide](docs/WorldMem_Setup_README.md) for detailed setup
4. (Optional) Use [Latent Encoding](docs/README_Latent_Encoding.md) for faster training
5. Choose appropriate evaluation: [Image Quality](#image-generation-quality-inferevalpy) or [Navigation Planning](#navigation-planning-quality-planning_evalpy)
6. (Advanced) Explore [Hybrid Memory System](docs/README_Hybrid_Memory.md) for custom implementations

## Acknowledgments
We thank Noriaki Hirose for his help with the HuRoN dataset and for sharing his insights, and to Manan Tomar, David Fan, Sonia Joseph, Angjoo Kanazawa, Ethan Weber, Nicolas Ballas, and the anonymous reviewers for their helpful discussions and feedback.

## License
The code and model weights are licensed under Creative Commons Attribution-NonCommercial 4.0 International. See [`LICENSE.txt`](LICENSE.txt) for details.