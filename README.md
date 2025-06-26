## Navigation World Models, CVPR 2025 (Oral) <br><sub>Official PyTorch Implementation</sub>

### [Paper](https://arxiv.org/abs/2412.03572) | [Project Page](https://www.amirbar.net/nwm/) | [Notebook Demo](interactive_model.ipynb) | [Models](https://huggingface.co/facebook/nwm)

This repo is based on the initial PyTorch implementation of [NWM](https://github.com/facebookresearch/nwm/tree/main) with which I and [Boyuan Zhang](https://github.com/FBI-openup) worked on to fit to our usage for the NavWare project.

We have mainly adapted the code to fit the model to a lower scale ressource (TELECOM Paris Cluster/ 1 4070ti for debugging).

We will suppose that you already have a set-up machine to work with pytorch and all other libraries involved in this project.

If that's not the case, check the [Requirements] section to setup the virtual environment used in this repo.

## Main changes to the initial code:
- We separated the VAE processing from the training process so that the models only takes latents as input instead of images and outputs latents.
- We added directly added the pre-processing module in order to not have to go back and forth between different repos.
- Other changes to come...


## Setup
First, download and setup the repo:

```bash
git clone https://github.com/ssymon8/nwm
cd nwm
```
## Requirements
You can initialize the venv we used by simply inputting 
```bash
bash setup_nwm_env.sh
mamba activate nwm-env
```

## Data Processing
As we separated the VAE processing from the training process, the data processing falls in 2 main parts. As our ressources were limited, we did not look to scale up the images.

#### Bag Processing
We mainly trained on the [SCAND](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/0PRYRH) and the [NavWare](https://anr-navware.github.io/navwareset) dataset.

In a first part you might want to process the Rosbags by following the instruction in [VisualNav-Transformer](https://github.com/robodhruv/visualnav-transformer) repo which we integrated as a submodule in ```  ./latent-encoding```. We removed the unused part and only kept the Bag processing part.
You might need to initiate another venv of its own.

After this processing you should end up with the following structure

```
├──./latent-encoding/ datasets
    ├── <dataset_name>
    │   ├── <name_of_traj1>
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   ├── ...
    │   │   ├── T_1.jpg
    │   │   └── traj_data.pkl
    │   ├── <name_of_traj2>
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   ├── ...
    │   │   ├── T_2.jpg
    │   │   └── traj_data.pkl
    │   ...
    └── └── <name_of_trajN>
        	├── 0.jpg
          	├── 1.jpg
    	    ├── ...
            ├── T_N.jpg
            └── traj_data.pkl
```  
You can then move the data up to the ``` ./data```  folder. 
```bash
mv <dataset_name> ../../../data
```

#### VAE Pre-processing

To pre-process the images into ready-to-use latents you just need to execute `encode_latents.py` with the relevant arguments and you should end up with the following structure:

```
├──./latents
    ├── <dataset_name>
    │   ├── <name_of_traj1>
    │   │   ├── 0.pt
    │   │   ├── 1.pt
    │   │   └── ...
    │   ├── <name_of_traj2>
    │   │   ├── 0.pt
    │   │   ├── 1.pt
    │   │   └── ...
    │   ...
    └── └── <name_of_trajN>
        	├── 0.pt
          	├── 1.pt
    	    └── ...
```  

## Training setup

In order to train
