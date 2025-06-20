import yaml
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from IPython.display import display, HTML
import ipywidgets as widgets
from diffusers.models import AutoencoderKL
import random

from diffusion import create_diffusion
from isolated_nwm_infer import model_forward_wrapper
from misc import transform
from models import CDiT_models
from datasets import TrainingDataset

EXP_NAME = 'nwm_cdit_xl'
MODEL_PATH = f'logs/{EXP_NAME}/checkpoints/0100000.pth.tar'

with open("config/data_config.yaml", "r") as f:
    default_config = yaml.safe_load(f)
config = default_config

with open(f'config/{EXP_NAME}.yaml', "r") as f:
    user_config = yaml.safe_load(f)
config.update(user_config)
latent_size = config['image_size'] // 8

print("loading model")
model = CDiT_models[config['model']](input_size=latent_size, context_size=config['context_size'])
ckp = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
print(model.load_state_dict(ckp["ema"], strict=True))
model.eval()
device = 'cuda:0'
model.to(device)
model = torch.compile(model)

diffusion = create_diffusion(str(250))
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
latent_size = config['image_size'] // 8

def url_to_pil_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

def load_internet_image(url):
    from torchvision import transforms
    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    img = url_to_pil_image(url)
    x_start = _transform(img)
    return x_start.unsqueeze(0).expand(config['context_size'], x_start.shape[0], x_start.shape[1], x_start.shape[2])

def reset():
    x_cond_pixels = x_start
    reconstructed_image=x_cond_pixels.to(device)
    preds['x_cond_pixels_display'] = (reconstructed_image[-1] * 127.5 + 127.5).clamp(0, 255).permute(1, 2, 0).to("cpu", dtype=torch.uint8).numpy()
    preds['x_cond_pixels'] = x_cond_pixels
    preds['video'] = [preds['x_cond_pixels_display']]

def read_tensors_from_file(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    blocs = content.split('torch.tensor')
    tensors = []

    for bloc in blocs[1:]:
        text_tensor = "torch.tensor" + bloc.strip().split(')')[0] + ')'
        try:
            tensor = eval(text_tensor, {'torch': torch})
            tensors.append(tensor)
        except Exception as e:
            print(f"Tenseur ignoré : {e}")
    
    return tensors

def apply_command(y_tensor):
    y = y_tensor.unsqueeze(0).to(device)
    x_cond_pixels = preds['x_cond_pixels'][-1:].unsqueeze(0).to(device)

    samples = model_forward_wrapper(
        (model, diffusion, vae), 
        x_cond_pixels, 
        y, 
        None, 
        latent_size, 
        device, 
        config["context_size"], 
        num_goals=1, 
        rel_t=rel_t, 
        progress=True
    )

    preds['x_cond_pixels'] = torch.cat([preds['x_cond_pixels'].to(samples.device), samples], dim=0)
    samples_disp = (samples * 127.5 + 127.5).permute(0, 2, 3, 1).clamp(0, 255).to("cpu", dtype=torch.uint8).numpy()
    preds['video'].append(samples_disp[0])
    display(Image.fromarray(samples_disp[0]))

def automatic_sequence_from_file(file):
    tensors = read_tensors_from_file(file)
    if not tensors:
        print("No tensor found")
        return

    reset()
    # random sequence select
    commands = random.choice(tensors)

    print("Selected tensor :", commands.shape)
    for i, command in enumerate(commands):
        if i<32:
            print(f"Step {i} - Command : {command.tolist()}")
            apply_command(command)
        else:
            break

    print("Séquence terminée. Images stockées dans preds['video'].")


x_start = load_internet_image('https://raw.githubusercontent.com/FBI-openup/nwm/main/warehouse%20Images/Low.png')


def main():

    preds={}

    automatic_sequence_from_file("actions.txt")