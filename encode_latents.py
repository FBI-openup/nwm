import os
import torch
from tqdm import tqdm
from PIL import Image
from diffusers.models import AutoencoderKL
from torchvision import transforms
from misc import transform  # ta transformation resize(160x160) + normalize

# === Configuration ===
DATA_DIR = "data/scand"
OUTPUT_DIR = "latents/scand"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
VAE_PATH = "stabilityai/sd-vae-ft-ema"

# === Charger le VAE ===
tokenizer = AutoencoderKL.from_pretrained(VAE_PATH).to(DEVICE)
tokenizer.eval()

# === Parcourir les trajets ===
for traj_name in tqdm(os.listdir(DATA_DIR), desc="Trajectoires"):
    traj_path = os.path.join(DATA_DIR, traj_name)
    if not os.path.isdir(traj_path):
        continue

    out_traj_path = os.path.join(OUTPUT_DIR, traj_name)
    os.makedirs(out_traj_path, exist_ok=True)

    # === Fichiers images ===
    image_files = sorted([f for f in os.listdir(traj_path) if f.endswith(".jpg")])

    for img_file in tqdm(image_files, desc=f"{traj_name}", leave=False):
        input_path = os.path.join(traj_path, img_file)
        output_path = os.path.join(out_traj_path, img_file.replace(".jpg", ".pt"))

        if os.path.exists(output_path):
            continue  # Skip if already encoded

        try:
            image = Image.open(input_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)  # [1, 3, 160, 160]

            with torch.no_grad():
                latents = tokenizer.encode(image_tensor).latent_dist.sample()
                latents = latents * 0.18215  # scale SD latent space

            torch.save({"latent": latents.squeeze(0).cpu()}, output_path)

        except Exception as e:
            print(f"[Warning] Failed to process {input_path}: {e}")
