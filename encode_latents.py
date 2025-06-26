import os
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from diffusers.models import AutoencoderKL
from torchvision import transforms
from misc import transform


# === Configuration ===
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
VAE_PATH = "stabilityai/sd-vae-ft-ema"

# === Charger le VAE ===
tokenizer = AutoencoderKL.from_pretrained(VAE_PATH).to(DEVICE)
tokenizer.eval()

def main(args: argparse.Namespace):
    # === Parcourir les trajets ===
    for traj_name in tqdm(os.listdir(args.input_dir), desc="Trajectoires"):
        traj_path = os.path.join(args.input_dir, traj_name)
        if not os.path.isdir(traj_path):
            continue

        out_traj_path = os.path.join(args.output_dir, traj_name)
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


if __name__ == "__main__":
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        "-i",
        type= str,
        help= "path of the dataset with images",
        required = True
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="./latents/scand",
        type= str,
        help= "path to the output directory"
    )

    args = parser.parse_args()
    main(args)