import os
import shutil
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from diffusers.models import AutoencoderKL
from misc import transform

# === Configuration ===
DEVICE   = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
VAE_PATH = "stabilityai/sd-vae-ft-ema"

def main(input_dir: str, output_dir: str, vae_model: str):
    # Load VAE model
    vae = AutoencoderKL.from_pretrained(vae_model).to(DEVICE)
    vae.eval()

    # Process each trajectory
    for traj in tqdm(sorted(os.listdir(input_dir)), desc="Trajectories"):
        src_traj = os.path.join(input_dir, traj)
        src_pkl  = os.path.join(src_traj, "traj_data.pkl")

        # Skip if invalid directory or missing .pkl
        if not os.path.isdir(src_traj) or not os.path.isfile(src_pkl):
            tqdm.write(f"[SKIP] {traj}: missing traj_data.pkl")
            continue

        dst_traj = os.path.join(output_dir, traj)
        os.makedirs(dst_traj, exist_ok=True)

        # Copy the .pkl file
        dst_pkl = os.path.join(dst_traj, "traj_data.pkl")
        if not os.path.exists(dst_pkl):
            shutil.copy2(src_pkl, dst_pkl)

        # List all images
        imgs = sorted(f for f in os.listdir(src_traj) if f.lower().endswith(".jpg"))
        for img_name in tqdm(imgs, desc=f"  {traj}", leave=False):
            src_img = os.path.join(src_traj, img_name)
            dst_lat = os.path.join(dst_traj, img_name.replace(".jpg", ".pt"))

            # Skip if already encoded
            if os.path.exists(dst_lat):
                continue

            try:
                img = Image.open(src_img).convert("RGB")
                x   = transform(img).unsqueeze(0).to(DEVICE)  # [1,3,224,224]
                with torch.no_grad():
                    lat = vae.encode(x).latent_dist.sample() * 0.18215
                torch.save({"latent": lat.squeeze(0).cpu()}, dst_lat)
            except Exception as e:
                tqdm.write(f"[WARN] failed {src_img}: {e}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Encode images to latents + copy traj.pkl")
    p.add_argument("-i","--input-dir",  required=True, help="input dataset root")
    p.add_argument("-o","--output-dir", required=True, help="output latents root")
    p.add_argument("--vae-model", default=VAE_PATH, help="Stable Diffusion VAE name")
    args = p.parse_args()
    main(args.input_dir, args.output_dir, args.vae_model)
