import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
DEFAULT_BATCH_SIZE = 32  # Default batch size for VAE encoding

def main(input_dir: str, output_dir: str, vae_model: str, batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Encode images to latents using VAE with batch processing.
    
    Args:
        input_dir: Input dataset root directory
        output_dir: Output latents root directory  
        vae_model: VAE model name or path
        batch_size: Batch size for VAE encoding
    """
    try:
        # Load VAE model
        print(f"Loading VAE model: {vae_model}")
        vae = AutoencoderKL.from_pretrained(vae_model).to(DEVICE)
        vae.eval()
        print(f"VAE model loaded successfully on {DEVICE}")
    except Exception as e:
        print(f"ERROR: Failed to load VAE model '{vae_model}': {e}")
        return False

    # Validate input directory
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return False
    
    if not os.path.isdir(input_dir):
        print(f"ERROR: Input path is not a directory: {input_dir}")
        return False

    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory prepared: {output_dir}")
    except Exception as e:
        print(f"ERROR: Failed to create output directory '{output_dir}': {e}")
        return False

    total_processed = 0
    total_errors = 0
    total_skipped = 0

    # Process each trajectory
    trajectories = sorted(os.listdir(input_dir))
    for traj in tqdm(trajectories, desc="Processing trajectories"):
        src_traj = os.path.join(input_dir, traj)
        src_pkl  = os.path.join(src_traj, "traj_data.pkl")

        # Skip if invalid directory or missing .pkl
        if not os.path.isdir(src_traj):
            tqdm.write(f"[SKIP] {traj}: not a directory")
            total_skipped += 1
            continue
            
        if not os.path.isfile(src_pkl):
            tqdm.write(f"[SKIP] {traj}: missing traj_data.pkl")
            total_skipped += 1
            continue

        dst_traj = os.path.join(output_dir, traj)
        try:
            os.makedirs(dst_traj, exist_ok=True)
        except Exception as e:
            tqdm.write(f"[ERROR] {traj}: failed to create output directory: {e}")
            total_errors += 1
            continue

        # Copy the .pkl file
        dst_pkl = os.path.join(dst_traj, "traj_data.pkl")
        if not os.path.exists(dst_pkl):
            try:
                shutil.copy2(src_pkl, dst_pkl)
            except Exception as e:
                tqdm.write(f"[ERROR] {traj}: failed to copy traj_data.pkl: {e}")
                total_errors += 1
                continue

        # List all images
        try:
            all_files = os.listdir(src_traj)
            imgs = sorted(f for f in all_files if f.lower().endswith(".jpg"))
        except Exception as e:
            tqdm.write(f"[ERROR] {traj}: failed to list images: {e}")
            total_errors += 1
            continue
            
        if not imgs:
            tqdm.write(f"[SKIP] {traj}: no .jpg images found")
            total_skipped += 1
            continue

        # Process images in batches
        traj_processed, traj_errors = process_images_batch(
            vae, src_traj, dst_traj, imgs, batch_size, traj
        )
        
        total_processed += traj_processed
        total_errors += traj_errors

    # Print summary
    print(f"\n=== Processing Summary ===")
    print(f"Total images processed: {total_processed}")
    print(f"Total errors: {total_errors}")
    print(f"Total trajectories skipped: {total_skipped}")
    print(f"Success rate: {total_processed/(total_processed+total_errors)*100:.1f}%" if (total_processed+total_errors) > 0 else "No images processed")
    
    return True


def process_images_batch(vae, src_traj: str, dst_traj: str, imgs: list, batch_size: int, traj_name: str):
    """
    Process images in batches for efficient VAE encoding.
    
    Args:
        vae: VAE model
        src_traj: Source trajectory directory
        dst_traj: Destination trajectory directory
        imgs: List of image filenames
        batch_size: Batch size for processing
        traj_name: Trajectory name for logging
        
    Returns:
        tuple: (processed_count, error_count)
    """
    processed_count = 0
    error_count = 0
    
    # Filter out already encoded images
    pending_imgs = []
    for img_name in imgs:
        dst_lat = os.path.join(dst_traj, img_name.replace(".jpg", ".pt"))
        if not os.path.exists(dst_lat):
            pending_imgs.append(img_name)
    
    if not pending_imgs:
        return 0, 0
    
    # Process in batches
    for i in range(0, len(pending_imgs), batch_size):
        batch_imgs = pending_imgs[i:i+batch_size]
        
        # Load batch images
        batch_tensors = []
        batch_paths = []
        batch_dst_paths = []
        
        for img_name in batch_imgs:
            src_img = os.path.join(src_traj, img_name)
            dst_lat = os.path.join(dst_traj, img_name.replace(".jpg", ".pt"))
            
            try:
                img = Image.open(src_img).convert("RGB")
                x = transform(img)  # [3,224,224]
                batch_tensors.append(x)
                batch_paths.append(src_img)
                batch_dst_paths.append(dst_lat)
            except Exception as e:
                tqdm.write(f"[ERROR] {traj_name}: failed to load {img_name}: {e}")
                error_count += 1
        
        if not batch_tensors:
            continue
            
        try:
            # Stack and move to device
            batch_input = torch.stack(batch_tensors).to(DEVICE)  # [batch_size, 3, 224, 224]
            
            # Encode batch
            with torch.no_grad():
                batch_latents = vae.encode(batch_input).latent_dist.sample() * 0.18215
            
            # Save individual latents
            for j, (latent, dst_path) in enumerate(zip(batch_latents, batch_dst_paths)):
                try:
                    torch.save({"latent": latent.cpu()}, dst_path)
                    processed_count += 1
                except Exception as e:
                    tqdm.write(f"[ERROR] {traj_name}: failed to save {os.path.basename(dst_path)}: {e}")
                    error_count += 1
                    
        except Exception as e:
            tqdm.write(f"[ERROR] {traj_name}: batch encoding failed for batch {i//batch_size + 1}: {e}")
            error_count += len(batch_tensors)
        
        # Clear GPU cache periodically
        if torch.cuda.is_available() and (i // batch_size + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    return processed_count, error_count

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Encode images to latents using VAE with batch processing")
    p.add_argument("-i", "--input-dir", required=True, help="Input dataset root directory")
    p.add_argument("-o", "--output-dir", required=True, help="Output latents root directory")
    p.add_argument("--vae-model", default=VAE_PATH, help="Stable Diffusion VAE model name or path")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, 
                   help=f"Batch size for VAE encoding (default: {DEFAULT_BATCH_SIZE})")
    args = p.parse_args()
    
    print(f"Starting latent encoding with batch size: {args.batch_size}")
    success = main(args.input_dir, args.output_dir, args.vae_model, args.batch_size)
    
    if success:
        print("Latent encoding completed successfully!")
    else:
        print("Latent encoding failed!")
        exit(1)
