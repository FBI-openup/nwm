#!/usr/bin/env python
"""
find_bad_latents.py – scan *.pt / *.safetensors latent checkpoints and
list those whose latent tensor isn't the expected spatial size.

•  Skips any *.pkl files (useful when trajectory data lives in the same tree).
•  Optionally understands SafeTensors if the library is installed.

Author: ChatGPT – 2025-07-02
"""

import argparse, glob, itertools, sys, torch
from pathlib import Path

# ---------- optional SafeTensors support ---------------------------------- #
try:
    from safetensors.torch import load_file as safe_load
except ImportError:
    safe_load = None  # script still works; just can't read *.safetensors
# -------------------------------------------------------------------------- #

# --------------------- helpers -------------------------------------------- #
def load_checkpoint(path):
    """
    First try torch.load; if that fails and SafeTensors is available,
    try safetensors.torch.load_file. Return (obj, err_msg).
    """
    try:
        return torch.load(path, map_location="cpu"), None
    except Exception as e:
        if safe_load and path.suffix == ".safetensors":
            try:
                return safe_load(str(path)), None
            except Exception:
                pass
        return None, str(e)


def extract_tensor(obj, keys):
    """Return the first tensor found inside obj; else None."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and isinstance(obj[k], torch.Tensor):
                return obj[k]
    return None


# ---------------------- main ---------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="latents/scand",
                   help="directory tree to scan (recursive)")
    p.add_argument("--expected", type=int, default=28,
                   help="expected spatial size (e.g. 28 for 28×28 latents)")
    p.add_argument("--keys", nargs="+", default=["latent", "z", "image_latent"],
                   help="possible dict keys that hold the latent tensor")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-file output; only show the summary")
    args = p.parse_args()

    # --- build iterator for *.pt and *.safetensors, skip *.pkl ------------- #
    pattern_pt = str(Path(args.root) / "**/*.pt")
    pattern_st = str(Path(args.root) / "**/*.safetensors")
    files = itertools.chain(glob.iglob(pattern_pt, recursive=True),
                            glob.iglob(pattern_st, recursive=True))

    bad, unreadable, total = [], [], 0

    for fname in files:
        path = Path(fname)

        # we *could* hit rogue *.pkl here if someone mis-named a file;
        # double-check and skip defensively
        if path.suffix == ".pkl":
            continue

        obj, err = load_checkpoint(path)
        if obj is None:
            unreadable.append((path, err))
            if not args.quiet:
                print(f"⚠️  unreadable {path}: {err}", file=sys.stderr)
            continue

        t = extract_tensor(obj, args.keys)
        if t is None:          # not something we recognise – skip
            continue

        total += 1
        if t.shape[-1] != args.expected:
            bad.append((path, tuple(t.shape)))
            if not args.quiet:
                print(f"{t.shape}  {path}")

    # ------------------ summary ------------------------------------------- #
    print("\nSummary")
    print("-------")
    print(f"✓ {total - len(bad)} correct")
    print(f"✗ {len(bad)} mismatched size")
    print(f"⚠️ {len(unreadable)} unreadable\n")

    if bad:
        print("Next steps:")
        print("  • Delete or regenerate the bad files above, OR")
        print("  • Interpolate/pad them on-the-fly in your dataset.")


if __name__ == "__main__":
    main()
