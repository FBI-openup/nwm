import torch, glob, pprint, os, itertools

sample = next(itertools.islice(glob.iglob("latents/scand/**/*.pt", recursive=True), 1, None))
obj = torch.load(sample, map_location="cpu")
print("Type:", type(obj))
if isinstance(obj, dict):
    print("Dict keys:", obj.keys())
    # Print the shape of any tensor values
    for k, v in obj.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
