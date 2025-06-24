import os
import torch
from torch.utils.data import Dataset

class LatentSequenceDataset(Dataset):
    def __init__(self, data_root, context_size, goals_per_obs, len_traj_pred=32, traj_stride=1):
        self.data_root = data_root
        self.context_size = context_size
        self.goals_per_obs = goals_per_obs
        self.seq_length = context_size + goals_per_obs
        self.len_traj_pred = len_traj_pred
        self.traj_stride = traj_stride

        self.samples = []  # list of (traj_name, start_idx)

        for traj_name in os.listdir(data_root):
            traj_path = os.path.join(data_root, traj_name)
            latent_files = sorted([f for f in os.listdir(traj_path) if f.endswith(".pt")])
            num_frames = len(latent_files)

            for start_idx in range(0, num_frames - self.seq_length + 1, self.traj_stride):
                self.samples.append((traj_name, start_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        traj_name, start_idx = self.samples[index]
        traj_path = os.path.join(self.data_root, traj_name)

        latents = []
        for i in range(start_idx, start_idx + self.seq_length):
            latent_path = os.path.join(traj_path, f"{i}.pt")
            latent_data = torch.load(latent_path)
            latents.append(latent_data['latent'].unsqueeze(0))

        x = torch.cat(latents, dim=0)  # shape: [T, 4, 20, 20]
        y = torch.zeros((self.goals_per_obs,))  # dummy target (à adapter si nécessaire)
        rel_t = torch.linspace(0, 1, steps=self.goals_per_obs)  # relative time (optionnel)

        return x, y, rel_t
