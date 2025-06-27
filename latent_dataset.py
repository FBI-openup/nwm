import os
import torch
import pickle
import numpy as np
from typing import List, Tuple
from torch.utils.data import Dataset
from misc import normalize_data, to_local_coords, angle_difference, get_delta_np


class LatentBaseDataset(Dataset):
    def __init__(
        self,
        latent_folder: str,
        data_split_folder: str,
        dataset_name: str,
        context_size: int,
        len_traj_pred: int,
        traj_stride: int,
        min_dist_cat: int,
        max_dist_cat: int,
        traj_names: str,
        normalize: bool = True,
        predefined_index: str = None,
        goals_per_obs: int = 1,
    ):
        super().__init__()
        self.latent_folder = latent_folder
        self.dataset_name = dataset_name
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.traj_stride = traj_stride
        self.goals_per_obs = goals_per_obs
        self.min_dist_cat = min_dist_cat
        self.max_dist_cat = max_dist_cat
        self.normalize = normalize

        traj_names_file = os.path.join(data_split_folder, traj_names)
        with open(traj_names_file, "r") as f:
            self.traj_names = [line.strip() for line in f if line.strip()]

        if predefined_index:
            with open(predefined_index, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)
        else:
            self.index_to_data, self.goals_index = self._build_index()
            index_path = os.path.join(
                data_split_folder,
                f"dataset_dist_{min_dist_cat}_to_{max_dist_cat}_n{context_size}_len_traj_pred_{len_traj_pred}.pkl",
            )
            with open(index_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _get_trajectory(self, trajectory_name: str):
        with open(os.path.join(self.latent_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
            traj_data = pickle.load(f)
        for k, v in traj_data.items():
            traj_data[k] = v.astype("float")
        return traj_data

    def _build_index(self):
        index = []
        goals_index = []
        for traj_name in self.traj_names:
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])
            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size - 1
            end_time = traj_len - self.len_traj_pred
            for curr_time in range(begin_time, end_time, self.traj_stride):
                max_goal_distance = min(self.max_dist_cat, traj_len - curr_time - 1)
                min_goal_distance = max(self.min_dist_cat, -curr_time)
                index.append((traj_name, curr_time, min_goal_distance, max_goal_distance))
        return index, goals_index

    def __len__(self):
        return len(self.index_to_data)

    def _compute_actions(self, meta: dict, curr_time: int) -> np.ndarray:
        positions = np.array(meta["position"])
        yaws = np.array(meta["yaw"])
        start = curr_time
        end = curr_time + self.len_traj_pred + 1
        waypos = to_local_coords(positions[start:end], positions[start], yaws[start])
        wayyaw = angle_difference(yaws[start], yaws[start:end])
        return np.concatenate([waypos, wayyaw.reshape(-1, 1)], axis=-1)


class LatentTrainingDataset(LatentBaseDataset):
    def __getitem__(self, index: int):
        traj_name, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[index]
        traj_path = os.path.join(self.latent_folder, traj_name)
        pkl_path = os.path.join(traj_path, "traj_data.pkl")
        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)

        goal_offsets = np.random.randint(min_goal_dist, max_goal_dist + 1, size=(self.goals_per_obs,))
        goal_times = (goal_offsets + curr_time).tolist()
        rel_time = goal_offsets.astype(np.float32) / float(self.len_traj_pred)

        indices = list(range(curr_time - self.context_size + 1, curr_time + 1)) + goal_times
        latents = []
        for t in indices:
            pt_path = os.path.join(traj_path, f"{t}.pt")
            data = torch.load(pt_path)
            latents.append(data["latent"].unsqueeze(0) if isinstance(data, dict) else data.unsqueeze(0))
        x = torch.cat(latents, dim=0)

        actions = self._compute_actions(meta, curr_time)
        goal_vec = actions[self.context_size:self.context_size + self.goals_per_obs]

        if self.normalize:
            stats = {
                'mean': np.array([-0.00716454,  0.00666485]),
                'std': np.array([3.05594113, 2.01350649])
            }  # fixed stats copied from original script
            goal_vec[:, :2] = (goal_vec[:, :2] - stats['mean']) / stats['std']

        return x, torch.tensor(goal_vec, dtype=torch.float32), torch.tensor(rel_time, dtype=torch.float32)


class LatentEvalDataset(LatentBaseDataset):
    def __init__(self, latent_folder: str, traj_names: List[str]):
        self.latent_folder = latent_folder
        self.traj_names = traj_names
        self.indexes = []
        for traj in traj_names:
            files = sorted(f for f in os.listdir(os.path.join(latent_folder, traj)) if f.endswith(".pt"))
            for fname in files:
                idx = int(fname.replace(".pt", ""))
                self.indexes.append((traj, idx))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index: int):
        traj, idx = self.indexes[index]
        traj_path = os.path.join(self.latent_folder, traj)
        latent_path = os.path.join(traj_path, f"{idx}.pt")
        data = torch.load(latent_path)
        latent = data["latent"] if isinstance(data, dict) else data

        with open(os.path.join(traj_path, "traj_data.pkl"), "rb") as f:
            meta = pickle.load(f)
        info = {
            "timestamp": meta["timestamps"][idx],
            "position": meta["position"][idx],
            "yaw": meta["yaw"][idx],
        }
        return latent, info


class TrajectoryEvalDataset(LatentBaseDataset):
    def __init__(self, latent_folder: str, traj_names: List[str]):
        self.latent_folder = latent_folder
        self.traj_names = traj_names

    def __len__(self):
        return len(self.traj_names)

    def __getitem__(self, index: int):
        traj = self.traj_names[index]
        traj_path = os.path.join(self.latent_folder, traj)
        files = sorted(f for f in os.listdir(traj_path) if f.endswith(".pt"))
        latents = []
        for fname in files:
            data = torch.load(os.path.join(traj_path, fname))
            latents.append(data["latent"].unsqueeze(0) if isinstance(data, dict) else data.unsqueeze(0))
        x = torch.cat(latents, dim=0)

        with open(os.path.join(traj_path, "traj_data.pkl"), "rb") as f:
            meta = pickle.load(f)

        return x, meta
