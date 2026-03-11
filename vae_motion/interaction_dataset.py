import random
import torch
import numpy as np
import joblib
from torch.utils.data import Dataset
from glob import glob
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class InteractionDataset(Dataset):
    def __init__(self, data_root, history_length=4, future_length=4):
        self.history_length = history_length
        self.future_length = future_length
        self.pkl_paths = sorted(glob(f"{data_root}/*.pkl"))

        try:
            self.scaler = joblib.load("smpl_scaler.pkl")
        except:
            print("Warning: smpl_scaler.pkl not found. Running prepare_global_scaler() first.")

            self.scaler = self.prepare_global_scaler(data_root)

    def __len__(self):
        return len(self.pkl_paths)

    @staticmethod
    def prepare_global_scaler(data_root):
        """
        Run this once before training to compute mean/std across the entire dataset.
        """
        paths = glob(f"{data_root}/*.pkl")
        all_samples = []

        print(f"Fitting scaler on {len(paths)} files...")
        for p in paths:
            raw = np.load(p, allow_pickle=True)
            # Concatenate A and B to treat them as the same distribution
            p1 = torch.cat([torch.from_numpy(raw["person1"][k]) for k in ["trans", "root_orient", "pose_body"]], dim=1)
            p2 = torch.cat([torch.from_numpy(raw["person2"][k]) for k in ["trans", "root_orient", "pose_body"]], dim=1)
            all_samples.append(p1.numpy())
            all_samples.append(p2.numpy())

        all_data = np.concatenate(all_samples, axis=0)
        scaler = StandardScaler()
        scaler.fit(all_data)
        joblib.dump(scaler, "smpl_scaler.pkl")
        print("Scaler saved to smpl_scaler.pkl")
        return scaler

    def _normalize(self, x):
        # x: [T, D]
        if self.scaler is None: return x
        return torch.from_numpy(self.scaler.transform(x.numpy())).float()

    def denormalize(self, x):
        # x: [..., D]
        if self.scaler is None: return x
        shape = x.shape
        x_reshaped = x.view(-1, shape[-1]).cpu().numpy()
        x_denorm = self.scaler.inverse_transform(x_reshaped)
        return torch.from_numpy(x_denorm).view(*shape).float()

    def __getitem__(self, idx):
        pkl_path = self.pkl_paths[idx]
        raw = np.load(pkl_path, allow_pickle=True)

        # 1. Collect and Normalize
        A = self._normalize(self._collect_dict(raw["person1"]))
        B = self._normalize(self._collect_dict(raw["person2"]))
        A = self._collect_dict(raw["person1"])
        B = self._collect_dict(raw["person2"])
        T = A.shape[0]
        H, F = self.history_length, self.future_length
        min_len = H + F + 1

        if T < min_len:
            pad = min_len - T
            A = torch.cat([A, A[-1:].repeat(pad, 1)], dim=0)
            B = torch.cat([B, B[-1:].repeat(pad, 1)], dim=0)
            T = A.shape[0]

        t = random.randint(H, T - F - 1)

        return {
            "A_hist": A[t - H: t],
            "B_hist": B[t - H: t],
            "A_future": A[t: t + F],
            "B_future": B[t: t + F],
        }

    @staticmethod
    def _collect_dict(seq):
        return torch.cat([
            torch.from_numpy(seq["trans"]),
            torch.from_numpy(seq["root_orient"]),
            torch.from_numpy(seq["pose_body"]),
        ], dim=1).float()