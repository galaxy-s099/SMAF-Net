import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class ABIDEMultiAtlasDataset(Dataset):
    def __init__(self, data_root, indices=None):
        self.data_root = Path(data_root)

        self.X_aal = np.load(self.data_root / "X_aal.npy").astype(np.float32)
        self.X_cc200 = np.load(self.data_root / "X_cc200.npy").astype(np.float32)
        self.X_ho = np.load(self.data_root / "X_ho.npy").astype(np.float32)
        self.y = np.load(self.data_root / "labels.npy").astype(np.int64)

        self.X_aal = np.clip(self.X_aal, -1.0, 1.0)
        self.X_cc200 = np.clip(self.X_cc200, -1.0, 1.0)
        self.X_ho = np.clip(self.X_ho, -1.0, 1.0)

        if indices is None:
            self.indices = np.arange(len(self.y))
        else:
            self.indices = np.array(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        sample = {
            "aal": torch.tensor(self.X_aal[real_idx], dtype=torch.float32),
            "cc200": torch.tensor(self.X_cc200[real_idx], dtype=torch.float32),
            "ho": torch.tensor(self.X_ho[real_idx], dtype=torch.float32),
            "label": torch.tensor(self.y[real_idx], dtype=torch.long),
        }

        return sample


def load_labels(data_root):
    data_root = Path(data_root)
    return np.load(data_root / "labels.npy").astype(np.int64)