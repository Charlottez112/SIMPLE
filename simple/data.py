import glob

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def custom_collate(data):
    position = np.stack([item["position"] for item in data])
    velocity = np.stack([item["velocity"] for item in data])
    temp = np.reshape(np.stack([item["temperature"] for item in data]), (-1, 1))
    dim = np.stack([item["boxdim"] for item in data])

    position = torch.from_numpy(position)
    velocity = torch.from_numpy(velocity)
    temp = torch.from_numpy(temp)
    dim = torch.from_numpy(dim)

    return {
        "position": position,
        "velocity": velocity,
        "temperature": temp,
        "boxdim": dim,
    }


class SimulationDataset(Dataset):
    def __init__(self, split: str = "train"):
        # TODO
        self.data = []
        for file in glob.glob("Dataset/*"):
            self.data.append(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        return np.load(file_path)


class SimulationDataLoader(DataLoader):
    def __init__(self, *args, split: str = "train", **kwargs) -> None:
        """Initialize the data loader."""
        if "dataset" in kwargs:
            raise ValueError("'dataset' keyword argument is not supported.")

        dataset = SimulationDataset(split)

        super().__init__(*args, dataset=dataset, **kwargs)
