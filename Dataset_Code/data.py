import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader

def custom_collate(data):
    position = np.stack([item["position"] for item in data])
    velocity = np.stack([item["velocity"] for item in data])
    temp = np.reshape(np.stack([item["temperature"] for item in data]) , (-1,1))
    dim = np.stack([item["boxdim"] for item in data])
        
    position = torch.from_numpy(position)
    velocity = torch.from_numpy(velocity)
    temp = torch.from_numpy(temp)
    dim = torch.from_numpy(dim)
    return {"position" : position , "velocity" : velocity , "temperature" : temp, "boxdim" : dim}


class SimulationDataset(Dataset):
    def __init__(self, split, split_ratio=[0.6, 0.2, 0.2], n_particles = 4096, n_frame = 60):
        self.n_particles = n_particles
        self.n_frame = n_frame
        self.path = []
        for file in glob.glob('Dataset/*'):
            self.path.append(file)
        if split == "train":
            init_idx = 0
            end_idx = int(len(self.path) * split_ratio[0])
        elif split == "test":
            init_idx = int(len(self.path) * split_ratio[0]) 
            end_idx = int(len(self.path) * (split_ratio[0] + split_ratio[1]))
        elif split == "eval":
            init_idx = int(len(self.path) * (split_ratio[0] + split_ratio[1]))
            end_idx = int(len(self.path))
        self.data = self.path[init_idx:end_idx]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_path = self.data[idx]
        return np.load(file_path)


class SimulationDataLoader(DataLoader):
    def __init__(self):
        super().__init__(collate_fn=custom_collate)
