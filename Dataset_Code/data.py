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
    def __init__(self, n_particles = 4096, n_frame = 60):
        self.n_particles = n_particles
        self.n_frame = n_frame
        self.data = []
        for file in glob.glob('Dataset/*'):
            self.data.append(file)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_path = self.data[idx]
        simulation = np.load(file_path)
        current_idx = {"position" : simulation['position'] , "velocity" : simulation['velocity']  , "temperature" : simulation['temperature'].item() , "boxdim" : simulation['boxdim']}
        return current_idx


