import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader

def custom_collate(data):
    position = np.stack([item["position"] for item in data])
    velocity = np.stack([item["velocity"] for item in data])
    temp = np.reshape(np.stack([item["temperature"] for item in data]) , (-1,1))
    config = np.reshape(np.stack([item["init_config"] for item in data]) , (-1,1)) 
        
    position = torch.from_numpy(position)
    velocity = torch.from_numpy(velocity)
    temp = torch.from_numpy(temp)
    config = torch.from_numpy(config)
    return {"position" : position , "velocity" : velocity , "temperature" : temp, "init_config" : config}


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
        current_idx = {"position" : simulation['position'] , "velocity" : simulation['velocity']  , "temperature" : simulation['temperature'].item() , "init_config" : simulation['init_config']}
        return current_idx


