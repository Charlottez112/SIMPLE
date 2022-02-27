import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader


def custom_collate(data):
    l = len(data)
    state = np.zeros( (l , 60, 4096, 6))
    t = np.zeros( (l , 1) )
    i = 0
    for item in data:
        state[i , :, :, 0:3] = item["position"]
        state[i , :, :, 3:6] = item["velocity"]
        t[i, 0] = item["temperature"]
        i+=1
        
    state = torch.from_numpy(state)
    t = torch.from_numpy(t)
    return {"position" : state[: , :, :, 0:3] , "velocity" : state[: , :, :, 3:6] , "temperature" : t}


class LoadSimulation(Dataset):
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
        current_idx = {"position" : simulation['position'] , "velocity" : simulation['velocity']  , "temperature" : simulation['temperature'].item()}
        return current_idx


