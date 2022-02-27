# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 08:50:48 2022
"""

import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader
#import pdb

def custom_collate(data):
    l = len(data)
    state = np.zeros( (l , 4096 , 6, 60))
    t = np.zeros( (l , 1) )
    i = 0
    for item in data:
        state[i , :, :, :] = item["state"]
        t[i, 0] = item["temperature"]
        i+=1

    state_tensor = torch.from_numpy(state)
    t_tensor = torch.from_numpy(t)
    batch = {"state" : state_tensor , "temperature" : t_tensor}
    return batch


class CustomDataset(Dataset):
    def __init__(self, n_particles = 4096, n_frame = 60):
        self.n_particles = n_particles
        self.n_frame = n_frame
        self.data = []
        for file in glob.glob('Dataset/*'):
            self.data.append(np.load(file))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        simulation = self.data[idx]
        velocity, position = simulation['velocity'] ,  simulation['position']
        frame_temp  = simulation['temperature'].item()        
        frames = np.zeros((self.n_particles , 6 , self.n_frame))
        
        for i in range(0 , self.n_frame):
            frames[:, 0:3, i] , frames[:, 3:6, i] = position[i , :, :], velocity[i, :, :]

        current_idx = {"state" : frames , "temperature" : frame_temp}
        return current_idx

if __name__ == '__main__':

    dataset = CustomDataset()
    dl = DataLoader(dataset, batch_size = 2, shuffle=False, collate_fn = custom_collate)


