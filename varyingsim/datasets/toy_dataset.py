
import numpy as np
import os
import pickle
import inspect

from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, location, obs_fn=None):
        self.location = location
        self.obs_fn = obs_fn

        with open(location, 'rb') as f:
            data = pickle.load(f)
        
        self.state = data['X']
        self.next_state = data['Y']
        if 'keys' in data.keys():
            n_key = len(data['keys'])
            N_ep = data['X'].shape[0]
            self.fov = np.zeros((N_ep, n_key))
            for i, k in enumerate(data['keys']):
                self.fov[:, i] = data[k]
        else:
            self.fov = data['psi']

        if 'thetas' in data:
            self.acts = np.expand_dims(data['thetas'], 2)

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("idx must be less than length!")

        N = self.state[idx].shape[0]
        state = np.stack(self.state[idx])
        next_state = np.stack(self.next_state[idx])
        if 'acts' in dir(self):
            act = self.acts[idx]
        else:
            act = np.zeros((N, 1))
    
        fov = self.fov[idx].repeat(N).reshape((N, -1), order='F')

        ret = dict(
            obs=state.astype(np.float32),
            obs_prime=next_state.astype(np.float32),
            act=act.astype(np.float32),
            fov=fov.astype(np.float32),
        )
        return ret
