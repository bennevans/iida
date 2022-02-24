import numpy as np
import os
import pickle
import inspect

from torch.utils.data import Dataset

class URRobotPushDataset(Dataset):
    def __init__(self, dataset_dir= '/data/datasets/iida_data/train', num_ctx_fn=lambda x: 4, vis = False):
        """
            dataset_dir - the directory that contains the push data
            num_ctx_fn - a function that returns the number of context points
                        could be a constant or a distribution
        """
        self.dataset_dir = dataset_dir
        self.num_ctx_fn = num_ctx_fn

        files = os.listdir(self.dataset_dir)

        self.objects = []

        self.start_states = []
        self.end_states = []
        self.actions = []


        for file in files:
            full_path = os.path.join(self.dataset_dir, file)
            obj = file.split('_full.pkl')[0]
            self.objects.append(obj) #TODO = FIX

            with open(full_path, 'rb') as f:
                data = pickle.load(f)

            self.start_states.append(np.array(data['obs']))
            self.end_states.append(np.array(data['obs_prime']))
            self.actions.append(np.array(data['act']))

            # flags = data['flag']
            # keep_idxs = np.where(flags)
            # self.start_states.append(np.array(data[obs])[keep_idxs])
            # self.end_states.append(np.array(data[obs_prime])[keep_idxs])
            # self.actions.append(np.array(data['act'])[keep_idxs])

        self.lens = np.array([len(s) for s in self.start_states])
        self.cum_lens = np.cumsum(self.lens)
        self.N = sum([len(s) for s in self.start_states])

    def get_buffer_idxs(self, index):
        object_idx = np.where(index < self.cum_lens)[0][0]
        push_idx = index - self.cum_lens[object_idx - 1] if object_idx > 0 else index
        return object_idx, push_idx

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        obj_idx, push_idxs = self.get_buffer_idxs(index)
        num_ctx = self.num_ctx_fn(index)

        ctx_idxs = np.random.randint(self.lens[obj_idx], size=num_ctx)

        #   TODO - Add dep, img
        ret = dict(
            obs=self.start_states[obj_idx][push_idxs].astype(np.float32)[:2],
            act=self.actions[obj_idx][push_idxs].astype(np.float32)[:2], # For nowonly theta and vel are learnable
            obs_prime=self.end_states[obj_idx][push_idxs].astype(np.float32)[:2],
            context_obs=self.start_states[obj_idx][ctx_idxs].astype(np.float32)[:, :2],
            context_act=self.actions[obj_idx][ctx_idxs].astype(np.float32)[:, :2],
            context_obs_prime=self.end_states[obj_idx][ctx_idxs].astype(np.float32)[:, :2],
            fov=np.array([], dtype=np.float32),
            object=self.objects[obj_idx]
        )

        return ret