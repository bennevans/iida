import numpy as np
import os
import pickle
import inspect

from torch.utils.data import Dataset

class RobotPushDataset(Dataset):
    def __init__(self, dataset_dir, num_ctx_fn=lambda x: 4):
        """
            dataset_dir - the directory that contains the push data
            num_ctx_fn - a function that returns the number of context points
                        could be a constant or a distribution
        """
        self.dataset_dir = dataset_dir
        self.num_ctx_fn = num_ctx_fn

        dirs = os.listdir(self.dataset_dir)

        self.objects = []
        paths = []

        self.start_states = []
        self.end_states = []
        self.actions = []


        for dir in dirs:
            full_path = os.path.join(self.dataset_dir, dir)
            if os.path.isfile(full_path) and 'lite.pickle' in dir:
                self.objects.append(dir[:-12])

                paths.append(full_path)

                with open(full_path, 'rb') as f:
                    data = pickle.load(f)
                self.start_states.append(np.array(data['transformed_start_pos']))
                self.end_states.append(np.array(data['transformed_end_pos']))
                self.actions.append(np.array(data['actions']))

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

        # print(obj_idx, push_idxs)
        # if push_idxs == 15 and obj_idx == 10:
        #     import ipdb; ipdb.set_trace()
        ret = dict(
            obs=self.start_states[obj_idx][push_idxs].astype(np.float32)[:2],
            act=self.actions[obj_idx][push_idxs].astype(np.float32),
            obs_prime=self.end_states[obj_idx][push_idxs].astype(np.float32)[:2],
            context_obs=self.start_states[obj_idx][ctx_idxs].astype(np.float32)[:, :2],
            context_act=self.actions[obj_idx][ctx_idxs].astype(np.float32),
            context_obs_prime=self.end_states[obj_idx][ctx_idxs].astype(np.float32)[:, :2],
            fov=np.array([], dtype=np.float32),
            object=self.objects[obj_idx]
        )

        return ret