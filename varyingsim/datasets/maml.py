from varyingsim.datasets.fov_dataset import EpisodicStartEndFovDataset
from varyingsim.datasets.relabeled_dataset import RelabeledEpisodicFovDataset
from varyingsim.datasets.ur_robot_push_dataset import URRobotPushDataset
import numpy as np

class MAMLDataset(EpisodicStartEndFovDataset):
    
    def __init__(self, location, k_shot, k_query, obs_fn=None):
        super().__init__(location, obs_fn=obs_fn)
        self.k_shot = k_shot
        self.k_query = k_query

    def __getitem__(self, idx):
        train_idx = np.random.randint(super().__len__())
        test_idx = np.random.randint(super().__len__())
        train_data = super().__getitem__(train_idx)
        test_data = super().__getitem__(test_idx)

        index_train = np.random.choice(len(train_data['state']), self.k_shot, True)
        index_test = np.random.choice(len(test_data['state']), self.k_query, True)


        if self.obs_fn:
            train_obs = self.obs_fn(train_data['state'][index_train]).astype(np.float32)
            train_obs_prime = self.obs_fn(train_data['state_prime'][index_train]).astype(np.float32)
            test_obs = self.obs_fn(test_data['state'][index_test]).astype(np.float32)
            test_obs_prime = self.obs_fn(test_data['state_prime'][index_test]).astype(np.float32)
        else:
            train_obs = train_data['state'][index_train].astype(np.float32)
            train_obs_prime = train_data['state_prime'][index_train].astype(np.float32)
            test_obs = test_data['state'][index_test].astype(np.float32)
            test_obs_prime = test_data['state_prime'][index_test].astype(np.float32)

        train_x = np.concatenate([train_obs, train_data['act'][index_train]], axis=1)
        train_y = train_obs_prime
        test_x = np.concatenate([test_obs, test_data['act'][index_test]], axis=1)
        test_y = test_obs_prime

        return train_x, train_y, test_x, test_y

class MAMLRelabeledDataset(RelabeledEpisodicFovDataset):
    def __init__(self, location, k_shot, k_query, obs_fn=None, normalize=True):
        super().__init__(location, obs_fn=obs_fn)
        self.k_shot = k_shot
        self.k_query = k_query
        self.normalize = normalize

        if self.normalize:
            self.mean_state = np.mean(self.state, axis=(0,1))
            self.mean_act = np.mean(self.act, axis=(0,1))
            self.mean_state_prime = np.mean(self.state_prime, axis=(0,1))

    def __len__(self):
        return len(self.lens)

    def __getitem__(self, idx):
        train_idx = np.random.randint(len(self.lens))
        test_idx = np.random.randint(len(self.lens))

        train_obs = self.state[train_idx]
        train_act = self.act[train_idx]
        train_obs_prime = self.state_prime[train_idx]

        test_obs = self.state[test_idx]
        test_act = self.act[test_idx]
        test_obs_prime = self.state_prime[test_idx]

        index_train = np.random.choice(len(train_obs), self.k_shot, True)
        index_test = np.random.choice(len(test_obs), self.k_query, True)

        train_x = np.concatenate([train_obs[index_train], train_act[index_train]], axis=1)
        train_y = train_obs_prime[index_train]
        test_x = np.concatenate([test_obs[index_test], test_act[index_test]], axis=1)
        test_y = test_obs_prime[index_test]

        if self.normalize:
            train_x -= np.concatenate([self.mean_state, self.mean_act])
            train_y -= self.mean_state_prime
            test_x -= np.concatenate([self.mean_state, self.mean_act])
            test_y -= self.mean_state_prime

        return train_x.astype(np.float32), train_y.astype(np.float32), test_x.astype(np.float32), test_y.astype(np.float32)

class MAMLRobotDataset(URRobotPushDataset):

    def __init__(self, location, k_shot, k_query, normalize=False):
        super().__init__(location)
        self.k_shot = k_shot
        self.k_query = k_query
        self.normalize = normalize

        if self.normalize:
            self.mean_state = np.mean(np.concatenate(self.start_states)[:, :2], axis=0)
            self.mean_act = np.mean(np.concatenate(self.actions)[:, :2], axis=0)
            self.mean_state_prime = np.mean(np.concatenate(self.end_states)[:, :2], axis=0)


    def __getitem__(self, idx):
        train_idx = np.random.randint(len(self.lens))
        test_idx = np.random.randint(len(self.lens))

        train_obs = self.start_states[train_idx]
        train_act = self.actions[train_idx]
        train_obs_prime = self.end_states[train_idx]

        test_obs = self.start_states[test_idx]
        test_act = self.actions[test_idx]
        test_obs_prime = self.end_states[test_idx]

        index_train = np.random.choice(len(train_obs), self.k_shot, True)
        index_test = np.random.choice(len(test_obs), self.k_query, True)
        train_x = np.concatenate([train_obs[index_train, :2], train_act[index_train, :2]], axis=1)
        train_y = train_obs_prime[index_train, :2]
        test_x = np.concatenate([test_obs[index_test, :2], test_act[index_test, :2]], axis=1)
        test_y = test_obs_prime[index_test, :2]

        if self.normalize:
            train_x -= np.concatenate([self.mean_state, self.mean_act])
            train_y -= self.mean_state_prime
            test_x -= np.concatenate([self.mean_state, self.mean_act])
            test_y -= self.mean_state_prime

        return train_x.astype(np.float32), train_y.astype(np.float32), test_x.astype(np.float32), test_y.astype(np.float32)

import torch
class iMAMLRobotDataset(MAMLRobotDataset):

    def __init__(self, location, k_shot, k_query, device='cpu', normalize=False):
        super().__init__(location, k_shot, k_query, normalize=normalize)
        self.device = device

    def __getitem__(self, idx):
        train_x, train_y, test_x, test_y = super().__getitem__(idx)
        train_x = torch.from_numpy(train_x).to(self.device)
        train_y = torch.from_numpy(train_y).to(self.device)
        test_x = torch.from_numpy(test_x).to(self.device)
        test_y = torch.from_numpy(test_y).to(self.device)
        return dict(x_train=train_x, y_train=train_y, x_val=test_x, y_val=test_y)

class iMAMLDataset(MAMLDataset):
    def __init__(self, location, k_shot, k_query, obs_fn=None, device='cpu'):
        super().__init__(location, k_shot, k_query, obs_fn=obs_fn)
        self.device = device

    def __getitem__(self, idx):
        train_x, train_y, test_x, test_y = super().__getitem__(idx)
        train_x = torch.from_numpy(train_x).to(self.device)
        train_y = torch.from_numpy(train_y).to(self.device)
        test_x = torch.from_numpy(test_x).to(self.device)
        test_y = torch.from_numpy(test_y).to(self.device)
        return dict(x_train=train_x, y_train=train_y, x_val=test_x, y_val=test_y)

class iMAMLRelabeledDataset(MAMLRelabeledDataset):
    def __init__(self, location, k_shot, k_query, device='cpu', normalize=False):
        super().__init__(location, k_shot, k_query, normalize=normalize)
        self.device = device

    def __getitem__(self, idx):
        train_x, train_y, test_x, test_y = super().__getitem__(idx)
        train_x = torch.from_numpy(train_x).to(self.device)
        train_y = torch.from_numpy(train_y).to(self.device)
        test_x = torch.from_numpy(test_x).to(self.device)
        test_y = torch.from_numpy(test_y).to(self.device)
        return dict(x_train=train_x, y_train=train_y, x_val=test_x, y_val=test_y)