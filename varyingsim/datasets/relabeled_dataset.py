from torch.utils.data import Dataset
import numpy as np
import pickle
import time

def all_same_len(lens):
    return len(np.unique(lens)) == 1

class RelabeledEpisodicFovDataset(Dataset):
    def __init__(self, location, obs_fn=None, context_size=1):
        self.location = location
        self.obs_fn = obs_fn
        self.context_size = context_size

        with open(location, 'rb') as f:
            data = pickle.load(f)

        self.state = [np.concatenate(s) for s in data['state']]
        self.act = [np.concatenate(a) for a in data['act']]
        self.state_prime = [np.concatenate(s) for s in data['state_prime']]
        self.fov = [np.concatenate(f) for f in data['fov']]

        self.num_episodes = len(self.state)

        self.lens = []

        for trajs in data['state']:
            lens = []
            for traj in trajs:
                lens.append(len(traj))
            self.lens.append(lens)
        
        self.N = np.sum(self.lens)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        """
            returns an element of a batch with the following keys:
            obs, act, obs_prime, context_obs, context_act, context_obs_prime, and fov
            the first 3 should be of dimension d_{obs|act},
            the second 3 should be of dimensions context_size x d{obs|act}

            ignores indexing for now, except to pick from the same episode
        """
        # make it so each same index returns data from same episode
        # episode = idx % self.num_episodes
        episode = np.random.randint(self.num_episodes)

        start_time = time.time()

        states = self.state[episode]
        acts = self.act[episode]
        state_primes = self.state_prime[episode]
        fovs = self.fov[episode]
        lens = self.lens[episode]
        T = np.sum(lens)

        start_rand_time = time.time()

        train_point_idx = np.random.randint(T)
        context_point_idxs = np.random.randint(T, size=self.context_size)

        state = states[train_point_idx]
        act = acts[train_point_idx]
        state_prime = state_primes[train_point_idx]
        context_state = states[context_point_idxs]
        context_act = acts[context_point_idxs]
        context_state_prime = state_primes[context_point_idxs]
        fov = fovs[train_point_idx]

        start_obs_time = time.time()

        if self.obs_fn:
            obs = self.obs_fn(state)
            obs_prime = self.obs_fn(state_prime)
            context_obs = self.obs_fn(context_state)
            context_obs_prime = self.obs_fn(context_state_prime)
        else:
            obs = state
            obs_prime = state_prime
            context_obs = context_state
            context_obs_prime = context_state_prime

        start_dict_time = time.time()
        ret = dict(
            obs=obs.astype(np.float32),
            act=act.astype(np.float32),
            obs_prime=obs_prime.astype(np.float32),
            context_obs=context_obs.astype(np.float32),
            context_act=context_act.astype(np.float32),
            context_obs_prime=context_obs_prime.astype(np.float32),
            fov=fov.astype(np.float32)
        )
        
        end_time = time.time()
        # print('cat time', start_rand_time - start_time)
        # print('rand time', start_obs_time - start_rand_time)
        # print('obs time', start_dict_time - start_obs_time)
        # print('dict time', end_time - start_dict_time)
        # print()
        return ret

    # def __getitem__(self, idx):
    #     if idx >= self.__len__():
    #         raise IndexError("idx must be less than length!")

    #     states = self.state[idx]
    #     acts = self.act[idx]
    #     state_primes = self.state_prime[idx]
    #     fovs = self.fov[idx]

    #     # TODO: for now just set obs = state, but can do obs_fn
    #     lens = [len(s) for s in states]
    #     if all_same_len(lens):
    #         so = np.stack(states).astype(np.float32)
    #         sop = np.stack(state_primes).astype(np.float32)
    #         act = np.stack(acts).astype(np.float32)
    #         fov=np.stack(fovs).astype(np.float32)
    #     else:
    #         max_len = np.max(lens)
    #         so = []
    #         sop = []
    #         act = []
    #         fov = []
    #         d_s = states[0][0].shape[0]
    #         d_a = acts[0][0].shape[0]
    #         d_f = fovs[0][0].shape[0]

    #         for s, a, sp, f in zip(states, acts, state_primes, fovs):
    #             s_len = len(s)
    #             s_pad = np.zeros((max_len, d_s))
    #             sp_pad = np.zeros((max_len, d_s))
    #             a_pad = np.zeros((max_len, d_a))
    #             f_pad = np.zeros((max_len, d_f))
    #             s_pad[:s_len] = np.array(s)
    #             sp_pad[:s_len] = np.array(sp)
    #             a_pad[:s_len] = np.array(a)
    #             f_pad[:s_len] = np.array(f)
    #             so.append(s_pad)
    #             sop.append(sp_pad)
    #             act.append(a_pad)
    #             fov.append(f_pad)

    #         so = np.stack(so).astype(np.float32)
    #         sop = np.stack(sop).astype(np.float32)
    #         act = np.stack(act).astype(np.float32)
    #         fov = np.stack(fov).astype(np.float32)

    #     if self.obs_fn is not None:
    #         o = self.obs_fn(so)
    #         op = self.obs_fn(sop)
    #     else:
    #         o = so
    #         op = sop

    #     ret = dict(
    #         state=so,
    #         obs=o,
    #         act=act,
    #         fov=fov,
    #         state_prime=sop,
    #         obs_prime=op,
    #         lens=lens,
    #     )
        
    #     return ret