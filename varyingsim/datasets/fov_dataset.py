
import numpy as np
import os
import pickle
import inspect
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from varyingsim.util.trajectory import get_traj_pol, get_traj_pol_contact
import multiprocessing
from varyingsim.util.system import get_true_cores
from mujoco_py.builder import MujocoException

# TODO: include trajectory length

class SmoothFovDatasetGenerator():
    def __init__(self, location, dataset_type, set_fov, act_fn,
                N, test_traj_len, other_info={}):
        self.location = location
        self.dataset_type = dataset_type
        self.set_fov = set_fov
        self.act_fn = act_fn
        self.N = N
        self.test_traj_len = test_traj_len
        self.other_info = other_info

        self.obs = []
        self.act = []
        self.fov = []
        self.rew = []
        self.is_start = []
        self.episodes = []

        self.generateDataset()

    # TODO: this is specific to our dataset and model


    def generateDataset(self):
        env = self.dataset_type(include_fov=False)
        i = 0
        n_traj = 0
        with tqdm(total=self.N)as pbar:
            while n_traj < self.N:
            # for i in range(self.N):

                ret = None
                while ret is None:
                    ret = get_traj_pol(env, i, self.act_fn, self.set_fov, self.test_traj_len)
                    # ret = get_traj_pol_contact(env, i, self.act_fn, self.set_fov, self.test_traj_len)
                    i += 1
                n_traj += 1
                obs, act, fov, rew = ret

                self.obs.append(obs)
                self.act.append(act)
                self.fov.append(fov)
                self.rew.append(rew)
                self.episodes.append(i)
            
                is_start = np.zeros(len(obs))
                is_start[0] = 1.
                self.is_start.append(is_start)
                pbar.update()

        act_fn_str = inspect.getsource(self.act_fn)
        set_fov_str = inspect.getsource(self.set_fov)
        dataset_str = inspect.getsource(self.dataset_type)

        data = dict(
            obs=self.obs,
            act=self.act,
            fov=self.fov,
            rew=self.rew,
            is_start=self.is_start,
            episodes=self.episodes,
            N=self.N,
            test_traj_len=self.test_traj_len,
            dataset_str=dataset_str,
            act_fn_str=act_fn_str,
            set_fov_str=set_fov_str,
        )

        data.update(self.other_info)

        with open(self.location, 'wb') as f:
            pickle.dump(data, f)

class SmoothEpisodicFovDatasetGenerator():
    def __init__(self, location, dataset_type, set_fov, act_fn,
                R, K, test_traj_len, other_info={}):
        self.location = location
        self.dataset_type = dataset_type
        self.set_fov = set_fov
        self.act_fn = act_fn
        self.R = R
        self.K = K
        self.test_traj_len = test_traj_len
        self.other_info = other_info

        self.state = []
        self.act = []
        self.fov = []
        self.rew = []
        self.is_start = []
        self.episodes = []

        self.generateDataset()

    def generateDataset(self):
        env = self.dataset_type(include_fov=False)
        i = 0
        with tqdm(total=self.K * self.R)as pbar:
            for k in range(self.K):
                n_traj = 0
                states = []
                acts = []
                fovs = []
                rews = []
                episodes = []
                is_starts = []

                while n_traj < self.R:
                    ret = None
                    while ret is None:
                        ret = get_traj_pol(env, i, self.act_fn, self.set_fov, self.test_traj_len)
                        # ret = get_traj_pol_contact(env, i, self.act_fn, self.set_fov, self.test_traj_len)
                        i += 1
                    n_traj += 1
                    state, act, fov, rew = ret

                    states.append(state)
                    acts.append(act)
                    fovs.append(fov)
                    rews.append(rew)
                    episodes.append(i)
                
                    is_start = np.zeros(len(state))
                    is_start[0] = 1.
                    is_starts.append(is_start)
                    pbar.update()

                self.state.append(states)
                self.act.append(acts)
                self.fov.append(fovs)
                self.rew.append(rews)
                self.is_start.append(is_starts)
                self.episodes.append(episodes)

        act_fn_str = inspect.getsource(self.act_fn)
        set_fov_str = inspect.getsource(self.set_fov)
        dataset_str = inspect.getsource(self.dataset_type)

        data = dict(
            state=self.state,
            act=self.act,
            fov=self.fov,
            rew=self.rew,
            is_start=self.is_start,
            episodes=self.episodes,
            R=self.R,
            K=self.K,
            test_traj_len=self.test_traj_len,
            dataset_str=dataset_str,
            act_fn_str=act_fn_str,
            set_fov_str=set_fov_str,
        )

        data.update(self.other_info)

        with open(self.location, 'wb') as f:
            pickle.dump(data, f)

class ParallelEpisodicFovDatasetGenerator():
    def __init__(self, location, dataset_type, set_fov, act_fn,
                R, K, test_traj_len, other_info={}, base_seed=None, timeout=32.0):
        self.location = location
        self.dataset_type = dataset_type
        self.set_fov = set_fov
        self.act_fn = act_fn
        self.R = R
        self.K = K
        self.test_traj_len = test_traj_len
        self.other_info = other_info
        self.base_seed = base_seed
        self.timeout = timeout

        self.state = []
        self.act = []
        self.fov = []
        self.rew = []
        self.is_start = []
        self.episodes = []

        self.generateDataset()

    def generateEpisodes(self, idx, start_i, episodes_per_proc, seed=None):

        # env = self.dataset_type(include_fov=False)

        state = []
        act = []
        fov = []
        rew = []
        is_start = []
        episodes = []

        i = start_i

        with tqdm(total=episodes_per_proc * self.R, position=idx, desc=str(os.getpid())) as pbar:
            for j in range(episodes_per_proc):
                result = self.generatePart(i, self.R, seed=seed+j, pbar=pbar)
                state.append(result['state'])
                act.append(result['act'])
                fov.append(result['fov'])
                rew.append(result['rew'])
                is_start.append(result['is_start'])
                episodes.append(result['episode'])

                i += self.R
                pbar.update(1)

        return dict(state=state, act=act, fov=fov, rew=rew, is_start=is_start, episodes=episodes)

    def generatePart(self, start_i, N, seed=None, pbar=None):
        env = self.dataset_type(include_fov=False)
        if seed:
            np.random.seed(seed)

        n_traj = 0
        states = []
        acts = []
        fovs = []
        rews = []
        episodes = []
        is_starts = []

        i = start_i

        while n_traj < N:
            ret = None
            while ret is None:
                ret = get_traj_pol(env, i, self.act_fn, self.set_fov, self.test_traj_len)
                # ret = get_traj_pol_contact(env, i, self.act_fn, self.set_fov, self.test_traj_len)
                i += 1
            n_traj += 1
            state, act, fov, rew = ret

            states.append(state)
            acts.append(act)
            fovs.append(fov)
            rews.append(rew)
            episodes.append(i)
        
            is_start = np.zeros(len(state))
            is_start[0] = 1.
            is_starts.append(is_start)

            if pbar:
                pbar.update(1)
        
        ret = dict(state=states, act=acts, fov=fovs, rew=rews, episode=episodes, is_start=is_starts)
        return ret

    def generateDataset(self):
        i = 0

        num_proc = multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(num_proc)

        # construct arguments

        episodes_per_proc = (self.K // num_proc) + 1
        args = [] # start_is, Ns, and seeds
        start_i = 0

        num_proc = min(num_proc, self.K)

        for p in range(num_proc):
            seed = start_i + self.base_seed if self.base_seed else start_i
            args.append((p, start_i, episodes_per_proc, seed))
            start_i += episodes_per_proc * self.R

        runs = [pool.apply_async(self.generateEpisodes, args=args[i]) for i in range(num_proc)]


        results = [p.get() for p in runs]

        for result in results:
            self.state += result['state']
            self.act += result['act']
            self.fov += result['fov']
            self.rew += result['rew']
            self.is_start += result['is_start']
            self.episodes += result['episodes']

        act_fn_str = inspect.getsource(self.act_fn)
        set_fov_str = inspect.getsource(self.set_fov)
        dataset_str = inspect.getsource(self.dataset_type)

        data = dict(
            state=self.state,
            act=self.act,
            fov=self.fov,
            rew=self.rew,
            is_start=self.is_start,
            episodes=self.episodes,
            R=self.R,
            K=self.K,
            test_traj_len=self.test_traj_len,
            dataset_str=dataset_str,
            act_fn_str=act_fn_str,
            set_fov_str=set_fov_str,
        )

        data.update(self.other_info)

        with open(self.location, 'wb') as f:
            pickle.dump(data, f)

class ParallelEpisodicStartEndFovDatasetGenerator():
    def __init__(self, location, dataset_constructor, set_fov, act_fn,
                R, K, test_traj_len, other_info={}, base_seed=None, n_cpu=None, start_fn=None, end_fn=None):
        self.location = location
        self.dataset_constructor = dataset_constructor
        self.set_fov = set_fov
        self.act_fn = act_fn
        self.R = R
        self.K = K
        self.test_traj_len = test_traj_len
        self.other_info = other_info
        self.base_seed = base_seed
        self.num_proc = get_true_cores() if n_cpu is None else n_cpu

        self.state = []
        self.act = []
        self.state_prime = []
        self.fov = []
        self.rew = []
        self.is_start = []
        self.episodes = []
        self.end_fn = end_fn
        self.start_fn = start_fn

        self.generateDataset()

    def generateEpisodes(self, idx, start_i, episodes_per_proc, seed=None):

        env = self.dataset_constructor()

        state = []
        act = []
        state_prime = []
        fov = []
        rew = []
        episodes = []
        infos = []

        i = start_i

        with tqdm(total=episodes_per_proc * self.R, position=idx, desc=str(os.getpid())) as pbar:
            for j in range(episodes_per_proc):
                
                result = self.generatePart(env, i, self.R, seed=seed+j, pbar=pbar)
                state.append(result['state'])
                act.append(result['act'])
                state_prime.append(result['state_prime'])
                fov.append(result['fov'])
                rew.append(result['rew'])
                episodes.append(result['episode'])
                infos.append(result['infos'])

                i += self.R
                pbar.update(1)

        return dict(state=state, act=act, state_prime=state_prime, fov=fov, rew=rew, episodes=episodes, infos=infos)

    def generatePart(self, env, start_i, N, seed=None, pbar=None):
        env = self.dataset_constructor() # this is for generating the same randomness for initial state
        env.episode = start_i
        # print('start_i', start_i)
        
        if seed:
            np.random.seed(seed)

        n_traj = 0
        states = []
        acts = []
        state_primes = []
        fovs = []
        rews = []
        episodes = []
        is_starts = []
        infos = []

        i = start_i
        n_exceptions = 0

        while n_traj < N:
            ret = None
            while ret is None:
                try:
                    env.reset_idx = i % self.R

                    ret = get_traj_pol(env, i, self.act_fn, self.set_fov, self.test_traj_len, ret_info=True)
                # ret = get_traj_pol_contact(env, i, self.act_fn, self.set_fov, self.test_traj_len)
                    i += 1
                except MujocoException as e:
                    n_exceptions += 1
                    pbar.write("n exceptions: {}".format(n_exceptions))
            n_traj += 1
            state, act, fov, rew, info = ret
            if self.start_fn is None:
                start_idx = 0
            else:
                start_idx = self.start_fn(state)

            start_state = np.copy(state[start_idx])

            if self.end_fn is None:
                end_idx = np.where(state[:, 17])[0][0]
            else:
                end_idx = self.end_fn(state, start_search=start_idx)
            
            end_state = np.copy(state[end_idx])

            if self.num_proc == 1:
                print('start_state', state[0])
                print('end_state', state[-1])
                print('fov', fov[0])
                print('end_idx', end_idx)

            states.append(start_state)
            acts.append(act[0])
            state_primes.append(end_state)
            fovs.append(fov[0])
            rews.append(rew[0])
            episodes.append(i)
            infos.append(info)

            if pbar:
                pbar.update(1)
        
        ret = dict(state=states, act=acts, state_prime=state_primes, fov=fovs, rew=rews, episode=episodes, infos=infos)
        return ret
    def generateDataset(self):
        if self.num_proc > 1:
            self.generateDatasetMP()
        else:
            self.generateDatasetSP()

    def generateDatasetSP(self):
        run = self.generateEpisodes(0, 0, self.K, seed=self.base_seed)

        self.state = run['state']
        self.act = run['act']
        self.fov = run['fov']
        self.state_prime = run['state_prime']
        self.rew = run['rew']
        self.episodes = run['episodes']

        act_fn_str = inspect.getsource(self.act_fn)
        set_fov_str = inspect.getsource(self.set_fov)
        dataset_str = inspect.getsource(self.dataset_constructor)

        data = dict(
            state=self.state,
            act=self.act,
            fov=self.fov,
            state_prime=self.state_prime,
            rew=self.rew,
            episodes=self.episodes,
            R=self.R,
            K=self.K,
            test_traj_len=self.test_traj_len,
            dataset_str=dataset_str,
            act_fn_str=act_fn_str,
            set_fov_str=set_fov_str,
            infos=run['infos']
        )

        data.update(self.other_info)

        with open(self.location, 'wb') as f:
            pickle.dump(data, f)

    def generateDatasetMP(self):
        i = 0

        pool = multiprocessing.Pool(self.num_proc)

        # construct arguments

        episodes_per_proc = (self.K // self.num_proc) + 1
        args = [] # start_is, Ns, and seeds
        start_i = 0

        num_proc = min(self.num_proc, self.K)

        for p in range(num_proc):
            seed = start_i + self.base_seed if self.base_seed else start_i
            args.append((p, start_i, episodes_per_proc, seed))
            start_i += episodes_per_proc * self.R
        # print('start_is')
        # print(list(arg[1] for arg in args))
        runs = [pool.apply_async(self.generateEpisodes, args=args[i]) for i in range(num_proc)]

        results = [p.get() for p in runs]
        
        infos = []
        for result in results:
            self.state += result['state']
            self.act += result['act']
            self.state_prime += result['state_prime']
            self.fov += result['fov']
            self.rew += result['rew']
            self.episodes += result['episodes']
            infos += result['infos']

        act_fn_str = inspect.getsource(self.act_fn)
        set_fov_str = inspect.getsource(self.set_fov)
        dataset_str = inspect.getsource(self.dataset_constructor)

        data = dict(
            state=self.state,
            act=self.act,
            fov=self.fov,
            state_prime=self.state_prime,
            rew=self.rew,
            episodes=self.episodes,
            R=self.R,
            K=self.K,
            test_traj_len=self.test_traj_len,
            dataset_str=dataset_str,
            act_fn_str=act_fn_str,
            set_fov_str=set_fov_str,
            infos=infos
        )

        data.update(self.other_info)

        with open(self.location, 'wb') as f:
            pickle.dump(data, f)

class SmoothFovDataset(Dataset):
    def __init__(self, location, H, obs_skip=1, include_full=False, pad_till=-1):
        self.location = location
        self.H = H
        self.obs_skip = obs_skip
        self.include_full = include_full
        self.pad_till = pad_till

        with open(location, 'rb') as f:
            data = pickle.load(f)
        
        self.obs = data['obs']
        self.act = data['act']
        self.fov = data['fov']
        self.is_start = data['is_start']
        self.rew = data['rew'] if 'rew' in data.keys() else [[0] * len(l) for l in self.obs]
        self.episodes = data['episodes'] if 'episodes' in data.keys() else [0 for _ in self.obs]

        lens = []
        for obs in self.obs:
            lens.append(len(obs) // self.obs_skip)
        
        prefix_lens = [0] * (len(lens) + 1)
        for i, l in enumerate(lens):
            prefix_lens[i + 1] = prefix_lens[i] + l - 1

        self.prefix_lens = np.array(prefix_lens)
        self.N = prefix_lens[-1]
        self.data = data

    def __len__(self):
        return self.N

    def pad(self, context):
        shape = np.array(context.shape)
        shape[-2] = self.H - shape[-2] 
        return np.concatenate([np.zeros(shape), context], axis=-2)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("idx must be less than length!")

        arr_idx = np.argmax(self.prefix_lens > idx) - 1
        arr_start_idx = (idx - self.prefix_lens[arr_idx]) * self.obs_skip

        context_start_idx = max(arr_start_idx - self.H * self.obs_skip, 0)

        context_obs = self.obs[arr_idx][context_start_idx:arr_start_idx:self.obs_skip]
        context_act = self.act[arr_idx][context_start_idx:arr_start_idx:self.obs_skip]

        if context_obs.shape[-2] < self.H:
            context_obs = self.pad(context_obs)
            context_act = self.pad(context_act)

        obs = self.obs[arr_idx][arr_start_idx]
        act = self.act[arr_idx][arr_start_idx]
        fov = self.fov[arr_idx][arr_start_idx]
        rew = self.rew[arr_idx][arr_start_idx]
        obs_prime = self.obs[arr_idx][arr_start_idx + self.obs_skip]
        is_start = self.is_start[arr_idx][arr_start_idx]
        episode = self.episodes[arr_idx]

        ret = dict(
            context_obs=context_obs.astype(np.float32),
            context_act=context_act.astype(np.float32),
            obs=obs.astype(np.float32),
            act=act.astype(np.float32),
            fov=fov.astype(np.float32),
            rew=rew,
            episode=episode,
            obs_prime=obs_prime.astype(np.float32), 
            is_start=is_start,
        )

        if self.include_full:
            T, d_obs = self.obs[arr_idx].shape
            d_act = self.act[arr_idx].shape[1]
            d_fov = self.fov[arr_idx].shape[1]
            if self.pad_till > 0 and T < self.pad_till:
                ret['obs_full'] = np.zeros((self.pad_till, d_obs))
                ret['obs_full'][:T] = self.obs[arr_idx].astype(np.float32)
                ret['act_full'] = np.zeros((self.pad_till, d_act))
                ret['act_full'][:T] = self.act[arr_idx].astype(np.float32)
                ret['fov_full'] = np.zeros((self.pad_till, d_fov))
                ret['fov_full'][:T] = self.fov[arr_idx].astype(np.float32)
                ret['T'] = T
            else:
                ret['obs_full'] = self.obs[arr_idx].astype(np.float32)
                ret['act_full'] = self.act[arr_idx].astype(np.float32)
                ret['fov_full'] = self.fov[arr_idx].astype(np.float32)

            ret['start_idx'] = arr_start_idx

        return ret

class SmoothEpisodicFovDataset(Dataset):
    def __init__(self, location, H, obs_skip=1, include_full=False, pad_till=-1,obs_fn=None):
        self.location = location
        self.H = H
        self.obs_skip = obs_skip
        self.include_full = include_full
        self.pad_till = pad_till
        self.obs_fn = obs_fn

        with open(location, 'rb') as f:
            data = pickle.load(f)
        
        self.state = []
        self.act = []
        self.fov = []

        if self.obs_fn:
            self.obs = []

        for episode_state, episode_act, episode_fov in zip(data['state'], data['act'], data['fov']):
            for state, act, fov in zip(episode_state, episode_act, episode_fov):
                self.state.append(state)
                self.act.append(act)
                self.fov.append(fov)

        lens = []
        for state in self.state:
            lens.append(len(state) // self.obs_skip)
        
        prefix_lens = [0] * (len(lens) + 1)
        for i, l in enumerate(lens):
            prefix_lens[i + 1] = prefix_lens[i] + l - 1

        self.prefix_lens = np.array(prefix_lens)
        self.N = prefix_lens[-1]
        self.data = data

    def __len__(self):
        return self.N

    def pad(self, context):
        shape = np.array(context.shape)
        shape[-2] = self.H - shape[-2] 
        return np.concatenate([np.zeros(shape), context], axis=-2)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("idx must be less than length!")

        arr_idx = np.argmax(self.prefix_lens > idx) - 1
        arr_start_idx = (idx - self.prefix_lens[arr_idx]) * self.obs_skip

        context_start_idx = max(arr_start_idx - self.H * self.obs_skip, 0)

        context_state = self.state[arr_idx][context_start_idx:arr_start_idx:self.obs_skip]
        context_act = self.act[arr_idx][context_start_idx:arr_start_idx:self.obs_skip]

        if context_state.shape[-2] < self.H:
            context_state = self.pad(context_state)
            context_act = self.pad(context_act)

        state = self.state[arr_idx][arr_start_idx]
        act = self.act[arr_idx][arr_start_idx]
        fov = self.fov[arr_idx][arr_start_idx]
        state_prime = self.state[arr_idx][arr_start_idx + self.obs_skip]
   
        ret = dict(
            state=state.astype(np.float32),
            act=act.astype(np.float32),
            fov=fov.astype(np.float32),
            state_prime=state_prime.astype(np.float32), 
        )

        if self.include_full:
            T, d_state = self.state[arr_idx].shape
            d_act = self.act[arr_idx].shape[1]
            d_fov = self.fov[arr_idx].shape[1]
            if self.pad_till > 0 and T < self.pad_till:
                ret['state_full'] = np.zeros((self.pad_till, d_state))
                ret['state_full'][:T] = self.state[arr_idx].astype(np.float32)
                ret['act_full'] = np.zeros((self.pad_till, d_act))
                ret['act_full'][:T] = self.act[arr_idx].astype(np.float32)
                ret['fov_full'] = np.zeros((self.pad_till, d_fov))
                ret['fov_full'][:T] = self.fov[arr_idx].astype(np.float32)
                ret['T'] = T
            else:
                ret['state_full'] = self.state[arr_idx].astype(np.float32)
                ret['act_full'] = self.act[arr_idx].astype(np.float32)
                ret['fov_full'] = self.fov[arr_idx].astype(np.float32)

            ret['start_idx'] = arr_start_idx

        return ret

class EpisodicStartEndFovDataDataset(Dataset):
    def __init__(self, data, obs_fn=None):
        self.obs_fn = obs_fn
        
        self.state = data['state']
        self.act = data['act']
        self.fov = data['fov']
        self.state_prime = data['state_prime']

        self.N = len(self.state)
        self.data = data

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("idx must be less than length!")

        state = np.stack(self.state[idx])
        act = np.stack(self.act[idx])
        fov = np.stack(self.fov[idx])
        state_prime = np.stack(self.state_prime[idx])
        
        ret = dict(
            state=state.astype(np.float32),
            act=act.astype(np.float32),
            fov=fov.astype(np.float32),
            state_prime=state_prime.astype(np.float32),
        )

        if self.obs_fn:
            ret['obs'] = self.obs_fn(state).astype(np.float32)
            ret['obs_prime'] = self.obs_fn(state_prime).astype(np.float32)
        else:
            ret['obs'] = state.astype(np.float32)
            ret['obs_prime'] = state_prime.astype(np.float32)

        return ret

class EpisodicStartEndFovDataset(EpisodicStartEndFovDataDataset):
    def __init__(self, location, obs_fn=None):
        with open(location, 'rb') as f:
            data = pickle.load(f)
        super().__init__(data, obs_fn=obs_fn)

class StartEndDataset(Dataset):
    def __init__(self, location, H, include_full=True):
        """ 
            A way of viewing the dataset that just predicts the end location, not the
            dt-level positions. H is the number of steps of context given
        """
        self.location = location
        self.H = H
        self.include_full = include_full

        with open(location, 'rb') as f:
            data = pickle.load(f)
        
        self.obs = data['obs']
        self.act = data['act']
        self.fov = data['fov']
        self.rew = data['rew']
        self.is_start = data['is_start']
        self.episodes = data['episodes']

        dels = []
        for i, obs in enumerate(self.obs):
            if len(obs) < self.H:
                # traj too short, skip
                dels.append(i)
        
        for i in reversed(dels):
            del self.obs[i]
            del self.act[i]
            del self.fov[i]
            del self.rew[i]
            del self.is_start[i]
            del self.episodes[i]

        
        self.dataset_type = data['dataset_type']

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("idx must be less than length!")

        context_obs = self.obs[idx][:self.H]
        context_act = self.act[idx][:self.H]
        obs = self.obs[idx][self.H]
        act = self.act[idx][self.H]
        fov = self.fov[idx][self.H]
        rew = self.rew[idx][self.H]
        obs_prime = self.obs[idx][-1]
        is_start = self.is_start[idx][0]
        episode = self.episodes[idx]

        ret = dict(
            context_obs=context_obs,
            context_act=context_act,
            obs=obs,
            act=act,
            fov=fov,
            rew=rew,
            episode=episode,
            obs_prime=obs_prime, 
            is_start=is_start,
        ) 
        if self.include_full:
            ret['obs_full']=self.obs[idx]
            ret['act_full']=self.act[idx]
            ret['fov_full']=self.fov[idx]
            
        return ret

class EpisodicStartEndDataset(Dataset):
    def __init__(self, location, obs_fn=None, include_full=True):
        """ 
            A way of viewing the dataset that just predicts the end location, not the
            dt-level positions.
        """
        self.location = location
        self.include_full = include_full
        self.obs_fn = obs_fn
        # TODO: act_fn

        with open(location, 'rb') as f:
            data = pickle.load(f)
        
        self.state = data['state']
        self.act = data['act']
        self.fov = data['fov']
        self.rew = data['rew']
        self.is_start = data['is_start']
        self.episodes = data['episodes']

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("idx must be less than length!")

        episode_state = self.state[idx]
        episode_act = self.act[idx]
        episode_fov = self.fov[idx]
        episode_rew = self.rew[idx]
        episode_is_start = self.is_start[idx][0]
        episode_episode = self.episodes[idx]


        states = []
        acts = []
        fovs = []
        rews = []
        state_primes = []
        is_starts = []
        episodes = []

        if self.obs_fn:
            obs_primes = []
            obss = []

        for i in range(len(episode_state)):
            start_state = episode_state[i][0]
            act = episode_act[i][0]
            end_idx = np.where(episode_state[i][:, 17])[0][0]
            end_state = episode_state[i][end_idx]

            states.append(start_state)
            acts.append(act)
            state_primes.append(end_state)
            episodes.append(episode_episode[i])

            fovs.append(episode_fov[i][0])

            if self.obs_fn:
                obss.append(self.obs_fn(start_state))
                obs_primes.append(self.obs_fn(end_state))

        ret = dict(
            state=np.stack(states).astype(np.float32),
            act=np.stack(acts).astype(np.float32),
            fov=np.stack(fovs).astype(np.float32),
            episode=np.stack(episodes),
            state_prime=np.stack(state_primes).astype(np.float32), 
        )

        if self.obs_fn:
            ret['obs'] = np.stack(obss).astype(np.float32)
            ret['obs_prime'] = np.stack(obs_primes).astype(np.float32)

        # This might be useful later in this case, but won't work now
        if self.include_full:
            ret['state_full']=self.state[idx]
            ret['act_full']=self.act[idx]
            ret['fov_full']=self.fov[idx]
            
        return ret

def split_four_way(arr, K, R, K_train, R_train):
    arr1 = np.stack(arr[:K_train])
    arr2 = np.stack(arr[K_train:K])

    train_arr = arr1[:, :R_train]
    test_state = arr1[:, R_train:R]
    test_fov = arr2[:, :R_train]
    test_fov_state = arr2[:, R_train:R]

    return train_arr, test_state, test_fov, test_fov_state

def split_episodic(location, K_train, R_train):
    # takes a location and returns 4 data objects 
    # splitting by R_train and K_train. Some datasets have extra data, but we won't include it 
    with open(location, 'rb') as f:
        data = pickle.load(f)

    R = data['R']
    K = data['K']

    train_state, test_state_state, test_fov_state, test_fov_state_state = split_four_way(data['state'], K, R, K_train, R_train)
    train_act, test_state_act, test_fov_act, test_fov_state_act = split_four_way(data['act'], K, R, K_train, R_train)
    train_fov, test_state_fov, test_fov_fov, test_fov_state_fov = split_four_way(data['fov'], K, R, K_train, R_train)
    train_state_prime, test_state_state_prime, test_fov_state_prime, test_fov_state_state_prime = split_four_way(data['state_prime'], K, R, K_train, R_train)

    train_data = dict(state=train_state, act=train_act, fov=train_fov, state_prime=train_state_prime)
    test_state_data = dict(state=test_state_state, act=test_state_act, fov=test_state_fov, state_prime=test_state_state_prime)
    test_fov_data = dict(state=test_fov_state, act=test_fov_act, fov=test_fov_fov, state_prime=test_fov_state_prime)
    test_fov_state_data = dict(state=test_fov_state_state, act=test_fov_state_act, fov=test_fov_state_fov, state_prime=test_fov_state_state_prime)

    # train_data = dict(state=train_state, act=train_act, fov=train_fov, state_prime=train_state_prime)
    # test_state_data = dict(state=test_fov_state, act=test_fov_act, fov=test_fov_fov, state_prime=test_state_state_prime)
    # test_fov_data = dict(state=test_fov_state_state, act=test_fov_state_act, fov=test_fov_state_fov, state_prime=test_fov_state_prime)
    # test_fov_state_data = dict(state=test_fov_state_state, act=test_fov_state_act, fov=test_fov_state_fov, state_prime=test_fov_state_state_prime)

    return train_data, test_state_data, test_fov_data, test_fov_state_data
    