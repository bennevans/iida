from varyingsim.datasets.fov_dataset import SmoothFovDatasetGenerator, SmoothFovDataset
import os
import numpy as np

class CLDatasetGenerator():
    def __init__(self, location_base, dataset_type, sample_init_fn, \
                        sample_next_fn, act_fn, K, N, T, seed=0):
        self.location_base = location_base
        self.name = 'dataset' # name will be prepended to _i where i is the walk number
        self.dataset_type = dataset_type
        self.sample_init_fn = sample_init_fn
        self.sample_next_fn = sample_next_fn
        self.act_fn = act_fn
        self.K = K # number of different environment walks
        self.N = N # number of episodes per environment
        self.T = T # length of each episode

        self.seed = seed

        self.generateDataset()
    
    def generate_fovs(self):
        param = self.sample_init_fn()
        params = []
        for n in range(self.N):
            params.append(param)
            param = self.sample_next_fn(param, n) # can be a function of just episode number if we want
        
        return params

    def generate_kth(self, k):
        np.random.seed(self.seed + k)
        location = os.path.join(self.location_base, "{}_{}.pickle".format(self.name, k))

        params = self.generate_fovs()
        def set_fov(env, i, t, memory):
            env.set_fovs(params[i])

        gen = SmoothFovDatasetGenerator(location, self.dataset_type, set_fov,
            self.act_fn, self.N, self.T)

    def generateDataset(self):
        for k in range(self.K):
            self.generate_kth(k)

            