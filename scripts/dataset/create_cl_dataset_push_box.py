from varyingsim.envs.push_box_circle import PushBoxCircle, sin_com_slow
from varyingsim.envs.push_box_offset import PushBoxOffset
from varyingsim.datasets.cl_dataset import CLDatasetGenerator
import numpy as np
import os
import shutil

# TODO: this could be done in parallel

seed = 0

# env = PushBoxCircle()
env = PushBoxOffset()
K = 10
N = 500
T = 2000

base_location = '/data/varyingsim/datasets/push_box_action_large_{}/'.format(seed)

try:
    os.mkdir(base_location)
except Exception as e:
    print('dataset with that name exists! delete it to regenerate')
    exit()

gen_name = os.path.basename(__file__)
shutil.copy(__file__, os.path.join(base_location, gen_name)) 

def act_fn(obs, i, t, memory):
    if i == 0:
        memory['velocity'] = 0.5
        memory['angle'] = 0.0
    elif t == 0:
        memory['velocity'] = 0.1 + np.random.random() * 0.9
        memory['angle'] = np.random.random() * 2 * np.pi
    return np.array([memory['velocity'], memory['angle'], 0.0])

com_limit = 0.14
d_com = 0.01

def sample_init_fn():
    # sample uniformly at random from [-0.14, -0.13, ... , 0.13, 0.14]
    n = int(2 * com_limit / d_com + 1)
    choices = (np.arange(n) - n // 2) * d_com
    init_offset = np.random.choice(choices)
    return [init_offset, None, None, None, None]

def sample_next_fn(fovs, n):
    offset = fovs[0]
    if np.random.random() > 0.5:
        offset += d_com
    else:
        offset -= d_com
    
    np.clip(offset, -com_limit, com_limit)
    return [offset, None, None, None, None]


gen = CLDatasetGenerator(base_location, PushBoxOffset, sample_init_fn, 
    sample_next_fn, act_fn, K, N, T, seed=seed)
