from varyingsim.envs.push_box_circle import PushBoxCircle, sin_com_slow
from varyingsim.envs.cartpole import CartpoleEnv
from varyingsim.datasets.fov_dataset import SmoothFovDataset, SmoothFovDatasetGenerator
import numpy as np
from varyingsim.policies.mppi import MPPIEnvPolicy

# TODO: this could be done in parallel

seed = 1
np.random.seed(seed)

# env = PushBoxCircle()
env = CartpoleEnv()
N = 100
T = 250

# location = '/misc/vlgscratch5/PintoGroup/bne215/datasets/varyingsim/push_box_circle.pickle'
location = '/data/varyingsim/cartpole_smooth_actual_slow_large_{}.pickle'.format(seed)
# location = 'D:\\data\\varyingsim\\push_box_contact_simple_tiny_{}.pickle'.format(seed)


def set_fov(env, i, t):
    mass = np.sin(i / (N / 2.3732) * 2 *np.pi )  + 1.1
    env.set_end_mass(mass)

def construct_fn(include_fov=False):
    return CartpoleEnv(mode=CartpoleEnv.SWINGUP, set_param_fn=set_fov, T=T, include_fov=include_fov)
    
env = construct_fn()
K = 32
H = 128
sigma = np.eye(env.model.nu) * 1.0
temp = 0.3

policy = MPPIEnvPolicy(construct_fn, K, H, sigma, set_fov, temp=temp)

def act_fn(obs, i, t):
    return policy.get_action(obs, t, i=i)

gen = SmoothFovDatasetGenerator(location, construct_fn, set_fov, act_fn, N, T)

