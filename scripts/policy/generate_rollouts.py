import gym
from mjrl.utils.gym_env import GymEnv
from mjrl.samplers.core import sample_paths
import pickle
import argparse
import numpy as np
import varyingsim.envs
import ctypes
import os
from stable_baselines3 import SAC

def is_registered(id):
    return id in [a.id for a in gym.envs.registry.all()]

SWIMMER = 'swimmer'
HOPPER = 'hopper'
HUMANOID = 'humanoid'

ENVS = [SWIMMER, HOPPER, HUMANOID]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=ENVS)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output-dir', default='data')

    args = parser.parse_args()
    seed = args.seed

    custom_base_seed = ctypes.c_uint32(hash(str(seed))).value
    
    T = 1000

    if args.env == SWIMMER:
        ENV_NAME = 'VaryingSwimmer-v0'
        env = 'training_swimmer'
        prefix = 'training_swimmer'
        NUM_TRAJS = 100
    elif args.env == HOPPER:
        ENV_NAME = 'VaryingHopper-v0'
        env = 'training_hopper'
        prefix = 'training_hopper'
        NUM_TRAJS = 100
    elif args.env == HUMANOID:
        ENV_NAME = 'Humanoid-v2'
        env = 'training_humanoid'
        prefix = 'training_humanoid'        
        NUM_TRAJS = 50
    else:
        raise Exception("Must select env from {}".format(ENVS))

    output_suffix = 'seed_{}'.format(seed)

    if args.env == HUMANOID:
        file = "data/humanoid_policy_5.7M"
        model = SAC.load(file)

        class policy:
            def get_action(obs):
                return model.predict(obs)[0], {}
    else:
        with open('{}/iterations/best_policy.pickle'.format(prefix), 'rb') as f:
            policy = pickle.load(f)

    env = GymEnv(ENV_NAME)

    paths = sample_paths(NUM_TRAJS, env, policy, base_seed=custom_base_seed)

    with open(os.path.join(args.output_dir, '{}_paths_{}.pickle'.format(prefix, output_suffix)), 'wb') as f:
        pickle.dump(paths, f)
