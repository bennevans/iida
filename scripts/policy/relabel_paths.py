import gym
import pickle
import yaml
from varyingsim.util.dataset import get_ranges, sample_ranges
from varyingsim.util.system import hash_config
import numpy as np
from tqdm import tqdm

import varyingsim.envs

import argparse
import random
import os

"""
copies each trajectory and samples new s' for each (s,a,s') given enviornment parameters
if there are 100 train paths and num_train_episodes is 10, then there will be 1000 resultant paths
"""

def gen_sample_train(fov_ranges):
    def sample_train():
        fov_values = {}
        for fov_name, info in fov_ranges.items():
            ranges = info['train_ranges']
            fov = sample_ranges(ranges, 1)
            fov_values[fov_name] = fov
        return fov_values
    return sample_train

def gen_sample_other(fov_ranges, p, other='test'):
    assert other in ['test', 'val']
    def sample_test():
        fov_values = {}
        for fov_name, info in fov_ranges.items():
            if random.random() > p: # sample from the train domain
                fov_values[fov_name] = sample_ranges(info['train_ranges'])
            else: # sample from test domain
                fov_values[fov_name] = sample_ranges(info['{}_ranges'.format(other)])
        return fov_values
    return sample_test

def set_fovs(env, fovs):
    for fov_name, fov in fovs.items():
        env.set_fov(fov_name, fov)

def modify_path(path, fovs):
    T, d = path['observations'].shape
    env = gym.make(ENV_NAME)
    env.reset()
    set_fovs(env, fovs)
    env.env.sim.set_constants()
    states = []
    actions = []
    state_primes = []
    for t in range(T):
        state = np.copy(path['states'][t])
        action = np.copy(path['actions'][t])

        qpos = state[:env.model.nq]
        qvel = state[env.model.nq:]
        env.set_state(qpos=qpos, qvel=qvel)
        obs_prime, rew, done, info = env.step(action)

        states.append(state)
        actions.append(action)
        state_prime_full = env.env.sim.get_state()
        state_prime = np.concatenate([state_prime_full.qpos, state_prime_full.qvel])
        state_primes.append(state_prime)
    
    return states, actions, state_primes

def flatten_fovs(env, fovs):
    fov_names = env.get_fov_names()
    fov_arr = np.zeros(len(fov_names))

    for idx, fov_name in enumerate(fov_names):
        fov_arr[idx] = fovs[fov_name] if fov_name in fovs.keys() else env.get_fov(fov_name)

    return fov_arr

def modify_paths(paths, sample_fovs):
    env = gym.make(ENV_NAME)
    all_states, all_actions, all_state_primes, all_fovs = [], [], [], []
    for k in tqdm(range(K)):
        fov = sample_fovs()
        states, actions, state_primes, fovs = [], [], [], []
        for i, path in enumerate(paths):
            state, action, state_prime = modify_path(path, fov)
            states.append(state)
            actions.append(action)
            state_primes.append(state_prime)
            fov_flat = flatten_fovs(env, fov)
            fov_rep = [fov_flat] * len(state_prime)
            fovs.append(fov_rep)

        all_states.append(states)
        all_actions.append(actions)
        all_state_primes.append(state_primes)
        all_fovs.append(fovs)
    
    return all_states, all_actions, all_state_primes, all_fovs


SWIMMER = 'swimmer'
HOPPER = 'hopper'
HUMANOID = 'humanoid'

ENVS = [SWIMMER, HOPPER, HUMANOID]

OUTPUT_DIR = 'data'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=ENVS)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()

    T = 1000
    seed = args.seed

    if args.env == SWIMMER:
        ENV_NAME = 'VaryingSwimmer-v0'
        env = 'training_swimmer'
        prefix = 'training_swimmer'
    elif args.env == HOPPER:
        ENV_NAME = 'VaryingHopper-v0'
        env = 'training_hopper'
        prefix = 'training_hopper'
    elif args.env == HUMANOID:
        ENV_NAME = 'VaryingHumanoid-v0'
        env = 'training_humanoid'
        prefix = 'training_humanoid'
    else:
        raise Exception("Must select env from {}".format(ENVS))
        
    with open('data/{}_paths_seed_{}.pickle'.format(env, seed), 'rb') as f:
        paths = pickle.load(f)

    config = 'config/{}_config.yaml'.format(prefix)

    with open(config, 'r') as f:
        config_info = yaml.load(f, yaml.Loader)

    random.seed(seed)
    np.random.seed(seed)

    suffix = 'seed_{}'.format(seed)

    num_train_paths = config_info['num_train_paths']
    num_val_paths = config_info['num_val_paths']
    num_test_paths = config_info['num_test_paths']
    K = config_info['num_episodes']
    novel_test_probs = config_info['novel_test_probs']
    n_stripes = config_info['n_stripes']

    # generate the train, test, and validation fov ranges
    # breaks down the ranges for each fov into uniform stripes and assigns 
    # them randomly to train, val, and test
    fov_ranges = {}
    for fov, params in config_info['fov_params'].items():
        ranges = get_ranges(params['low'], params['high'], n_stripes)
        random.shuffle(ranges)
        n_train_stripe = int(config_info['prop_train_stripe'] * len(ranges))
        n_val_stripe = int(config_info['prop_val_stripe'] * len(ranges))
        n_test_stripe = len(ranges) - n_train_stripe - n_val_stripe
        train_ranges = ranges[:n_train_stripe]
        val_ranges = ranges[n_train_stripe:n_train_stripe+n_val_stripe]
        test_ranges = ranges[-n_test_stripe:]
    
        fov_ranges[fov] = dict(train_ranges=train_ranges,
            test_ranges=test_ranges, val_ranges=val_ranges)

    sample_train = gen_sample_train(fov_ranges)
    sample_val = gen_sample_other(fov_ranges, 1.0, other='val')
    sample_tests = [gen_sample_other(fov_ranges, p) for p in novel_test_probs]

    test_names = ['test_{}'.format(p) for p in novel_test_probs]


    train_states, train_actions, train_state_primes, train_fovs = modify_paths(paths[:num_train_paths], sample_train)
    val_states, val_actions, val_state_primes, val_fovs = modify_paths(paths[num_train_paths:num_train_paths+num_val_paths], sample_val)

    with open(os.path.join(args.output_dir,'{}_relabeled_train_{}.pickle'.format(prefix, suffix)), 'wb') as f:
        info=dict(state=train_states, act=train_actions, state_prime=train_state_primes, 
            fov=train_fovs, fov_ranges=fov_ranges, config_info=config_info, args=args)
        pickle.dump(info, f)

    with open(os.path.join(args.output_dir,'{}_relabeled_val_{}.pickle'.format(prefix, suffix)), 'wb') as f:
        info=dict(state=val_states, act=val_actions, state_prime=val_state_primes, 
            fov=val_fovs, fov_ranges=fov_ranges, config_info=config_info, args=args)
        pickle.dump(info, f)

    for test_name, test_fn in zip(test_names, sample_tests):
        test_states, test_actions, test_state_primes, test_fovs = modify_paths(paths[-num_test_paths:], test_fn)
        info=dict(state=test_states, act=test_actions, state_prime=test_state_primes,
            fov=test_fovs, fov_ranges=fov_ranges, config_info=config_info, args=args)
        with open(os.path.join(args.output_dir,'{}_relabeled_{}_{}.pickle'.format(prefix, test_name, suffix)), 'wb') as f:
            pickle.dump(info, f)

