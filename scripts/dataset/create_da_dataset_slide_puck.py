import argparse
import ctypes
from varyingsim.envs.slide_puck import SlidePuck
from varyingsim.datasets.fov_dataset import ParallelEpisodicStartEndFovDatasetGenerator
import numpy as np
import os
from varyingsim.util.system import get_true_cores, hash_config
from varyingsim.util.dataset import get_ranges, sample_ranges
import yaml
from mujoco_py import functions
import random


def get_fixed_acts(R, reset_list):
    start_acts = np.zeros((R, env.model.nu))
    cur_acts = np.zeros((R, env.model.nu))
    for r in range(R):
        box_xy = reset_list[r][:2] #[::-1]
        print('r', r)
        print('box_xy', box_xy)

        # generate push
        # theta1 = np.random.random() * 2 * np.pi
        print('WARNING THIS IS DIFFERENT')
        theta1 = np.random.uniform(-0.1, 0.1)
        theta2 = np.random.normal(theta1 + np.pi, action_std) # center around other side of box
        start_offset_xy = push_radius * np.array([np.cos(theta1), np.sin(theta1)])
        end_offset_xy = push_radius * np.array([np.cos(theta2), np.sin(theta2)])
        start_xy = box_xy + start_offset_xy
        print('start_xy', start_xy)

        # + pi for opposite angle
        angle = np.pi + np.arctan2(end_offset_xy[1] - start_offset_xy[1], end_offset_xy[0] - start_offset_xy[0])
        push_vel = np.random.random() * 0.6 + 0.4

        start_acts[r] = np.concatenate([start_xy, [angle], [push_vel], [1]], axis=0)
        cur_acts[r] = np.concatenate([start_xy, [angle], [push_vel], [0]], axis=0)
    return start_acts, cur_acts

def gen_fixed_act_fn(R, start_acts, cur_acts):
    def act_fn(state, i, t, memory):
        if t == 0:
            return start_acts[i % R_train]
        return cur_acts[i % R_train]
    return act_fn

next_act = None
def act_fn(state, i, t, memory):
    global next_act
    if t == 0:
        # generate new random action and store
        # get current box pos
        box_xy = state[:2]
        # generate push
        theta1 = np.random.random() * 2 * np.pi
        theta2 = np.random.normal(theta1 + np.pi, action_std) # center around other side of box
        start_offset_xy = push_radius * np.array([np.cos(theta1), np.sin(theta1)])
        end_offset_xy = push_radius * np.array([np.cos(theta2), np.sin(theta2)])
        start_xy = box_xy + start_offset_xy

        # + pi for opposite angle
        angle = np.pi + np.arctan2(end_offset_xy[1] - start_offset_xy[1], end_offset_xy[0] - start_offset_xy[0])
        push_vel = np.random.uniform(min_vel, max_vel)

        next_act = np.concatenate([start_xy, [angle], [push_vel], [0]], axis=0)
        cur_act = np.concatenate([start_xy, [angle], [push_vel], [1]], axis=0)

        return cur_act
    return next_act

# functions that returns index of first and data point
def start_fn(state):
    try:
        first_moving_idx = np.where(np.linalg.norm(state[:, 9:11], axis=1) > START_THRESH)[0][0]
    except Exception as e:
        return 0
    return max(first_moving_idx - 1, 50)

def end_fn(state, start_search=0):
    stopped_moving = np.where(np.linalg.norm(state[start_search:, 9:11], axis=1) < END_THRESH)[0]
    if len(stopped_moving) == 0:
        return max_episode_length
    return start_search + stopped_moving[0]

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

def gen_set_fov(K, R, fovs):
    def set_fov(env, i, t, memory):
        if i >= K * R:
            return
        if t == 0:
            fov_value = fovs[i // R]
            set_fovs(env, fov_value)
    return set_fov

def gen_same_state(R, K):
    reset_list_short = []
    for i in range(R):
        rand_x = np.random.random()- 0.5
        rand_y = np.random.random()- 0.5
        rand_angle = np.random.random() * 2 * np.pi
        reset_list_short.append((rand_x, rand_y, rand_angle))

    reset_list = []
    for j in range(K):
        for x in reset_list_short:
            reset_list.append(x)
    return reset_list

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-s', '--seed', default=0, type=int)
    args = parser.parse_args()

    env = SlidePuck()

    with open(args.config, 'r') as f:
        puck_info = yaml.load(f)

    seed = args.seed
    base_dir = puck_info['base_dir']

    if not os.path.isdir(base_dir):
        raise FileNotFoundError('{} is not a valid folder'.format(base_dir))

    same_act = puck_info['same_act']
    K_train = puck_info['num_train_episodes'] # number of episodes
    R_train = puck_info['num_train_actions'] # number of actions per epsidoe
    K_val = puck_info['num_val_episodes'] # number of episodes
    R_val = puck_info['num_val_actions'] # number of actions per epsidoe
    K_test = puck_info['num_test_episodes'] # number of episodes
    R_test = puck_info['num_test_actions'] # number of actions per epsidoe

    START_THRESH = 1e-2
    END_THRESH = 1e-4
    max_episode_length = 6000
    push_radius = env.max_puck_rad + 0.01
    action_std = puck_info['action_std']
    min_vel = puck_info['min_vel']
    max_vel = puck_info['max_vel']
    n_stripes = puck_info['n_stripes']

    suffix = puck_info['suffix']
    rand_reset = puck_info['rand_reset']
    same_state = puck_info['same_state']

    np.random.seed(seed)

    train_seed = custom_base_seed = ctypes.c_uint32(hash(str(seed))).value
    val_seed = custom_base_seed = ctypes.c_uint32(hash(str(seed+1))).value
    test_seed = custom_base_seed = ctypes.c_uint32(hash(str(seed+2))).value

    if same_state:
        # gives same start state in each env        
        train_reset_list = gen_same_state(R_train, K_train)
        val_reset_list = gen_same_state(R_val, K_val)
        test_reset_list = gen_same_state(R_test, K_test)
    else:
        train_reset_list = None
        test_reset_list = None
        val_reset_list = None

    if same_act:
        train_start_acts, train_cur_acts = get_fixed_acts(R_train, train_reset_list)
        val_start_acts, val_cur_acts = get_fixed_acts(R_val, val_reset_list)
        test_start_acts, test_cur_acts = get_fixed_acts(R_test, test_reset_list)


    # load and calculate train, test fovs
    fov_ranges = {}
    for fov, params in puck_info['fov_params'].items():
        ranges = get_ranges(params['low'], params['high'], n_stripes)

        n_train_stripe = int(puck_info['prop_train_stripe'] * len(ranges))
        n_val_stripe = int(puck_info['prop_val_stripe'] * len(ranges))
        n_test_stripe = len(ranges) - n_train_stripe - n_val_stripe
        train_ranges = ranges[:n_train_stripe]
        val_ranges = ranges[n_train_stripe:n_train_stripe+n_val_stripe]
        test_ranges = ranges[-n_test_stripe:]

        fov_ranges[fov] = dict(train_ranges=train_ranges,
                test_ranges=test_ranges, val_ranges=val_ranges)

    train_fovs = []
    test_fovs = []
    val_fovs = []

    sample_train = gen_sample_train(fov_ranges)
    sample_val = gen_sample_other(fov_ranges, 1.0, 'val')
    sample_test = gen_sample_other(fov_ranges, 1.0, 'test')

    for i in range(K_train):
        train_fovs.append(sample_train())

    for i in range(K_val):
        val_fovs.append(sample_val())

    for i in range(K_test):
        test_fovs.append(sample_test())


    with open(__file__, 'r') as f:
        file_lines = f.readlines()
        file_source = ''.join(file_lines)

    other_info = dict(generate_str=file_source, config=puck_info, config_hash=hash_config(puck_info))

    sa_str = 'same_act_' if same_act else ''
    rr = '' if rand_reset else 'no_rr_'
    train_location = os.path.join(base_dir, 'slide_puck_K_{}_R_{}_seed_{}_{}{}{}train.pickle'.format(K_train, R_train, seed, sa_str, rr, suffix))
    val_location = os.path.join(base_dir, 'slide_puck_K_{}_R_{}_seed_{}_{}{}{}val.pickle'.format(K_val, R_val, seed, sa_str, rr, suffix))
    test_location = os.path.join(base_dir, 'slide_puck_K_{}_R_{}_seed_{}_{}{}{}test.pickle'.format(K_test, R_test, seed, sa_str, rr, suffix))

    data_folder, data_file = os.path.split(train_location)
    if not os.path.isdir(data_folder):
        raise FileNotFoundError('{} is not a valid folder'.format(data_folder))


    print('*****************************************')
    print('* using {} cores                        *'.format(get_true_cores()))
    print('*****************************************')

    if same_act:
        raise Exception('same act not implemented yet')
        # gen_fixed_act_fn
        # used_act_fn = fixed_test_act_fn
    else:
        used_act_fn = act_fn

    def train_dataset_constructor(include_fov=False):
        return SlidePuck(include_fov=include_fov, rand_reset=rand_reset, reset_list=train_reset_list)

    def val_dataset_constructor(include_fov=False):
        return SlidePuck(include_fov=include_fov, rand_reset=rand_reset, reset_list=val_reset_list)

    def test_dataset_constructor(include_fov=False):
        return SlidePuck(include_fov=include_fov, rand_reset=rand_reset, reset_list=test_reset_list)

    if R_train > 0 and K_train > 0:
        def set_train_fov(env, i, t, memory):
            if i >= K_train * R_train:
                return
            if t == 0:
                fov_value = train_fovs[i // R_train]
                set_fovs(env, fov_value)
                
        train_gen = ParallelEpisodicStartEndFovDatasetGenerator(train_location, train_dataset_constructor, set_train_fov,
            used_act_fn, R_train, K_train, max_episode_length, other_info=other_info, base_seed=train_seed, start_fn=start_fn, end_fn=end_fn)

    if R_val > 0 and K_val > 0:
        def set_val_fov(env, i, t, memory):
            if i >= K_val * R_val:
                return
            if t == 0:
                fov_value = val_fovs[i // R_val]
                set_fovs(env, fov_value)

        val_gen = ParallelEpisodicStartEndFovDatasetGenerator(val_location, val_dataset_constructor, set_train_fov,
            used_act_fn, R_val, K_val, max_episode_length, other_info=other_info, base_seed=val_seed, start_fn=start_fn, end_fn=end_fn)


    if R_test > 0 and K_test > 0:

        def set_test_fov(env, i, t, memory):
            if i >= K_test * R_test:
                return
            if t == 0:
                fov_value = test_fovs[i // R_test]
                set_fovs(env, fov_value)
                
        test_gen = ParallelEpisodicStartEndFovDatasetGenerator(test_location, test_dataset_constructor, set_test_fov,
            used_act_fn, R_test, K_test, max_episode_length, other_info=other_info, base_seed=test_seed, start_fn=start_fn, end_fn=end_fn)

    print(train_location)
    print(val_location)
    print(test_location)
