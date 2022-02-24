import argparse
import ctypes
from varyingsim.envs.push_box_offset import PushBoxOffset
from varyingsim.datasets.fov_dataset import ParallelEpisodicStartEndFovDatasetGenerator
import numpy as np
import os
import matplotlib.pyplot as plt
from varyingsim.util.system import get_true_cores
import time
from create_da_dataset_slide_puck import set_fovs, gen_sample_train, gen_sample_other
from varyingsim.util.dataset import get_ranges
import yaml

env = PushBoxOffset()

# next_act = None
# def get_fixed_acts(R, reset_list, push_radius, push_length):
#     start_acts = np.zeros((R, env.model.nu))
#     cur_acts = np.zeros((R, env.model.nu))
#     for r in range(R):
#         # box_xy = env.model.qpos0[:2]
#         box_xy = reset_list[r][:2][::-1]
#         # generate push
#         theta1 = np.random.random() * 2 * np.pi
#         theta2 = np.random.normal(theta1 + np.pi, 0.5) # center around other side of box
#         start_offset_xy = push_radius * np.array([np.cos(theta1), np.sin(theta1)])
#         end_offset_xy = push_radius * np.array([np.cos(theta2), np.sin(theta2)])
#         start_xy = box_xy + start_offset_xy
#         end_xy = box_xy + end_offset_xy

#         # + pi for opposite angle
#         angle = np.pi + np.arctan2(end_offset_xy[1] - start_offset_xy[1], end_offset_xy[0] - start_offset_xy[0])
#         push_vel = np.random.random() * 0.6 + 0.4
#         push_time = push_length / push_vel * 4 # distance / (distance / time) = time

#         start_acts[r] = np.concatenate([start_xy, [angle], [push_vel], [push_time], [1]], axis=0)
#         cur_acts[r] = np.concatenate([start_xy, [angle], [push_vel], [push_time], [0]], axis=0)
#     return start_acts, cur_acts

# start_acts, cur_acts = get_fixed_acts()
# def fixed_act_fn(state, i, t, memory):
#     if t == 0:
#         return start_acts[i % R]
#     return cur_acts[i % R]

# def seeded_act_fn(state, i, t, memory):
#     np.random.seed(i % R) # ensures we have the same action for each episode

#     # get current box pos
#     box_xy = state[:2]
#     # generate push
#     theta1 = np.random.random() * 2 * np.pi
#     theta2 = np.random.normal(theta1 + np.pi, 0.5) # center around other side of box
#     start_offset_xy = push_radius * np.array([np.cos(theta1), np.sin(theta1)])
#     end_offset_xy = push_radius * np.array([np.cos(theta2), np.sin(theta2)])
#     start_xy = box_xy + start_offset_xy
#     end_xy = box_xy + end_offset_xy

#     # + pi for opposite angle
#     angle = np.pi + np.arctan2(end_offset_xy[1] - start_offset_xy[1], end_offset_xy[0] - start_offset_xy[0])
#     push_vel = np.random.random() * 0.6 + 0.4
#     push_time = push_length / push_vel * 4 # distance / (distance / time) = time

#     if t == 0:
#         return np.concatenate([start_xy, [angle], [push_vel], [push_time], [1]], axis=0)
#     else:
#         return np.concatenate([start_xy, [angle], [push_vel], [push_time], [0]], axis=0)

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
        end_xy = box_xy + end_offset_xy

        # + pi for opposite angle
        angle = np.pi + np.arctan2(end_offset_xy[1] - start_offset_xy[1], end_offset_xy[0] - start_offset_xy[0])
        push_vel = np.random.random() * 0.6 + 0.4
        push_time = push_length / push_vel * 4 # distance / (distance / time) = time

        next_act = np.concatenate([start_xy, [angle], [push_vel], [push_time], [0]], axis=0)
        cur_act = np.concatenate([start_xy, [angle], [push_vel], [push_time], [1]], axis=0)

        return cur_act
    return next_act

# TODO: same state and same act
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-s', '--seed', default=0, type=int)
    args = parser.parse_args()

    seed = args.seed

    with open(args.config, 'r') as f:
        push_box_info = yaml.load(f)

    base_dir = push_box_info['base_dir']

    if not os.path.isdir(base_dir):
        raise FileNotFoundError('{} is not a valid folder'.format(base_dir))

    action_std = push_box_info['action_std']
    K_train = push_box_info['num_train_episodes'] # number of episodes
    R_train = push_box_info['num_train_actions'] # number of actions per epsidoe
    K_val = push_box_info['num_val_episodes'] # number of episodes
    R_val = push_box_info['num_val_actions'] # number of actions per epsidoe
    K_test = push_box_info['num_test_episodes'] # number of episodes
    R_test = push_box_info['num_test_actions'] # number of actions per epsidoe

    push_radius = push_box_info['push_radius']
    push_length = push_box_info['push_length']
    n_stripes = push_box_info['n_stripes']

    episode_length = 2000

    np.random.seed(seed)

    train_seed = custom_base_seed = ctypes.c_uint32(hash(str(seed))).value
    val_seed = custom_base_seed = ctypes.c_uint32(hash(str(seed+1))).value
    test_seed = custom_base_seed = ctypes.c_uint32(hash(str(seed+2))).value

    def dataset_constructor(include_fov=False):
        return PushBoxOffset(include_fov=include_fov, rand_reset=True)

    fov_ranges = {}
    for fov, params in push_box_info['fov_params'].items():
        ranges = get_ranges(params['low'], params['high'], n_stripes)

        n_train_stripe = int(push_box_info['prop_train_stripe'] * len(ranges))
        n_val_stripe = int(push_box_info['prop_val_stripe'] * len(ranges))
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

    def set_train_fov(env, i, t, memory):
        if i >= K_train * R_train:
            return
        if t == 0:
            fov_value = train_fovs[i // R_train]
            set_fovs(env, fov_value)

    def set_val_fov(env, i, t, memory):
        if i >= K_val * R_val:
            return
        if t == 0:
            fov_value = val_fovs[i // R_val]
            set_fovs(env, fov_value)
    
    def set_test_fov(env, i, t, memory):
        if i >= K_test * R_test:
            return
        if t == 0:
            fov_value = test_fovs[i // R_test]
            set_fovs(env, fov_value)

    with open(__file__, 'r') as f:
        file_lines = f.readlines()
        file_source = ''.join(file_lines)

    other_info = dict(generate_str=file_source, config=push_box_info)

    train_location = os.path.join(base_dir, 'push_box_K_{}_R_{}_seed_{}_train.pickle'.format(K_train, R_train, seed))
    val_location = os.path.join(base_dir, 'push_box_K_{}_R_{}_seed_{}_val.pickle'.format(K_val, R_val, seed))
    test_location = os.path.join(base_dir, 'push_box_K_{}_R_{}_seed_{}_test.pickle'.format(K_test, R_test, seed))


    print('*****************************************')
    print('* using {} cores                        *'.format(get_true_cores()))
    print('*****************************************')

    train_gen = ParallelEpisodicStartEndFovDatasetGenerator(train_location, dataset_constructor, set_train_fov,
        act_fn, R_train, K_train, episode_length, other_info=other_info, base_seed=train_seed)
    val_gen = ParallelEpisodicStartEndFovDatasetGenerator(val_location, dataset_constructor, set_val_fov,
        act_fn, R_val, K_val, episode_length, other_info=other_info, base_seed=val_seed)
    test_gen = ParallelEpisodicStartEndFovDatasetGenerator(test_location, dataset_constructor, set_test_fov,
        act_fn, R_test, K_test, episode_length, other_info=other_info, base_seed=test_seed)
