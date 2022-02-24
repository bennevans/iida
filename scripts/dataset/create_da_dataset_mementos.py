from varyingsim.envs.push_box_circle import PushBoxCircle, sin_com_slow
from varyingsim.envs.push_box_offset import PushBoxOffset
from varyingsim.datasets.fov_dataset import ParallelEpisodicFovDatasetGenerator, ParallelEpisodicStartEndFovDatasetGenerator
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from varyingsim.util.view import euler_from_quaternion
from varyingsim.util.system import is_cassio, is_greene, get_true_cores
import time

seed = 0
K = 100 # number of episodes
R = 20 # number of actions per episode
push_radius = 0.15
push_length = 0.5
episode_length = 2000

np.random.seed(seed)


reset_list = []
for i in range(R):
    rand_x = np.random.random()- 0.5
    rand_y = np.random.random()- 0.5
    rand_angle = np.random.random() * 2 * np.pi
    reset_list.append((rand_x, rand_y, rand_angle))

print(reset_list)

def dataset_constructor(include_fov=False):
    # return PushBoxOffset(include_fov=include_fov, rand_reset=True, reset_list=reset_list)
    return PushBoxOffset(include_fov=include_fov, rand_reset=True)
    # return PushBoxOffset(include_fov=include_fov, rand_reset=False)

env = dataset_constructor()

next_act = None

def get_fixed_acts():
    start_acts = np.zeros((R, env.model.nu))
    cur_acts = np.zeros((R, env.model.nu))
    for r in range(R):
        # box_xy = env.model.qpos0[:2]
        box_xy = reset_list[r][:2][::-1]
        # generate push
        theta1 = np.random.random() * 2 * np.pi
        theta2 = np.random.normal(theta1 + np.pi, 0.5) # center around other side of box
        start_offset_xy = push_radius * np.array([np.cos(theta1), np.sin(theta1)])
        end_offset_xy = push_radius * np.array([np.cos(theta2), np.sin(theta2)])
        start_xy = box_xy + start_offset_xy
        end_xy = box_xy + end_offset_xy

        # + pi for opposite angle
        angle = np.pi + np.arctan2(end_offset_xy[1] - start_offset_xy[1], end_offset_xy[0] - start_offset_xy[0])
        push_vel = np.random.random() * 0.6 + 0.4
        push_time = push_length / push_vel * 4 # distance / (distance / time) = time

        start_acts[r] = np.concatenate([start_xy, [angle], [push_vel], [push_time], [1]], axis=0)
        cur_acts[r] = np.concatenate([start_xy, [angle], [push_vel], [push_time], [0]], axis=0)
    return start_acts, cur_acts

start_acts, cur_acts = get_fixed_acts()
def fixed_act_fn(state, i, t, memory):
    if t == 0:
        return start_acts[i % R]
    return cur_acts[i % R]

def seeded_act_fn(state, i, t, memory):
    np.random.seed(i % R) # ensures we have the same action for each episode

    # get current box pos
    box_xy = state[:2]
    # generate push
    theta1 = np.random.random() * 2 * np.pi
    theta2 = np.random.normal(theta1 + np.pi, 0.5) # center around other side of box
    start_offset_xy = push_radius * np.array([np.cos(theta1), np.sin(theta1)])
    end_offset_xy = push_radius * np.array([np.cos(theta2), np.sin(theta2)])
    start_xy = box_xy + start_offset_xy
    end_xy = box_xy + end_offset_xy

    # + pi for opposite angle
    angle = np.pi + np.arctan2(end_offset_xy[1] - start_offset_xy[1], end_offset_xy[0] - start_offset_xy[0])
    push_vel = np.random.random() * 0.6 + 0.4
    push_time = push_length / push_vel * 4 # distance / (distance / time) = time

    if t == 0:
        return np.concatenate([start_xy, [angle], [push_vel], [push_time], [1]], axis=0)
    else:
        return np.concatenate([start_xy, [angle], [push_vel], [push_time], [0]], axis=0)

def act_fn(state, i, t, memory):
    global next_act
    if t == 0:
        # generate new random action and store
        # get current box pos
        box_xy = state[:2]
        # generate push
        theta1 = np.random.random() * 2 * np.pi
        theta2 = np.random.normal(theta1 + np.pi, 0.5) # center around other side of box
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

        # print('end_xy', end_xy)
        # push_dist = push_vel * push_time
        # calc_offset = push_dist * np.array([np.cos(angle), np.sin(angle)])
        # print('calculated_end_xy', start_xy + calc_offset)

        # plt.xlim(-1,1)
        # plt.ylim(-1,1)
        # plt.title('angle {}'.format(angle))
        # plt.scatter(box_xy[0], box_xy[1], label='box_xy')
        # plt.scatter(start_xy[0], start_xy[1], label='start_xy')
        # plt.scatter(end_xy[0], end_xy[1], label='end_xy')
        # plt.legend()
        # plt.show()

        return cur_act
    return next_act


com_offsets = np.random.uniform(-0.14, 0.14, K)
box_masses = np.random.uniform(0.5, 2.5, K)
floor_frictions = np.random.uniform(0.7, 1.3, K)
box_frictions = np.random.uniform(0.7, 1.3, K)
pusher_frictions = np.random.uniform(0.7, 1.3, K)


def set_fov(env, i, t, memory):
    if i >= K * R:
        return

    env.set_com_offset(com_offsets[i // R])
    # env.set_box_mass(box_masses[i // R])
    # env.set_floor_friction(floor_frictions[i // R])
    # env.set_box_friction(box_frictions[i // R])
    # env.set_pusher_friction(pusher_frictions[i // R])


with open(__file__, 'r') as f:
    file_lines = f.readlines()
    file_source = ''.join(file_lines)

other_info = dict(generate_str=file_source)

if is_cassio():
    location = '/misc/vlgscratch5/PintoGroup/bne215/data/varyingsim/datasets/push_box_se_same_act_K_{}_R_{}_seed_{}.pickle'.format(K, R, seed)
if is_greene():
    location = '/home/bne215/data/varyingsim/datasets/push_box_se_same_act_K_{}_R_{}_seed_{}.pickle'.format(K, R, seed)
else:
    # location = '/data/mob/push_box_se_same_act_K_{}_R_{}_seed_{}.pickle'.format(K, R, seed)
    location = '/data/mob/push_box_se_only_offset_K_{}_R_{}_seed_{}.pickle'.format(K, R, seed)


data_folder, data_file = os.path.split(location)
if not os.path.isdir(data_folder):
    raise FileNotFoundError('{} is not a valid folder'.format(data_folder))

print('*****************************************')
print('* using {} cores                        *'.format(get_true_cores()))
print('*****************************************')

time.sleep(5.0)

gen = ParallelEpisodicFovDatasetGenerator(location, dataset_constructor, set_fov,
    act_fn, R, K, episode_length, other_info=other_info, base_seed=seed)

# location = '/data/varyingsim/datasets/push_box_same_act_K_{}_R_{}_seed_{}.pickle'.format(K, R, seed)
# gen = ParallelEpisodicFovDatasetGenerator(location, dataset_constructor, set_fov, seeded_act_fn, R, K, episode_length, other_info=other_info, base_seed=seed)
