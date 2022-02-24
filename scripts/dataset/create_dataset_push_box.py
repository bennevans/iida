from varyingsim.envs.push_box_circle import PushBoxCircle, sin_com_slow
from varyingsim.envs.push_box_offset import PushBoxOffset
from varyingsim.datasets.fov_dataset import SmoothFovDataset, SmoothFovDatasetGenerator
import numpy as np

# TODO: this could be done in parallel

seed = 1
np.random.seed(seed)

# env = PushBoxCircle()
env = PushBoxOffset()
N = 4
T = 2000

# location = '/misc/vlgscratch5/PintoGroup/bne215/datasets/varyingsim/push_box_circle.pickle'
location = '/data/varyingsim/datasets/push_box_no_variation_{}.pickle'.format(seed)
# location = 'D:\\data\\varyingsim\\push_box_contact_simple_tiny_{}.pickle'.format(seed)


# cur_act = np.array([0.0, 0, 0])
# def act_fn(obs, i, t):
#     global cur_act # TODO: is there a better way?
#     if t == 0: # new episode! new control!
#         cur_act = np.array([0.1 + np.random.rand() * 0.7,
#                             np.random.rand() * 2 * np.pi,
#                             np.random.randn() * 0.1])
#     return cur_act

def act_fn(obs, i, t, memory):
    if i == 0:
        memory['velocity'] = 0.5
        memory['angle'] = 0.0
    elif t == 0:
        memory['velocity'] = 0.1 + np.random.random() * 0.9
        memory['angle'] = np.random.random() * 2 * np.pi
    return np.array([memory['velocity'], memory['angle'], 0.0])

# def act_fn(obs, i, t, memory):
#     if i == 0:
#         return np.array([0.5, 0.0, 0.0])
#     elif i == 1:
#         return np.array([0.5, np.pi / 2, 0.0])
#     elif i == 2:
#         return np.array([0.5, np.pi, 0.0])
#     elif i == 3:
#         return np.array([0.5, 3 * np.pi / 2, 0.0])
#     else:
#         return np.array([0.5, 0.0, 0.0])

def set_fov(env, i, t, memory):
    # env.set_com_offset(0.1)
    pass

# def set_fov(env, i, t, memory):
#     if i == 0:
#         return

#     my_t = T * i + t
#     delta = 0.01
#     dt = env.model.opt.timestep
#     max_d_com_offset = 1e-3

#     if 'com_offset' not in memory:
#         memory['com_offset'] = 0.0

#     if 'change' in memory and memory['since_change'] <= memory['change']:
#         memory['since_change'] += 1
#     else:
#         memory['A'] = 1 / (10 + np.random.randn())
#         memory['new_time'] = np.random.random() * max(1e-3, (8 + 2*np.random.randn()))
#         memory['f'] = 1 / (memory['new_time'] * (1/dt) * 2) # every new_time seconds switch
#         memory['since_change'] = 0
#         memory['change'] = memory['new_time'] / dt

#     memory['target_com_offset'] = (2*memory['A']/np.pi)*np.arctan(np.sin(2*np.pi*my_t*memory['f'])/delta)
#     error = memory['target_com_offset'] - memory['com_offset']
#     control = 1.0 * error
#     memory['com_offset'] = memory['com_offset'] + np.clip(control, -max_d_com_offset, max_d_com_offset)

#     env.set_com_offset(memory['com_offset'])

#     if 'fric_change' in memory and memory['since_fric_change'] <= memory['fric_change']:
#         memory['since_fric_change'] += 1
#         if 'sin_mode' in memory and memory['sin_mode']:
#             memory['box_fric'] = np.sin(my_t / (T * 5.37) * 2 * np.pi) / 10.0 + 1.0
#     else:
#         memory['since_fric_change'] = 0
#         memory['fric_change'] = (1 + np.random.random() * 9 ) / dt
#         if np.random.random() > 0.7:
#             memory['box_fric'] = 1 + np.random.randn() * 0.1
#             memory['floor_fric'] = 1 + np.random.randn() * 0.1
#             memory['sin_mode'] = False
#         else:
#             memory['sin_mode'] = True
#             memory['box_fric'] = np.sin(my_t / (T * 5.37) * 2 * np.pi) / 10.0 + 1.0
#             memory['floor_fric'] = 1.0
#     env.set_box_friction(memory['box_fric'])
#     env.set_floor_friction(memory['floor_fric'])

gen = SmoothFovDatasetGenerator(location, PushBoxOffset, set_fov, act_fn, N, T)
