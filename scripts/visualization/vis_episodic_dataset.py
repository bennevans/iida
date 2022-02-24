from varyingsim.datasets.fov_dataset import SmoothEpisodicFovDataset
from varyingsim.envs.push_box_offset import PushBoxOffset
from varyingsim.envs.cartpole import CartpoleEnv
from varyingsim.envs.slide_puck import SlidePuck
import matplotlib.pyplot as plt
import time
from varyingsim.util.view import obs_to_relative
import numpy as np

# location = "/data/varyingsim/datasets/push_box_action_0/dataset_0.pickle"
location = '/data/varyingsim/datasets/push_box_se_same_act_K_20_R_20_seed_5.pickle'
# location = "D:\\data\\varyingsim\\datasets\\push_box_K_10_R_2_seed_0.pickle"
dataset = SmoothEpisodicFovDataset(location, 32, obs_skip=1)
# env_type = 'CARTPOLE'
env_type = 'SLIDEPUCK'

if env_type == 'PUSHBOX':
    env = PushBoxOffset()
elif env_type == 'CARTPOLE':
    env = CartpoleEnv()
elif env_type == 'SLIDEPUCK':
    env = SlidePuck()

env.reset()

idx_start = 60
idx_end = 61

# idx_start = 20
# idx_end = idx_start + 1

idx_1 = dataset.prefix_lens[idx_start]
idx_2 = dataset.prefix_lens[idx_end]
# idx_2 = dataset.prefix_lens[idx_end + 1]
# idx_2 = idx_1 + 2000

print(idx_1, idx_2)

while True:
    xs = []
    ys = []
    for i in range(idx_1, idx_2):
        traj = dataset[i]
        state = traj['state']
        act = traj['act']
        
        if env_type == 'PUSHBOX':
            qpos = state[:9]
            qvel = state[9:17]
        elif env_type == 'CARTPOLE':
            qpos = state[:2]
            qvel = state[2:]

        xs.append(qpos[0])
        ys.append(qpos[1])
        env.set_state(qpos, qvel)
        env.sim.step()
        fov = traj['fov']
        env.set_fovs(fov)
        print(i, traj['fov'], act)
        pusher_xy = state[7:9] - np.array([0.2, 0])
        print('pusher_xy', pusher_xy)
        env.render()
        # time.sleep(0.01)
        
    # plt.plot(xs, ys)
    # plt.show()