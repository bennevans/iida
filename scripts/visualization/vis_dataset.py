from varyingsim.datasets.fov_dataset import SmoothFovDataset
from varyingsim.envs.push_box_circle import PushBoxCircle
from varyingsim.envs.cartpole import CartpoleEnv
import matplotlib.pyplot as plt
import time
from varyingsim.util.view import obs_to_relative

# location = "/data/varyingsim/datasets/push_box_action_0/dataset_0.pickle"
location = '/data/varyingsim/datasets/push_box_n_1_seed_0.pickle'
# location = "D:\\data\\varyingsim\\push_box_contact_simple_0.pickle"
dataset = SmoothFovDataset(location, 32, obs_skip=1)

# env_type = 'CARTPOLE'
env_type = 'PUSHBOX'

if env_type == 'PUSHBOX':
    env = PushBoxCircle()
elif env_type == 'CARTPOLE':
    env = CartpoleEnv()

env.reset()

idx_start = 0
idx_end = -1

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
        obs = traj['obs']
        act = traj['act']
        
        if env_type == 'PUSHBOX':
            qpos = obs[:9]
            qvel = obs[9:17]
        elif env_type == 'CARTPOLE':
            qpos = obs[:2]
            qvel = obs[2:]

        xs.append(qpos[0])
        ys.append(qpos[1])
        env.set_state(qpos, qvel)
        env.sim.step()
        fov = traj['fov']
        env.set_fovs(fov)
        print(i, traj['fov'], act)
        print(obs_to_relative(obs, act))
        env.render()
        # time.sleep(0.01)
        
    # plt.plot(xs, ys)
    # plt.show()