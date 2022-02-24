
import gym
import varyingsim.envs
import numpy as np

import time
# env = gym.make('VaryingHopper-v0')
# env.env.set_torso_length(0.06)
# env.env.set_thigh_length(0.06)
# env.env.set_leg_length(0.03)


# env = gym.make('VaryingSwimmer-v0')
# env.env.set_torso_length(0.12)
# env.env.set_mid_length(0.09)
# env.env.set_back_length(0.1)

env = gym.make('VaryingHumanoid-v0')
env.env.set_thigh_offset(0.2)
env.env.set_foot_offset(-0.45)
env.env.set_upper_arm_length(0.25)
# env.env.set_lower_arm_length(0.25)


env.sim.set_constants()
env.reset()


for i in range(5000):
    if i == 0:
        act = env.action_space.sample()
    else:
        act = np.zeros_like( env.action_space.sample())
    env.step(act)
    # import ipdb; ipdb.set_trace()
    env.render()

    if i > 10:
        env.env.viewer._paused = True