
import numpy as np
from varyingsim.envs.slide_puck import SlidePuck
from time import sleep
from mujoco_py import functions


def set_all(env, i, t):
    pass

reset_list = [(0., 0., 0.)] * 10
env = SlidePuck(set_param_fn=set_all, rand_reset=False, reset_list=reset_list)
env.reset_idx=0

env.set_fovs([ 0.025     ,  0.16673632,  0.03734593,  0.06859797, -0.46803377,
         -0.20832266,  3.69001357, -4.11623421,  0.01413173])

actions = [
    np.array([0.3, 0.0, 0.0, 0.2]),
]

T = 6000

# env.set_puck_size(0.2)
# env.set_friction(0.00001)
# env.set_damping(0.0)
# env.set_table_tilt_x(10.0)
# env.set_table_tilt_y(10.0)
# functions.mj_setConst(env.model, env.sim.data)

obs = None

step = 0
action_idx = 0
traj_idx = 0

push_radius = env.max_puck_rad + 0.01
action_std = 0.1 * 0.0

min_vel, max_vel = 0.1, 0.3

next_act = None
def act_fn(state, i, t, memory):
    # if t == 0:
    #     return [0.3, 0.0, 0.0, 0.2, True]
    # else:
    #     return [0.3, 0.0, 0.0, 0.2, False]
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

obss = []
while True:
    obss = []
    # env.reset()
    if step % T == 0:
        if obs is not None:
            print('last obs', obs)
        obs = env.reset()
        obss.append(obs.copy())
        """
        env.set_fovs([ 2.50000000e-02,  2.47575458e-01,  6.78765698e-02,  6.10208336e-02,
        2.55620052e-01, -1.23764556e-01, -4.13725658e+00,  1.82737964e+00,
        1.56834652e-03])
        functions.mj_setConst(env.model, env.sim.data)
        """

        print('first obs', obs)
        print('fov', env.get_fovs())

        # damping = 10.0 ** (-traj_idx)
        # print(damping)
        # env.set_damping(damping)
        # if step % 4000 == 0 :
        # a = actions[0]
        # env.set_puck_size(np.sqrt(step) / 1000)
        # env.sim.set_constants()
        # else:
        #     a = np.array([0.5, np.pi / 2, 0.0])
        traj_idx += 1
    


    # if step % 50 == 0 and obs is not None:
    #     print(np.linalg.norm(obs[9:15])) 
    
    # if step % T == 0:
    #     obs, rew, done, info = env.step(np.concatenate([a, [True]]))
    # else:
    #     obs, rew, done, info = env.step(np.concatenate([a, [False]]))
    act = act_fn(obs, 0, step % T, None)
    obs, rew, done, info = env.step(act)

    if step % T == 0:
        print('second obs', obs)

    # print('act', act)
    # if step % T == 550:
        # import ipdb; ipdb.set_trace()

    env.render()
    step += 1

    # if step % 50 == 0:
    #     print(step)
    #     print(obs)
    #     print(a)
    #     print(env.model.body_mass)
    #     print()

  