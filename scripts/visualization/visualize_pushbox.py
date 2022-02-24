
import numpy as np
from varyingsim.envs.push_box_offset import PushBoxOffset
def set_all(env, i, t):
    com_offset = np.sin(t / 2000) * 0.05
    env.set_com_offset(com_offset)
    # env.set_box_friction(t/100.0 + 1)
    # env.set_floor_friction(t/100.0 + 1)
    pass



push_radius = 0.15
push_length = 0.5
episode_length = 2000
R = 10

env = PushBoxOffset(include_fov=False, rand_reset=False)
env.set_com_offset(0.01)

next_act = None 

def act_fn(state, i, t, memory):
    global next_act
    if t == 0:
        # generate new random action and store
        # get current box pos
        box_xy = state[:2]
        # generate push
        # theta1 = np.random.random() * 2 * np.pi
        # theta2 = np.random.normal(theta1 + np.pi, 0.5) # center around other side of box
        theta2 = 0.0
        theta1 = np.pi

        start_offset_xy = push_radius * np.array([np.cos(theta1), np.sin(theta1)])
        end_offset_xy = push_radius * np.array([np.cos(theta2), np.sin(theta2)])
        start_xy = box_xy + start_offset_xy 
        print('box')
        print(box_xy)
        print('start')
        print(start_offset_xy)
        print('end')
        print(end_offset_xy)

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

step = 0
obs = env.reset()
while True:
    # env.reset()
    # if step % 2000 == 0:
    #     # if step % 4000 == 0 :
    #     a = np.array([0, 0, 0, 1.0, 1.0])
    #     # else:
    #     #     a = np.array([0.5, np.pi / 2, 0.0])
    # if step % 2000 == 0:
    #     obs, rew, done, info = env.step(np.concatenate([a, [True]]))
    # else:
    #     obs, rew, done, info = env.step(np.concatenate([a, [False]]))
    a = act_fn(obs, 0, step % 2000 , {})
    obs, rew, done, info = env.step(a)
    if step % 2000 == 0:
        env.set_com_offset(np.random.uniform(-0.1,0.1))
        env.sim.set_constants()
        obs = env.reset()
    env.render()
    step += 1
    # print(obs)
    # print(a)
    # print()

  