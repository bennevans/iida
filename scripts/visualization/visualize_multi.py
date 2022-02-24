
import argparse
import os
import pickle
from varyingsim.datasets.fov_dataset import StartEndDataset
from varyingsim.util.trajectory import rollout_algo_single
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import math

# TODO: algo idx
 
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def show_rollout(exp_dir, data_dir, idx):
    algo_file = 'algo_100.pickle'
    with open(os.path.join(exp_dir, algo_file), 'rb') as f:
        algo = pickle.load(f)
    with open(os.path.join(exp_dir, 'info.pickle'), 'rb') as f:
        info = pickle.load(f)
    
    dataset = StartEndDataset(data_dir, 32)

    print(len(dataset))
    datum = dataset[idx]
    start_obs = torch.from_numpy(datum['obs']).float()

    end_obs = datum['obs_prime']

    traj = rollout_algo_single(algo, datum, end=2)
    print(traj)

    last_pos_hat = torch.from_numpy(traj[1][:7])
    last_pos = torch.from_numpy(end_obs[:7])
    print('7mse', F.mse_loss(last_pos, last_pos_hat).item())

    last_pos_hat = torch.from_numpy(traj[1][:2])
    last_pos = torch.from_numpy(end_obs[:2])
    print('2mse', F.mse_loss(last_pos, last_pos_hat).item())

    roll_x, pitch_y, yaw_z = euler_from_quaternion(traj[1, 4],traj[1,5],
                                                    traj[1,6],traj[1,3])

    mx, my = np.cos(yaw_z), np.sin(yaw_z)

    plt.plot(traj[:,0], traj[:, 1], label='rollout')
    plt.quiver(traj[1,0], traj[1, 1], mx, my, label='final obs hat')
    plt.scatter(datum['obs'][0], datum['obs'][1], label='obs_0')

    roll_x, pitch_y, yaw_z = euler_from_quaternion(end_obs[4],end_obs[5],
                                                    end_obs[6],end_obs[3])

    mx, my = np.cos(yaw_z), np.sin(yaw_z)

    print(roll_x, pitch_y, yaw_z)
    # plt.scatter(end_obs[0], end_obs[1], label='final obs')
    plt.quiver(end_obs[0], end_obs[1], mx, my, label='final obs')

    obs_primes = []
    for datum in dataset:
        obs_primes.append(datum['obs_prime'])
    obs_primes = np.stack(obs_primes)
    mean_xy = np.mean(obs_primes[:, :2], axis=0)
    plt.scatter(mean_xy[0], mean_xy[1], label='mean obs_prime')
    plt.legend()
    plt.title('rollout number: {} algo: {}'.format(idx, algo_file))
    plt.show()

    return algo, info, dataset

def show_rollouts(exp_dir, data_dir):
    
    with open(os.path.join(exp_dir, 'algo.pickle'), 'rb') as f:
        algo = pickle.load(f)
    with open(os.path.join(exp_dir, 'info.pickle'), 'rb') as f:
        info = pickle.load(f)
    
    dataset = StartEndDataset(data_dir, 16)
    datum = dataset[0]
    plt.scatter(datum['obs'][0], datum['obs'][1], label='obs_0')

    for datum in dataset:
        end_obs = datum['obs_prime']
        traj = rollout_algo(algo, datum, end=2)
        plt.plot(traj[:,0], traj[:, 1], c='b')#, label='rollout')
        plt.scatter(end_obs[0], end_obs[1], c='g')#, label='final obs')

    obs_primes = []
    for datum in dataset:
        obs_primes.append(datum['obs_prime'])
    obs_primes = np.stack(obs_primes)
    mean_xy = np.mean(obs_primes[:, :2], axis=0)
    plt.scatter(mean_xy[0], mean_xy[1], label='mean obs_prime')
    plt.legend()
    plt.title('all rollouts')
    plt.show()

    return algo, info, dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--exp-dir', required=True, help='experiment directory')
    parser.add_argument('-a', '--data-dir', required=True, help='dataset directory')
    parser.add_argument('-i', '--index', required=False, default=0, type=int)
    args = parser.parse_args()

    if args.index >= 0:
        algo, info, dataset = show_rollout(args.exp_dir, args.data_dir,
        args.index)
    else:
        algo, info, dataset = show_rollouts(args.exp_dir, args.data_dir)
