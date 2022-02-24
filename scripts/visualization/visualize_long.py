
import argparse
import os
import pickle
from varyingsim.datasets.fov_dataset import StartEndDataset
from varyingsim.util.trajectory import rollout_algo_single
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import yaml

def show_rollout(exp_dir, data_dir, idx, start=0, end=None):
    
    with open(os.path.join(exp_dir, 'algo.pickle'), 'rb') as f:
        algo = pickle.load(f)
    with open(os.path.join(exp_dir, 'info.pickle'), 'rb') as f:
        info = pickle.load(f)
    with open(os.path.join(exp_dir, 'params.yaml'), 'r') as f:
        params = yaml.load(f)

    
    dataset = StartEndDataset(data_dir, 16)

    datum = dataset[idx]
    obs_skip = params.obs_skip

    obs_full = datum['obs_full'][start+1:end:obs_skip]
    act_full = datum['obs_full'][start:end:obs_skip]

    traj = rollout_algo_single(algo, datum['obs_full'][0], act_full, end=end)
    plt.scatter(traj[start:end,0], traj[start:end, 1])
    plt.scatter(obs_full[:, 0], obs_full[:, 1])
    plt.show()

    return algo, info, dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--exp-dir', required=True, help='experiment directory')
    parser.add_argument('-a', '--data-dir', required=True, help='dataset directory')
    parser.add_argument('-i', '--index', required=False, default=0, type=int)
    parser.add_argument('-s', '--start', default=0, type=int)
    parser.add_argument('-e', '--end', default=None, type=int)
    args = parser.parse_args()

    algo, info, dataset = show_rollout(args.exp_dir, args.data_dir,
        args.index, args.start, args.end)

