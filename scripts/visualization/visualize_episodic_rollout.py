
import argparse
import os
import pickle
from varyingsim.datasets.fov_dataset import EpisodicStartEndDataset
from varyingsim.util.trajectory import rollout_algo
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import yaml

import numpy as np

def show_rollout(exp_dir, data_dir, idx, start=0, end=None):
    model_file = 'model.pickle'

    with open(os.path.join(exp_dir, model_file), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(exp_dir, 'info.pickle'), 'rb') as f:
        info = pickle.load(f)
    with open(os.path.join(exp_dir, 'params.yaml'), 'rb') as f:
        params = yaml.load(f)

    dataset = EpisodicStartEndDataset(data_dir, H)
    obs_skip = params.obs_skip

    datum = dataset[idx]
    print('idx', idx)
    print('datum', datum['obs'])
    T = len(datum['obs_full'])
    num_obs = T // obs_skip
    print(num_obs)
    acts = datum['act_full'][::obs_skip]
    fovs = datum['fov_full'][::obs_skip]

    # TODO: rollout from not zeroth observation. right now we're only starting from the first

    traj = rollout_algo(algo, datum, acts, fovs, end=num_obs)
    plt.scatter(traj[start:end,0], traj[start:end, 1], label='Predicted trajectory')
    plt.scatter(datum['obs_full'][start+1:end:obs_skip, 0], datum['obs_full'][start+1:end:obs_skip, 1], label='True trajectory')
    plt.legend()
    title_str = "exp: {} data: {} index: {} algo_iter: {}".format(os.path.basename(exp_dir), os.path.basename(data_dir), idx, algo_file)
    plt.title(title_str)
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

    print('index', args.index)

    algo, info, dataset = show_rollout(args.exp_dir, args.data_dir,
        args.index, args.start, args.end)

