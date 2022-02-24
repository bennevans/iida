"""
Takes a trained IIDA model and a dataset and dataset index and finds the closest 
neighbor in z space
"""

import argparse
from varyingsim.util.parsers import parse_dataset, parse_env, parse_model
from varyingsim.util.learn import torchify
import os
import yaml
import torch

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np


def find_and_visualize(batch, info, idx=0):
    zs = info['latents'].detach().cpu().numpy()
    zs_copy = np.copy(zs)

    query = np.copy(zs[idx])
    zs[idx] = 0

    closest = np.argmin(np.linalg.norm(zs - query, axis=1))

    objects = np.array(batch['object'])

    tsne = TSNE()
    zs_low = tsne.fit_transform(zs_copy)

    uniq_objects = np.unique(objects)

    for obj in uniq_objects:
        idxs = np.where(objects == obj)
        plt.scatter(zs_low[idxs, 0], zs_low[idxs, 1], label=obj)

    s = 144
    plt.scatter(zs_low[idx, 0], zs_low[idx, 1], label='query {} idx {}'.format(objects[idx], idx), marker='x', s=s)
    plt.scatter(zs_low[closest, 0], zs_low[closest, 1], label='closest {} idx {}'.format(objects[closest], closest), marker='x', s=s)

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('nearest neighbor to idx {}'.format(idx))
    plt.legend()
    plt.show()

def find_nearest_all(batch, info):
    objects = np.array(batch['object'])
    zs = info['latents'].detach().cpu().numpy()

    obj_stats = {}

    for i, obj in enumerate(objects):
        zs_copy = np.copy(zs)

        query = np.copy(zs[i])
        zs_copy[i] = 0
        closest = np.argmin(np.linalg.norm(zs_copy - query, axis=1))
        
        same_obj = objects[closest] == obj
        if obj in obj_stats:
            obj_stats[obj].append(same_obj)
        else:
            obj_stats[obj] = [same_obj]

    all_success = []
    for obj, success in obj_stats.items():
        print(obj, np.mean(success), len(success))
        all_success += success
    print('mean success', np.mean(all_success))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, \
        # default='/data/domain_adaptation/experiments/sliding/baseline/sliding_transformer-2022-01-11_18-49-46')
        # default='/data/domain_adaptation/experiments/sliding/baseline/sliding_continuous-2022-01-11_18-45-29')
        default='/data/domain_adaptation/experiments/sliding/baseline/sliding_rnn-2022-01-11_18-50-38')

    parser.add_argument('--idx', default=0, type=int)
    parser.add_argument('--context-size', default=None, type=int)
    parser.add_argument('--run-all', action='store_true', default=False)
    parser.add_argument('--dataset', default='train', choices=['train', 'test', 'val'])

    args = parser.parse_args()

    exp_loc = args.experiment
    param_loc = os.path.join(exp_loc, 'params.yaml')
    # model_loc = os.path.join(exp_loc, 'model.pickle')
    model_loc = os.path.join(exp_loc, 'model_best_val_loss.pickle')
    model_opt_loc = os.path.join(exp_loc, 'model_options.yaml')

    with open(param_loc, 'r') as f:
        params = yaml.load(f)
        if args.context_size is not None:
            params.context_size = args.context_size
    with open(model_opt_loc, 'r') as f:
        model_options = yaml.load(f)
    with open(model_loc, 'rb') as f:
        model_params = torch.load(f)
        env = parse_env(params)
        model = parse_model(params, model_options, env)
        model.load_state_dict(model_params)

    train_dataset, test_sets, val_set = parse_dataset(params)

    if args.dataset == 'train':
        dataset = train_dataset
    elif args.dataset == 'val':
        dataset = val_set
    elif args.dataset == 'test':
        dataset = test_sets['test']

    loader = DataLoader(dataset, batch_size=len(dataset))
    batch = next(iter(loader))

    result, info = model(batch, ret_extra=True)
    
    if args.run_all:
        find_nearest_all(batch, info)
    else:
        find_and_visualize(batch, info, args.idx)
