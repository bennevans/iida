"""
Takes a trained IIDA model and a dataset and visualizes the learned latent space
"""

import argparse
from varyingsim.util.parsers import parse_dataset, parse_env, parse_model
from varyingsim.util.learn import torchify, train
import os
import yaml
import torch

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

def visualize_pca(batch, info):
    zs = info['latents'].detach().cpu().numpy()
    objects = np.array(batch['object'])

    pca = PCA(2)
    zs_low = pca.fit_transform(zs)

    visualize(zs_low, objects, 'pca')

def visualize_tsne(batch, info, title='TSNE'):
    zs = info['latents'].detach().cpu().numpy()
    objects = np.array(batch['object'])

    tsne = TSNE()
    zs_low = tsne.fit_transform(zs)

    visualize(zs_low, objects, title)


def visualize_cloth(batch, info, title=''):
    zs = info['latents'].detach().cpu().numpy()
    objects = np.array(batch['object'])

    tsne = TSNE()
    zs_low = tsne.fit_transform(zs)

    uniq_objects = np.unique(objects)
    uniq_cloths = np.unique([o for o in batch['object'] if o.startswith('cloth')])

    c_lab, nc_lab = False, False
    for obj in uniq_objects:
        idxs = np.where(objects == obj)
        if obj in uniq_cloths:
            c = 'r'
            label = 'cloth' if not c_lab else ''
            c_lab = True
        else:
            c = 'b'
            label = 'no cloth' if not nc_lab else ''
            nc_lab = True        
        plt.scatter(zs_low[idxs, 0], zs_low[idxs, 1], label=label, c=c)

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(title)
    plt.legend()
    plt.show()

def rand_color_str():
    r = np.random.randint(255)
    g = np.random.randint(255)
    b = np.random.randint(255)
    # b = 255 - (r+g) // 2
    return '#%02X%02X%02X' % (r, g, b)

def visualize(zs, objects, title=''):

    uniq_objects = np.unique(objects)
    plt.figure(figsize=(12,6))

    # colors = {u: rand_color_str() for u in uniq_objects}
    cmap = plt.cm.get_cmap('tab20')

    for i, obj in enumerate(uniq_objects):
        idxs = np.where(objects == obj)
        color_i = i * cmap.N // len(uniq_objects)
        print(i, color_i)
        plt.scatter(zs[idxs, 0], zs[idxs, 1], label=obj, color=cmap(color_i))
        # plt.scatter(zs[idxs, 0], zs[idxs, 1], label=obj, color=colors[obj])


    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, \
        # default='/data/domain_adaptation/experiments/sliding/baseline/sliding_transformer-2022-01-11_18-49-46')
        # default='/data/domain_adaptation/experiments/sliding/baseline/sliding_continuous-2022-01-11_18-45-29')
        default='/data/domain_adaptation/experiments/sliding/baseline/sliding_rnn-2022-01-11_18-50-38')
    parser.add_argument('--context-size', default=None, type=int)
    parser.add_argument('--dataset', default='train', choices=['train', 'test', 'val'])
    parser.add_argument('--cloth', action='store_true')
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
    if args.cloth:
        visualize_cloth(batch, info, title='Context size {}'.format(params.context_size))
    else:
        visualize_tsne(batch, info, title='Context size {}'.format(params.context_size))
    # visualize_pca(batch, info)
    # import ipdb; ipdb.set_trace()