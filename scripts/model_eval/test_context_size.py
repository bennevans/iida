import sys
import os
from numpy.core.numeric import full
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import yaml
from varyingsim.util.parsers import parse_model
from varyingsim.envs.push_box_offset import PushBoxOffset
from varyingsim.envs.slide_puck import SlidePuck
from varyingsim.util.learn import create_train_point
from varyingsim.datasets.toy_dataset import ToyDataset
from varyingsim.util.view import push_box_state_to_xyt, slide_box_state_to_xyt
from varyingsim.datasets.fov_dataset import EpisodicStartEndFovDataset
import matplotlib.pyplot as plt

from copy import deepcopy

TOY = False

def get_errors(model_directory, context_sizes):

    with open(os.path.join(model_directory, 'model.pickle'),'rb') as f:
        state_dict = torch.load(f)
    with open(os.path.join(model_directory, 'params.yaml'),'r') as f:
        params = yaml.load(f)
    with open(os.path.join(model_directory, 'model_options.yaml'),'r') as f:
        model_options = yaml.load(f)

    print(model_options)

    # env = PushBoxOffset()
    env = SlidePuck()

    model = parse_model(params, model_options, env)
    model.load_state_dict(state_dict)

    # location = '/data/varyingsim/datasets/push_box_se_same_act_split_train_K_100_R_2000_KT_90_RT_1000_seed_0.pickle'
    # location = '/data/varyingsim/datasets/push_box_se_same_act_split_test_fov_K_100_R_2000_KT_90_RT_1000_seed_0.pickle'
    # dataset = EpisodicStartEndFovDataset(location, obs_fn=push_box_state_to_xyt)
    
    # location = "/data/varyingsim/datasets/slide_puck_K_100_R_10_seed_0_test.pickle"
    # location = "/data/varyingsim/datasets/slide_puck_K_5_R_4_seed_3735928559_same_act_vis_train.pickle"
    location = "/data/varyingsim/datasets/slide_puck_K_1000_R_10_seed_0_act_std_train.pickle"

    
    dataset = EpisodicStartEndFovDataset(location, obs_fn=slide_box_state_to_xyt)


    errors = []
    for context_size in context_sizes:
        n = 0
        s = 0.0
        loader = DataLoader(dataset, shuffle=True)
        for batch in loader:
            n += 1
            data = create_train_point(batch, n_batch=128, n_context=context_size, device='cuda')
            print(data['context_obs'].shape)
            loss, info = model.loss(data, return_info=True)
            s += info['base_loss']
        errors.append(s / n)
    return errors, model.context_size

if __name__ == '__main__':
    context_sizes = [1, 2, 4, 8, 16, 32]

    # directories = ['slidepuck-2021-06-22_12-24-42', 'slidepuck-2021-06-22_12-24-36', 'slidepuck-2021-06-22_12-24-30', 'slidepuck-2021-06-22_12-24-25']
    directories = ['slidepuck-2021-06-23_17-49-59'] # no vel
    # directories = ['slidepuck-2021-06-23_17-50-13'] # no vel ff
    
    for model_directory in directories:
        full_path = os.path.join('/data/domain_adaptation/experiments/slide_puck/baseline/', model_directory)
        errors, ctx = get_errors(full_path, context_sizes)
        plt.plot(context_sizes, errors, label='trained with {} ctx'.format(ctx))
    plt.legend()
    plt.xlabel('context size')
    plt.ylabel('test error')
    plt.title('test error vs context size for models trained w/ different context')
    plt.show()