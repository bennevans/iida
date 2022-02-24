
# TODO: also test robot training

import argparse
import itertools
import os
import yaml
import sys

from varyingsim.util.configs import add_default_options
from varyingsim.util.learn import train
from varyingsim.util.parsers import parse_env, parse_model, parse_dataset


def parse_args(env, train_dataset_loc, dataset_type, model_config):
    return p.parse_args(['--env', env, '-d', train_dataset_loc, '-t', dataset_type, 
    '--model-config', model_config, '--test-datasets', train_dataset_loc, '--test-names', 'test'])

def override_args(args):
    args.device = 'cuda'
    args.epochs = 1
    args.save_iter = 1
    args.no_wandb = True
    args.output = '/tmp'
    args.use_obs_fn = True

def run(args):
    with open(args.model_config, 'r') as f:
        model_options = yaml.load(f, Loader=yaml.Loader)

    train_dataset, test_datasets = parse_dataset(args)
    env = parse_env(args)
    model = parse_model(args, model_options, env)

    model, info = train(train_dataset, args, model, '/tmp', None, None, test_datasets=test_datasets)
    return model, info

if __name__ == '__main__':
    # simulated envs
    sim_envs = ['Hopper', 'Humanoid', 'PushBoxOffset', 'SlidePuck', 'Swimmer']
    # sim_envs = ['PushBoxOffset']
    dataset_locs = [
        '/data/varyingsim/datasets/training_hopper_no_variation_relabeled_train.pickle',
        '/home/bne215/projects/varyingsim/scripts/policy/data/training_humanoid_no_variation_relabeled_train.pickle',
        '/data/varyingsim/datasets/push_box_se_same_act_split_train_K_100_R_2000_KT_90_RT_1000_seed_0.pickle',
        '/data/varyingsim/datasets/slide_puck_K_18_R_20_seed_0_same_act_robot_train.pickle',
        '/home/bne215/projects/varyingsim/scripts/policy/data/training_swimmer_no_variation_relabeled_train.pickle'
    ]
    # dataset_locs = [
    #     '/data/varyingsim/datasets/push_box_se_same_act_split_train_K_100_R_2000_KT_90_RT_1000_seed_0.pickle',
    # ]

    dataset_types = [
        'RelabeledEpisodicFovDataset',
        'RelabeledEpisodicFovDataset',
        'EpisodicStartEndFovDataset',
        'EpisodicStartEndFovDataset', 
        'RelabeledEpisodicFovDataset'
    ]
    # dataset_types = ['EpisodicStartEndFovDataset']

    model_config_prefix = '../scripts/model_configs/'
    model_config_suffix = '.yaml'
    model_configs = ['feed_forward', 'cont_latent', 'feed_forward_fov', 'rnn_latent', 'transformer_latent']
    model_configs = ['feed_forward']

    p = argparse.ArgumentParser()
    add_default_options(p)

    failures = []
    exceptions = []

    for env_idx, model in itertools.product(range(len(sim_envs)), model_configs):
        env = sim_envs[env_idx]
        dataset_loc = dataset_locs[env_idx]
        dataset_type = dataset_types[env_idx]

        model_config = os.path.join(model_config_prefix, '{}{}'.format(model, model_config_suffix))

        args = parse_args(env, dataset_loc, dataset_type, model_config)
        override_args(args)

        print(env, model)

        try:
            # sys.stdout = open(os.devnull, 'w')
            run(args)
            sys.stdout = sys.__stdout__
        except Exception as e:
            sys.stdout = sys.__stdout__
            print('Run failed! Env: {} Model: {}'.format(env, model))
            failures.append((env, model))
            exceptions.append(e)
    
    print('failed runs:')
    for env, model in failures:
        print(env, model)
    print('raising latest error')
    raise(exceptions[-1])
    
