import os
import pickle
import yaml
from datetime import datetime, timedelta
import time
from varyingsim.util.configs import get_default_parser

from varyingsim.util.learn import train
from varyingsim.util.parsers import parse_env, parse_model, parse_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
import wandb
import socket


def run(options, model_options, exp_path, sess=None):
    train_dataset, test_datasets, val_dataset = parse_dataset(options)
    env = parse_env(options)
    model = parse_model(options, model_options, env)

    if not options.no_wandb:
        wandb.config.update(dict(model_options=model_options))
        wandb.watch(model)
        print('run name:', wandb.run.name)
        print('host name', socket.gethostname())
        
    
    model, info = train(train_dataset, options, model, exp_path, sess, wandb, \
        test_datasets=test_datasets, val_dataset=val_dataset)
    return model, info

def run_exp(options):
    exp_dir = options.name + datetime.today().strftime('-%Y-%m-%d_%H-%M-%S') + "_" + str(options.seed)
    exp_path = os.path.join(options.output, exp_dir)
    os.mkdir(exp_path)

    np.random.seed(options.seed)
    torch.manual_seed(options.seed)

    print('running at', exp_path)
    print('pwd', os.getcwd())

    full_path = os.path.join(os.getcwd(), options.model_config)

    with open(full_path, 'r') as f:
        model_options = yaml.load(f, yaml.Loader)

    # copy params into exp directory
    with open(os.path.join(exp_path, 'params.yaml'), 'w') as f:
        yaml.dump(options, f)
    with open(os.path.join(exp_path, 'model_options.yaml'), 'w') as f:
        yaml.dump(model_options, f)

    print('options')
    for k, v in vars(options).items():
        print('\t{}: {}'.format(k, v))
    print()
    print('model options')
    for k, v in model_options.items():
        print('\t{}: {}'.format(k, v))
    print()

    if not options.no_wandb:
        wandb.init(project=options.project, group=options.group)
        wandb.config.update(options)
        wandb.config.update(dict(exp_dir=exp_dir, slurm_job_id=os.environ.get('SLURM_JOB_ID')))

    try:
        model, info = run(options, model_options, exp_path, sess=None)
    except Exception as e:
        raise e

    with open(os.path.join(exp_path, 'info.pickle'), 'wb') as f:
        pickle.dump(info, f)

    with open(os.path.join(exp_path, 'model.pickle'), 'wb') as f:
        # pickle.dump(algo, f)
        model.save(f)
    
    return exp_dir

if __name__ == '__main__':
    p = get_default_parser()

    options = p.parse_args()
    start_time = time.time()

    try:
        exp_dir = run_exp(options)
    except Exception as e:
        end_time = time.time()
        run_time = end_time - start_time
        print('runtime: {}'.format(timedelta(seconds=run_time)))
        raise e

    end_time = time.time()
    run_time = end_time - start_time
    print('runtime: {}'.format(timedelta(seconds=run_time)))
