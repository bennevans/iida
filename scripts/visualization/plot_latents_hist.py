
import argparse
from varyingsim.datasets.fov_dataset import EpisodicStartEndFovDataset
from varyingsim.util.view import push_box_state_to_xyt
import matplotlib.pyplot as plt
import yaml
import torch
import os
from varyingsim.envs.push_box_offset import PushBoxOffset
from varyingsim.util.parsers import parse_model
import numpy as np

# DEFAULT_LOCATION = '/data/varyingsim/datasets/push_box_se_K_10000_R_20_seed_0.pickle'
DEFAULT_LOCATION = '/data/varyingsim/datasets/push_box_se_K_100_R_20_seed_1.pickle'
N_LATENT = 20

# TODO: allow for selecting different latent

def make_datum(episode, idx, context_idx):
    datum = dict(obs=episode['obs'][idx],
            act=episode['act'][idx],
            obs_prime=episode['obs_prime'][idx],
            fov=episode['fov'][idx],
            context_obs=np.expand_dims(episode['obs'][context_idx], axis=0),
            context_act=np.expand_dims(episode['act'][context_idx], axis=0),
            context_obs_prime=np.expand_dims(episode['obs_prime'][context_idx], axis=0))
    return datum

def torchify(datum, device):
    new_datum = {}
    for k, v in datum.items():
        new_datum[k] = torch.from_numpy(v).float().to(device).unsqueeze(0)
    return new_datum
    

def visualize(args):
    scale= 10.0
    dataset = EpisodicStartEndFovDataset(args.dataset, obs_fn=push_box_state_to_xyt)

    exp_name = os.path.basename(args.exp_dir)

    try:
        with open(os.path.join(args.exp_dir, 'params.yaml'), 'r') as f:
            params = yaml.load(f)
        with open(os.path.join(args.exp_dir, 'model_options.yaml'), 'r') as f:
            model_options = yaml.load(f)
        with open(os.path.join(args.exp_dir, 'model.pickle'), 'rb') as f:
            model_state = torch.load(f)
    except Exception as e:
         raise e

    env = PushBoxOffset()
    model = parse_model(params, model_options, env)
    model.load_state_dict(model_state)

    latents = []

    if args.episode_idx is None: # show latent distribution for all data
        for i in range(len(dataset)):
            episode = dataset[i]
            for r in range(episode['obs'].shape[0]):
                if not args.ctx_idx:
                    ctx_idx = np.random.randint(len(episode))
                else:
                    ctx_idx = args.ctx_idx
                datum = make_datum(episode, r, ctx_idx)
                datum = torchify(datum, 'cpu')
                model_hat, z_e, emb, argmin = model.encode_decode(datum)
                latents.append(argmin.item())
    else:
        episode = dataset[args.episode_idx]
        if not args.ctx_idx:
            ctx_idx = np.random.randint(len(episode))
        else:
            ctx_idx = args.ctx_idx
        
        if args.run_idx: # do all runs
            datum = make_datum(episode, args.run_idx, ctx_idx)
            datum = torchify(datum, 'cpu')
            model_hat, z_e, emb, argmin = model.encode_decode(datum)
            latents.append(argmin.item())

        else:
            for r in range(episode['obs'].shape[0]):
                if not args.ctx_idx:
                    ctx_idx = np.random.randint(len(episode))
                else:
                    ctx_idx = args.ctx_idx
                datum = make_datum(episode, r, ctx_idx)
                datum = torchify(datum, 'cpu')
                model_hat, z_e, emb, argmin = model.encode_decode(datum)
                latents.append(argmin.item())

    if args.run_idx:
        plt.title('{} latents episode:{} run: {}'.format(exp_name, args.episode_idx, args.run_idx)) # TODO put context idx and latent idx
    elif args.episode_idx:
        plt.title('{} latents episode:{}'.format(exp_name, args.episode_idx)) # TODO put context idx and latent idx
    else:
        plt.title('{} latents'.format(exp_name))

    counts, bins = np.histogram(latents, bins=np.arange(N_LATENT+1))
    bins = np.arange(N_LATENT)

    plt.bar(bins, counts)

    if args.save_dir:
        plt.savefig(args.save_dir)
    else:
        plt.show()

    print(np.unique(latents))
    if model:
        return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default=DEFAULT_LOCATION, help='Directories')
    parser.add_argument('-x', '--exp-dir', required=True)
    parser.add_argument('-e', '--episode-idx', default=None, type=int)
    parser.add_argument('-r', '--run-idx', default=None, type=int)
    parser.add_argument('-c', '--ctx-idx', default=None, type=int)
    parser.add_argument('-s', '--save-dir', type=str)
    args = parser.parse_args()

    dataset = EpisodicStartEndFovDataset(args.dataset, obs_fn=push_box_state_to_xyt)

    model = visualize(args)

