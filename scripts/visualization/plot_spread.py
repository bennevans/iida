
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
import torch.nn.functional as F
from varyingsim.models.vq_vae import VQVAE

# DEFAULT_LOCATION = '/data/varyingsim/datasets/push_box_se_K_10000_R_20_seed_0.pickle'
DEFAULT_LOCATION = '/data/varyingsim/datasets/push_box_se_K_100_R_20_seed_1.pickle'


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

def loss(obs_hat, obs):
    xy = torch.from_numpy(obs[:2]).float()
    theta = torch.tensor([obs[2]]).float()

    xy_hat = torch.from_numpy(obs_hat[:2]).float()
    theta_hat = torch.tensor([obs_hat[2]]).float()

    xy_loss = F.mse_loss(xy_hat, xy)
    theta_loss = torch.mean(1 - torch.cos(theta - theta_hat))
    loss = xy_loss + theta_loss
    return loss.item()

def visualize(args):
    scale= 10.0
    dataset = EpisodicStartEndFovDataset(args.dataset, obs_fn=push_box_state_to_xyt)

    exp_name = os.path.basename(args.exp_dir)


    episode = dataset[0]
    start_obs = episode['obs'][args.run_idx]
    end_obs = episode['obs_prime'][args.run_idx]
    print('fovs', episode['fov'][0])
    N_RUN = len(episode['state'])


    start_theta = start_obs[2]
    end_theta = end_obs[2]
    start_vx = np.cos(start_theta) / scale
    start_vy = np.sin(start_theta) / scale
    end_vx = np.cos(end_theta) / scale
    end_vy = np.sin(end_theta) / scale

    plt.scatter(start_obs[0], start_obs[1], label='start position')
    plt.scatter(end_obs[0], end_obs[1], label='end position')
    plt.arrow(start_obs[0], start_obs[1], start_vx, start_vy)
    plt.arrow(end_obs[0], end_obs[1], end_vx, end_vy)

    with open(os.path.join(args.exp_dir, 'params.yaml'), 'r') as f:
        params = yaml.load(f)
    with open(os.path.join(args.exp_dir, 'model_options.yaml'), 'r') as f:
        model_options = yaml.load(f)
    with open(os.path.join(args.exp_dir, 'model.pickle'), 'rb') as f:
        model_state = torch.load(f)


    env = PushBoxOffset()
    model = parse_model(params, model_options, env)
    model.load_state_dict(model_state)
    model_vqvae = (type(model) == VQVAE)
   
    xs = []
    ys = []
    latents = []
    vxs, vys = [], []
    for i in range(N_RUN):
        episode = dataset[i]
        for ctx_idx in range(N_RUN):
            datum = make_datum(episode, args.run_idx, ctx_idx)
            datum = torchify(datum, 'cpu')
            if model_vqvae:
                model_hat, z_e, emb, argmin = model.encode_decode(datum)
            else:
                model_hat = model(datum)

            obs_hat = model_hat.squeeze(0).cpu().detach().numpy()
            xs.append(obs_hat[0])
            ys.append(obs_hat[1])
            theta_hat = obs_hat[2]
            vx = np.cos(theta_hat) / scale
            vy = np.sin(theta_hat) / scale
            vxs.append(vx)
            vys.append(vy)
            if model_vqvae:
                latents.append(argmin)
    if model_vqvae:
        uniq_lat = np.unique([l.item() for l in latents])
    # plt.title('{} episode:{} run: {} latents: {}'.format(exp_name, args.episode_idx, args.run_idx, uniq_lat))
    plt.scatter(xs, ys, label='model end position')

    for x, y, vx, vy in zip(xs, ys, vxs, vys):
        plt.arrow(x, y, vx, vy)
    if model_vqvae:
        print('unique latents', uniq_lat)

    plt.title('{} run: {}'.format(exp_name, args.run_idx)) # TODO put context idx and latent idx
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.legend()
    if args.save_dir:
        plt.savefig(args.save_dir)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default=DEFAULT_LOCATION, help='Directories')
    parser.add_argument('-x', '--exp-dir', required=True)
    parser.add_argument('-r', '--run-idx', default=0, type=int)
    parser.add_argument('-s', '--save-dir', type=str)
    args = parser.parse_args()

    visualize(args)

