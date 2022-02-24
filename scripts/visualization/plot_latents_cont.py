
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
from sklearn.decomposition import PCA
from varyingsim.models.vq_vae import VQVAE

# DEFAULT_LOCATION = '/data/varyingsim/datasets/push_box_se_K_10000_R_20_seed_0.pickle'
DEFAULT_LOCATION = '/data/varyingsim/datasets/push_box_se_K_100_R_20_seed_1.pickle'
N_LATENT = 20

mode_VQVAE = False

np.random.seed(5)

cmap = {}
for i in range(N_LATENT):
    cmap[i] = (np.random.random(), np.random.random(), np.random.random())

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
    

def get_colors(argmins):
    ret = []
    if type(argmins) == list or type(argmins) == np.ndarray:
        for c in argmins:
            ret.append(cmap[c[0]])
        return ret
    return cmap[argmins]            

def visualize(args):
    scale= 10.0
    dataset = EpisodicStartEndFovDataset(args.dataset, obs_fn=push_box_state_to_xyt)

    exp_name = os.path.basename(args.exp_dir)
    cmap = plt.get_cmap('plasma')

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

    mode_VQVAE = (type(model) == VQVAE)

    model.load_state_dict(model_state)

    latents = []
    assignments = []

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
                if mode_VQVAE:
                    model_hat, z_e, emb, argmin = model.encode_decode(datum) # TODO: code to check if we're using vqvae
                    latents.append(z_e)
                    assignments.append(argmin)
                else:
                    z = model.encoder(datum)
                    latents.append(z)
    else:
        episode = dataset[args.episode_idx]
        if not args.ctx_idx:
            ctx_idx = np.random.randint(len(episode))
        else:
            ctx_idx = args.ctx_idx
        
        if args.run_idx: # do all runs
            datum = make_datum(episode, args.run_idx, ctx_idx)
            datum = torchify(datum, 'cpu')
            if mode_VQVAE:
                model_hat, z_e, emb, argmin = model.encode_decode(datum)
                latents.append(z_e)
                assignments.append(argmin)
            else:
                z = model.encoder(datum)
                latents.append(z)

        else:
            for r in range(episode['obs'].shape[0]):
                if not args.ctx_idx:
                    ctx_idx = np.random.randint(len(episode))
                else:
                    ctx_idx = args.ctx_idx
                datum = make_datum(episode, r, ctx_idx)
                datum = torchify(datum, 'cpu')
                if mode_VQVAE:
                    model_hat, z_e, emb, argmin = model.encode_decode(datum)
                    latents.append(z_e)
                    assignments.append(argmin)
                else:
                    z = model.encoder(datum)
                    latents.append(z)

    if args.run_idx:
        plt.title('{} latents episode:{} run: {}'.format(exp_name, args.episode_idx, args.run_idx)) # TODO put context idx and latent idx
    elif args.episode_idx:
        plt.title('{} latents episode:{}'.format(exp_name, args.episode_idx)) # TODO put context idx and latent idx
    else:
        plt.title('{} latents'.format(exp_name))

    latents_np = np.concatenate([l.cpu().detach().numpy() for l in latents])
    pca = PCA()
    pca.fit(latents_np)
    low_dim = pca.transform(latents_np)

    if mode_VQVAE:
        argmin_np = np.array([a.cpu().detach().numpy() for a in assignments])
        colors = get_colors(argmin_np)

        plt.scatter(low_dim[:, args.xaxis], low_dim[:, args.yaxis], c=colors)
    else:
        plt.scatter(low_dim[:, args.xaxis], low_dim[:, args.yaxis])
    
    plt.xlabel("principal component {}".format(args.xaxis))
    plt.ylabel("principal component {}".format(args.yaxis))

    if mode_VQVAE:
        unique_latents = np.unique(argmin_np)
        latent_locs = model.emb.weight[:, unique_latents].T.cpu().detach().numpy()
        latent_locs_low = pca.transform(latent_locs)

        for i, u in enumerate(unique_latents):
            color = get_colors(u)
            plt.scatter(latent_locs_low[i, 0], latent_locs_low[i, 0],
                label='latent center {}'.format(u), marker='*', s=100, c=color, edgecolors=(0,0,0.))

    plt.legend()
    if args.save_dir:
        plt.savefig(args.save_dir)
    else:
        plt.show()

    if args.var:
        plt.title('explained variance ratio')
        plt.plot(pca.explained_variance_ratio_)
        plt.show()

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
    parser.add_argument('-a', '--yaxis', type=int, default=1)
    parser.add_argument('-b', '--xaxis', type=int, default=0)
    parser.add_argument('-v', '--var', default=False, action='store_true')
    args = parser.parse_args()

    dataset = EpisodicStartEndFovDataset(args.dataset, obs_fn=push_box_state_to_xyt)

    model = visualize(args)

