
import argparse
from varyingsim.util.view import push_box_state_to_xyt
import matplotlib.pyplot as plt
import yaml
import torch
import os
from varyingsim.envs.push_box_offset import PushBoxOffset
from varyingsim.util.parsers import parse_dataset, parse_model, parse_env
import numpy as np
import torch.nn.functional as F
from varyingsim.models.vq_vae import VQVAE
from torch.utils.data import DataLoader
from varyingsim.util.learn import create_train_point, select_datas
import pickle

import torch.nn.functional as F

# DEFAULT_LOCATION = '/data/varyingsim/datasets/push_box_se_K_10000_R_20_seed_0.pickle'
# DEFAULT_LOCATION = '/data/varyingsim/datasets/push_box_se_K_100_R_20_seed_1.pickle'
# DEFAULT_LOCATION = '/data/varyingsim/datasets/push_box_se_same_act_K_100_R_200_seed_1.pickle'
DEFAULT_LOCATION = '/data/varyingsim/datasets/cartpole_relabeled_test.pickle'

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

def create_train_point_ordered(datum, R=20, device='cpu'):
    n_batch = R
    B = datum['obs'].shape[0]

    s = datum['obs'].view(-1, datum['obs'].shape[-1])
    a = datum['act'].view(-1, datum['act'].shape[-1])
    sp = datum['obs_prime'].view(-1, datum['obs_prime'].shape[-1])
    fov = datum['fov'].view(-1, datum['fov'].shape[-1])

    ctx_s, ctx_a, ctx_sp, ctx_fov = select_datas(datum, n=n_batch, device=device)

    ret = dict(
        obs=s.view(B * n_batch, -1),
        act=a.view(B * n_batch, -1),
        obs_prime=sp.view(B * n_batch, -1),
        context_obs=ctx_s.view(B * n_batch, 1, -1),
        context_act=ctx_a.view(B * n_batch, 1, -1),
        context_obs_prime=ctx_sp.view(B * n_batch, 1, -1),
        fov=fov.view(B * n_batch, -1),
    )
    return ret

def loss(obs_hat, obs):
    xy = torch.from_numpy(obs[:2]).float()
    theta = torch.tensor([obs[2]]).float()

    xy_hat = torch.from_numpy(obs_hat[:2]).float()
    theta_hat = torch.tensor([obs_hat[2]]).float()

    xy_loss = F.mse_loss(xy_hat, xy)
    theta_loss = torch.mean(1 - torch.cos(theta - theta_hat))
    loss = xy_loss + theta_loss
    return loss.item()

def gen_model(params, model_options, model_state):
    env = parse_env(params)
    model = parse_model(params, model_options, env)
    model.load_state_dict(model_state)
    return model

def regress_latents(loader, pred_model, model, optim, num_iter=1000, print_iter=50):
    load_iter = iter(loader)
    for i in range(num_iter):
        try:
            batch = next(load_iter)
        except:
            load_iter = iter(loader)
            batch = next(load_iter)
        
        datum = create_train_point(batch, n_batch=1, n_context=1, device='cuda')

        y = datum['fov'].to('cuda')
        recon, info = model.encode_decode(datum)
        x = info['z']
        y_hat = pred_model(x)
        optim.zero_grad()
        loss = F.mse_loss(y, y_hat)
        loss.backward()
        optim.step()
        if i % print_iter == 0:
            print(i, loss.item())
    return pred_model

def entropy_matrix(datum, model):
    n = datum['fov'].shape[0]

    _, extra = model.encode_decode(datum)

    heatmap = np.zeros((n, n))

    ps = extra['log_p_s'].exp().detach()

    for i in range(n):
        rep = ps[i].repeat(n, 1)
        heatmap[i] = F.binary_cross_entropy(rep, ps, reduction='none').sum(1).detach().cpu().numpy()
    return heatmap

def avg_heatmap(heatmap, R=20):
    heatmap_t = torch.tensor(heatmap).unsqueeze(0)
    return F.avg_pool2d(heatmap_t, kernel_size=R, stride=R).squeeze(0)

def get_info(exp_dir):
    with open(os.path.join(exp_dir, 'info.pickle'), 'rb') as f:
        info = pickle.load(f)
    return info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default=DEFAULT_LOCATION, help='Directories')
    parser.add_argument('-x', '--exp-dir', required=True)
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('-n', '--n-batch', default=1, type=int)
    parser.add_argument('-c', '--context-size', default=1, type=int)
    args = parser.parse_args()

    # dataset = EpisodicStartEndFovDataset(args.dataset, obs_fn=push_box_state_to_xyt)


    with open(os.path.join(args.exp_dir, 'params.yaml'), 'r') as f:
        params = yaml.load(f)
    with open(os.path.join(args.exp_dir, 'model_options.yaml'), 'r') as f:
        model_options = yaml.load(f)
    with open(os.path.join(args.exp_dir, 'model_35000.pickle'), 'rb') as f:
        model_state = torch.load(f)

    model = gen_model(params, model_options, model_state)
    train_dataset, test_datasets = parse_dataset(params)
    dataset = next(iter(test_datasets.items()))[1]

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    loader_ordered = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    try:
        info = get_info(args.exp_dir)
    except:
        pass

    K = len(dataset)
    R = len(dataset[0]['act'])

    load_iter = iter(loader)
    data = next(load_iter)
    datum = create_train_point(data, n_batch=args.n_batch, n_context=args.context_size, device='cuda')

    ordered_load_iter = iter(loader_ordered)
    ordered_data = next(ordered_load_iter)
    ordered_datum = create_train_point_ordered(ordered_data, R=R, device='cuda')
    # recon, extra = model.encode_decode(ordered_datum)

    # heatmap = entropy_matrix(ordered_datum, model)
    # heatmap_avg = avg_heatmap(heatmap, R=R)