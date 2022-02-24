import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import yaml
from varyingsim.util.parsers import parse_model
from varyingsim.envs.dummy_env import DummyEnv
from varyingsim.envs.push_box_offset import PushBoxOffset
from varyingsim.util.learn import create_train_point
from varyingsim.datasets.toy_dataset import ToyDataset
from varyingsim.util.view import push_box_state_to_xyt
from varyingsim.datasets.fov_dataset import EpisodicStartEndFovDataset
import matplotlib.pyplot as plt

from copy import deepcopy

TOY = False

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage:', sys.argv[0], '<model directory>')
        asdf()

    model_directory = sys.argv[1]

    with open(os.path.join(model_directory, 'model_0.pickle'),'rb') as f:
        state_dict = torch.load(f)
    with open(os.path.join(model_directory, 'params.yaml'),'r') as f:
        params = yaml.load(f)
    with open(os.path.join(model_directory, 'model_options.yaml'),'r') as f:
        model_options = yaml.load(f)

    if TOY:
        env = DummyEnv(1,2,0)
    else:
        env = PushBoxOffset()

    model = parse_model(params, model_options, env)
    model.load_state_dict(state_dict)

    if TOY:
        location = '/data/varyingsim/datasets/toy_2_test.pickle'
        # location = '/data/varyingsim/datasets/toy.pickle'
        dataset = ToyDataset(location)
    else:
        location = '/data/varyingsim/datasets/push_box_se_same_act_split_train_K_100_R_2000_KT_90_RT_1000_seed_0.pickle'
        dataset = EpisodicStartEndFovDataset(location, obs_fn=push_box_state_to_xyt)
    
    loader = DataLoader(dataset, shuffle=True)
    batch = next(iter(loader))
    data = create_train_point(batch, n_batch=128, n_context=32, device='cuda')

def plot():
    xs = data['obs']
    ys = data['obs_prime']
    y_hat = model(data)
    xs = data['obs'].cpu().numpy()
    ys = data['obs_prime'].cpu().numpy()
    y_hat = y_hat.cpu().detach().numpy()
    plt.scatter(xs[:,0], xs[:, 1], label='x')
    plt.scatter(ys[:,0], ys[:, 1], label='y')
    plt.scatter(y_hat[:,0], y_hat[:, 1], label='y_hat')
    plt.legend()
    plt.show()

def plot_zs():
    loader = DataLoader(dataset, shuffle=True)
    for i, batch in enumerate(loader):
        if i == 10:
            break
        data = create_train_point(batch, n_batch=128, n_context=1, device='cuda')
        z, _ = model.optim_dcem(data)
        pts = z.detach().cpu().numpy()
        plt.scatter(pts, [data['fov'][0].item()] * len(pts), label='theta: {}'.format(data['fov'][0].item()))
    plt.legend()
    plt.xlabel('z')
    plt.ylabel('theta')
    plt.title('found z vs theta')
    plt.show()

def scatter_preds():
    # show the possible range of s' for various zs
    loader = DataLoader(dataset, shuffle=True)
    num_scatter = 10

    all_zs = []
    for i, batch in enumerate(loader):
        if i == num_scatter:
            break
        data = create_train_point(batch, n_batch=128, n_context=1, device='cuda')
        z, _ = model.optim_dcem(data)
        all_zs.append(z)
    
    data = create_train_point(batch, n_batch=1, n_context=1, device='cuda')
    print(data['obs'].shape)
    plt.scatter(data['obs'][0,0].cpu().numpy(), data['obs'][0,1].cpu().numpy())

    xs, ys = [], []
    for i, zs in enumerate(all_zs):
        if i == num_scatter:
            break
        x = torch.cat([data['obs'], data['act']], dim=-1)
        z = torch.unsqueeze(zs[0], 0)
        recon = model.evaluate(x, z)
        xy = recon.detach().cpu().numpy()
        xs.append(xy[0,0])
        ys.append(xy[0,1])
    plt.scatter(xs, ys)
    plt.show()



def train_2():
    my_model = nn.Sequential(nn.Linear(3, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2)).to('cuda')
    my_optim = torch.optim.Adam(my_model.parameters(), lr=1e-3)
    for i in range(1000):
        data = create_train_point(batch, n_batch=32, n_context=1, device='cuda')
        my_optim.zero_grad()
        recon = my_model(torch.cat([data['obs'], data['act']], dim=-1))
        loss = F.mse_loss(recon, data['obs_prime'])
        loss.backward()
        my_optim.step()
        print(i, loss.item())

def test_num_ctx(max_ctx_size=1, batch_size=1):
    data = create_train_point(batch, n_batch=batch_size, n_context=max_ctx_size, device='cuda')
    mult_zs = []
    single_zs = []
    for i in range(1, max_ctx_size+1):
        data_cpy = deepcopy(data)
        data_cpy['context_act'] = data_cpy['context_act'][:, :i]
        data_cpy['context_obs'] = data_cpy['context_obs'][:, :i]
        data_cpy['context_obs_prime'] = data_cpy['context_obs_prime'][:, :i]
        z, _ = model.optim_dcem(data_cpy)
        mult_zs.append(z)

        data_cpy = deepcopy(data)
        data_cpy['context_act'] = data_cpy['context_act'][:, i-1:i]
        data_cpy['context_obs'] = data_cpy['context_obs'][:, i-1:i]
        data_cpy['context_obs_prime'] = data_cpy['context_obs_prime'][:, i-1:i]
        z, _ = model.optim_dcem(data_cpy)
        single_zs.append(z)
    return mult_zs, single_zs, data