from varyingsim.util.learn import create_train_point
from varyingsim.util.parsers import parse_model
from varyingsim.envs.push_box_offset import PushBoxOffset
import os
import yaml
import torch
from varyingsim.util.view import push_box_state_to_xyt
from varyingsim.datasets.fov_dataset import EpisodicStartEndFovDataset
from torch.utils.data import DataLoader
from varyingsim.util.learn import create_train_point
import matplotlib.pyplot as plt
# train data
device = 'cuda'
data_dir = '/data/varyingsim/datasets/push_box_se_same_act_split_train_K_100_R_2000_KT_90_RT_1000_seed_0.pickle'
dataset = EpisodicStartEndFovDataset(data_dir, obs_fn=push_box_state_to_xyt)
loader = DataLoader(dataset)
data = next(iter(loader))
datum = create_train_point(data, n_batch=1, n_context=1, device=device)

# if we want the same context as prediction point
datum['context_obs'][:,:] = datum['obs']
datum['context_act'][:,:] = datum['act']
datum['context_obs_prime'][:,:] = datum['obs_prime']



exp_dir = '/data/domain_adaptation/experiments/push_box/baseline/lvebm-2021-05-19_00-35-44'

model_options_file = os.path.join(exp_dir, 'model_options.yaml')
params_file = os.path.join(exp_dir, 'params.yaml')

with open(model_options_file,'r') as f:
    model_options = yaml.load(f)
with open(params_file,'r') as f:
    options = yaml.load(f)

env = PushBoxOffset()

model = parse_model(options, model_options, env)

model_names = ['model_0.pickle', 'model_10000.pickle', 'model_30000.pickle', 'model.pickle']

idxs = torch.where(data['obs'].to(device) ==  datum['obs'])
chosen_idx = idxs[1][0]
all_loader = DataLoader(dataset, batch_size=len(dataset))
all_data = next(iter(all_loader))
s_true = all_data['obs_prime'][:, chosen_idx]

s_prime = datum['obs_prime'][0]

print('chosen_idx', chosen_idx)

for model_name in model_names:
    model_full = os.path.join(exp_dir, model_name)
    with open(model_full, 'rb') as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)

    N_rep = 20
    x = torch.cat([datum['obs'], datum['act']], dim=-1)
    x_rep = x.repeat(N_rep, 1)

    # optimized z spread
    zs = []
    for _ in range(N_rep):
        z, info = model.optim_dcem(datum)
        zs.append(z)
    optim_zs = torch.cat(zs)
    s_hat_optim = model.evaluate(x_rep, optim_zs)

    # unit gaussian z spread
    z = torch.randn(N_rep, model.z_dim, device=device)
    s_hat_unit = model.evaluate(x_rep, z)

    plt.title('true and predicted spread model: {} data_idx: {}'.format(model_name.split('.')[0], chosen_idx.item()))
    plt.scatter(s_hat_optim.cpu().detach().numpy()[:, 0], s_hat_optim.cpu().detach().numpy()[:, 1], label='optim spread')
    plt.scatter(s_hat_unit.cpu().detach().numpy()[:, 0], s_hat_unit.cpu().detach().numpy()[:, 1], label='unit spread')
    plt.scatter(s_true.cpu().detach().numpy()[:, 0], s_true.cpu().detach().numpy()[:, 1], label='true spread')
    plt.scatter(s_prime.cpu().detach().numpy()[0], s_prime.cpu().detach().numpy()[1], label='true next state')
    plt.legend()
    plt.show()