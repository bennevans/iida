
from varyingsim.datasets.robot_push_dataset import RobotPushDataset
import numpy as np
from varyingsim.util.learn import torchify
from varyingsim.util.parsers import parse_model
from varyingsim.envs.dummy_env import DummyEnv
import yaml
import os
import torch

from torch.utils.data import DataLoader
import torch.nn.functional as F

num_ctx = 1

train_location = "/data/sliding/train"
test_location = "/data/sliding/test"
train_dataset = RobotPushDataset(train_location)
test_dataset = RobotPushDataset(test_location, num_ctx_fn= lambda x: num_ctx)

# model_location = "/data/domain_adaptation/experiments/sliding/baseline/sliding_transformer-2021-09-14_17-56-17"
# model_location = "/data/domain_adaptation/experiments/sliding/baseline/sliding_rnn-2021-09-14_17-54-22"
model_location = "/data/domain_adaptation/experiments/sliding/baseline/sliding_continuous-2021-09-14_17-51-39"


with open(os.path.join(model_location, 'params.yaml'), 'r') as f:
    params = yaml.load(f)

with open(os.path.join(model_location, 'model_options.yaml'), 'r') as f:
    model_options = yaml.load(f)

model_state = torch.load(os.path.join(model_location, 'model.pickle'))

env = DummyEnv(2,2,0)
model = parse_model(params, model_options, env)
model.load_state_dict(model_state)

losses = []

for num_ctx in [0, 1, 2, 4, 8, 16]:
    test_dataset = RobotPushDataset(test_location, num_ctx_fn= lambda x: num_ctx)
    loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)
    batch = next(iter(loader))
    if num_ctx == 0:
        batch['context_obs'] = torch.zeros(160,1,2)
        batch['context_act'] = torch.zeros(160,1,2)
        batch['context_obs_prime'] = torch.zeros(160,1,2)
    y_hat = model(batch)
    loss = F.mse_loss(y_hat.cpu(), batch['obs_prime'])    
    losses.append(loss.item())
    print(num_ctx, loss.item())

