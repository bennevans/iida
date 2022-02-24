
from varyingsim.datasets.fov_dataset import EpisodicStartEndFovDataset
import numpy as np
from varyingsim.util.learn import torchify
from varyingsim.util.parsers import parse_model
from varyingsim.envs.slide_puck import SlidePuck
import yaml
import os
import torch
import torch.nn.functional as F

train_location = "/data/varyingsim/datasets/slide_puck_K_1000_R_10_seed_0_act_std_train.pickle"
test_location = "/data/varyingsim/datasets/slide_puck_K_100_R_10_seed_0_act_std_test.pickle"
train_dataset = EpisodicStartEndFovDataset(train_location)
test_dataset = EpisodicStartEndFovDataset(test_location)

model_location = "/data/domain_adaptation/experiments/slide_puck/baseline/slidepuck_rnn-2021-09-13_19-23-13"
# model_location = "/data/domain_adaptation/experiments/slide_puck/baseline/slidepuck_continuous-2021-09-13_19-18-58"

with open(os.path.join(model_location, 'params.yaml'), 'r') as f:
    params = yaml.load(f)

with open(os.path.join(model_location, 'model_options.yaml'), 'r') as f:
    model_options = yaml.load(f)

model_state = torch.load(os.path.join(model_location, 'model.pickle'))

env = SlidePuck()
model = parse_model(params, model_options, env)
model.load_state_dict(model_state)


test_idx = np.random.randint(len(test_dataset))
print(test_idx)
test_points = test_dataset[test_idx]

num_ctx = 8

context_idxs = np.random.randint(len(test_points['obs']), size=num_ctx)

test_datum = {
    'context_obs': test_points['obs'][context_idxs].reshape(1, num_ctx, -1),
    'context_act': test_points['act'][context_idxs].reshape(1, num_ctx, -1),
    'context_obs_prime': test_points['obs_prime'][context_idxs].reshape(1, num_ctx, -1)
}

test_datum = torchify(test_datum)

test_latent = model.encoder(test_datum)

dists = []
fov_dists = []
# calculate the latents for all train envs:
N_TRAIN_ENVS = len(train_dataset)
for i in range(N_TRAIN_ENVS):
    train_points = train_dataset[test_idx]
    context_idxs = np.random.randint(len(train_points['obs']), size=num_ctx)
    train_datum = {
        'context_obs': train_points['obs'][context_idxs].reshape(1, num_ctx, -1),
        'context_act': train_points['act'][context_idxs].reshape(1, num_ctx, -1),
        'context_obs_prime': train_points['obs_prime'][context_idxs].reshape(1, num_ctx, -1)
    }
    train_datum = torchify(train_datum)

    env_z = model.encoder(train_datum)
    dists.append(torch.norm(env_z - test_latent).item())
    fov_dists.append(np.linalg.norm(train_points['fov'][0] - test_points['fov'][0]))

closest = np.argmin(dists)
closest_fov = np.argmin(fov_dists)

print(closest, closest_fov)

# closest_data = train_dataset[closest]
closest_data = train_dataset[closest_fov]

closest_fovs = closest_data['fov'][0]
env.set_fovs(closest_fovs.tolist())
env.sim.set_constants()
env.reset()

id_end_obss = []

for i, action in enumerate(closest_data['act']):
    env.reset()
    env.set_state(closest_data['obs'][i][:9], closest_data['obs'][i][9:-1])
    env.step(action)
    act_cont = action.copy()
    act_cont[-1] = 0.0
    obs =[0]
    while obs[-1] != 1:
        obs, rew, done, info = env.step(act_cont)

    id_end_obss.append(obs)

end = torch.tensor(closest_data['obs_prime'])
end_hat = torch.tensor(id_end_obss)
loss = F.mse_loss(end_hat, end)
print("closest train env", closest)
print(loss.item())

# rand_idxs = np.random.randint(len(train_dataset), size=20)
# losses = []
# for num, idx in enumerate(rand_idxs): #range(len(train_dataset)):
#     cur_env = train_dataset[idx]
#     cur_fovs = cur_env['fov'][0]
#     env.set_fovs(cur_fovs.tolist())
#     env.sim.set_constants()
#     env.reset()

#     # print(num, end=' ')
#     # print()

#     other_end_obss = []
#     for i, action in enumerate(cur_env['act']):
#         env.reset()
#         env.set_state(cur_env['obs'][i][:9], cur_env['obs'][i][9:-1])
#         env.step(action)
#         act_cont = action.copy()
#         act_cont[-1] = 0.0
#         obs =[0]
#         while obs[-1] != 1:
#             obs, rew, done, info = env.step(act_cont)

#         other_end_obss.append(obs)

#     end = torch.tensor(cur_env['obs_prime'])
#     end_hat = torch.tensor(other_end_obss)
#     loss = F.mse_loss(end_hat, end)
#     losses.append(loss)

# print("mean train set")
# print(np.mean(losses))