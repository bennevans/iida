
import argparse
from varyingsim.datasets.fov_dataset import EpisodicStartEndFovDataset
from varyingsim.util.view import slide_box_state_to_xyt
import matplotlib.pyplot as plt
import yaml
import torch
import os
from varyingsim.envs.slide_puck import SlidePuck
from varyingsim.util.parsers import parse_model
import numpy as np
import torch.nn.functional as F
from varyingsim.models.vq_vae import VQVAE
from plot_spread_slide import make_datum

# DEFAULT_LOCATION = "/data/varyingsim/datasets/slide_puck_K_5_R_4_seed_3735928559_same_act_vis_train.pickle"
DEFAULT_LOCATION = "/data/varyingsim/datasets/slide_puck_K_100_R_10_seed_0_act_std_test.pickle"


# TODO: allow for selecting different latent

# def make_datum(episode, idx, context_idx):
#     datum = dict(obs=episode['obs'][idx],
#             act=episode['act'][idx],
#             obs_prime=episode['obs_prime'][idx],
#             fov=episode['fov'][idx],
#             context_obs=np.expand_dims(episode['obs'][context_idx], axis=0),
#             context_act=np.expand_dims(episode['act'][context_idx], axis=0),
#             context_obs_prime=np.expand_dims(episode['obs_prime'][context_idx], axis=0))
#     return datum

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
    if args.all_episodes:
        visualize_all(args)
    else:
        visualize_single(args)

def visualize_all(args):
# visualizes all state primes in the
    scale= 10.0
    dataset = EpisodicStartEndFovDataset(args.dataset, obs_fn=slide_box_state_to_xyt)

    starts = [episode['obs'][args.run_idx] for episode in dataset]
    if len(np.unique(starts, axis=0)) > 1:
        print('WARNING START STATES ARE NOT THE SAME!')
    acts = [episode['act'][args.run_idx] for episode in dataset]
    if len(np.unique(acts, axis=0)) > 1:
        print('WARNING ACTIONS ARE NOT THE SAME!')

    episode = dataset[0]
    start_obs = episode['obs'][args.run_idx]
    start_theta = start_obs[2]
    start_vx = np.cos(start_theta) / scale
    start_vy = np.sin(start_theta) / scale

    plt.scatter(start_obs[0], start_obs[1], label='start position')
    plt.arrow(start_obs[0], start_obs[1], start_vx, start_vy)

    obs_primes = []

    for i, episode in enumerate(dataset):
        end_obs = episode['obs_prime'][args.run_idx]
        obs_primes.append(end_obs)

        end_theta = end_obs[2]
     
        end_vx = np.cos(end_theta) / scale
        end_vy = np.sin(end_theta) / scale

        plt.scatter(end_obs[0], end_obs[1], label='end position {}'.format(i))
        plt.arrow(end_obs[0], end_obs[1], end_vx, end_vy)
    
    mean_obs_prime = np.mean(obs_primes, axis=0)
    losses = []
    for obs_prime in obs_primes:
        losses.append(loss(mean_obs_prime, obs_prime))
    mean_loss = np.mean(losses)
    print('mean_loss of obs_prime', mean_loss)
    obs_ends = np.array(obs_primes)
    xs = obs_ends[:, 0]
    ys = obs_ends[:, 1]
    min_x = min(-1, np.min(xs))
    max_x = max(1, np.max(xs))
    min_y = min(-1, np.min(ys))
    max_y = max(1, np.max(ys))

    plt.title('rollouts for run {}'.format(args.run_idx))
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.legend()
    if args.save_dir:
        plt.savefig(args.save_dir)
    else:
        plt.show()



def visualize_all_envs(args):
    scale= 10.0
    dataset = EpisodicStartEndFovDataset(args.dataset, obs_fn=slide_box_state_to_xyt)

    starts = [episode['obs'][args.run_idx] for episode in dataset]
    if len(np.unique(starts, axis=0)) > 1:
        print('WARNING START STATES ARE NOT THE SAME!')
    acts = [episode['act'][args.run_idx] for episode in dataset]
    if len(np.unique(acts, axis=0)) > 1:
        print('WARNING ACTIONS ARE NOT THE SAME!')

    episode = dataset[0]
    start_obs = episode['obs'][args.run_idx]
    start_theta = start_obs[2]
    start_vx = np.cos(start_theta) / scale
    start_vy = np.sin(start_theta) / scale

    plt.scatter(start_obs[0], start_obs[1], label='start position')
    plt.arrow(start_obs[0], start_obs[1], start_vx, start_vy)

    obs_primes = []

    for i, episode in enumerate(dataset):
        end_obs = episode['obs_prime'][args.run_idx]
        obs_primes.append(end_obs)

        end_theta = end_obs[2]
     
        end_vx = np.cos(end_theta) / scale
        end_vy = np.sin(end_theta) / scale

        plt.scatter(end_obs[0], end_obs[1], label='end position {}'.format(i))
        plt.arrow(end_obs[0], end_obs[1], end_vx, end_vy)
    
    mean_obs_prime = np.mean(obs_primes, axis=0)
    losses = []
    for obs_prime in obs_primes:
        losses.append(loss(mean_obs_prime, obs_prime))
    mean_loss = np.mean(losses)
    print('mean_loss of obs_prime', mean_loss)
    import ipdb; ipdb.set_trace()
    xlims = torch.min(mean_obs_prime[:, 0])

    plt.title('rollouts for run {}'.format(args.run_idx))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.legend()
    if args.save_dir:
        plt.savefig(args.save_dir)
    else:
        plt.show()


def visualize_single(args):
    scale= 10.0
    dataset = EpisodicStartEndFovDataset(args.dataset, obs_fn=slide_box_state_to_xyt)

    N_EPISODES = len(dataset)
    episode = dataset[args.episode_idx]
    start_obs = episode['obs'][args.run_idx]
    end_obs = episode['obs_prime'][args.run_idx]
    print('fovs', episode['fov'][0])

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

    if args.exp_dir:
        for exp_dir in args.exp_dir:
            exp_name = os.path.basename(exp_dir)

            try:
                with open(os.path.join(exp_dir, 'params.yaml'), 'r') as f:
                    params = yaml.load(f)
                with open(os.path.join(exp_dir, 'model_options.yaml'), 'r') as f:
                    model_options = yaml.load(f)
                # with open(os.path.join(exp_dir, 'model.pickle'), 'rb') as f:
                with open(os.path.join(exp_dir, 'model_10000.pickle'), 'rb') as f:
                    model_state = torch.load(f)
            except Exception as e:
                raise e
                continue 

            env = SlidePuck()
            model = parse_model(params, model_options, env)
            model.load_state_dict(model_state)
            model_vqvae = (type(model) == VQVAE)
            
            if not (args.all_ctx or args.all_lat):
                print('single run')
                if not args.ctx_idx:
                    if args.num_ctx:
                        num_ctx = args.num_ctx
                    else:
                        num_ctx = 8
                    datum = make_datum(episode, args.run_idx, num_ctx=num_ctx)
                else:
                    ctx_idx = args.ctx_idx
                    datum = make_datum(episode, args.run_idx, ctx_idx)
                datum = torchify(datum, model.device)

                if model_vqvae:
                    if args.lat_idx is None:
                        model_hat, z_e, emb, argmin = model.encode_decode(datum)
                    else:
                        model_hat, z_e, emb, argmin = model.encode_decode(datum, set_argmin=[args.lat_idx])
                else:
                    model_hat = model(datum)

                obs_hat = model_hat.squeeze(0).cpu().detach().numpy()
                print('obs_hat')
                print(obs_hat)
                print('obs')
                print(datum['obs_prime'])
                loss = model.loss(datum).item()
                print('loss', loss)

                theta_hat = obs_hat[2]
                vx = np.cos(theta_hat) / scale
                vy = np.sin(theta_hat) / scale

                plt.arrow(obs_hat[0], obs_hat[1], vx, vy)

                if model_vqvae:
                    if type(argmin) is torch.tensor:
                        argmin = argmin.item()
                    else:
                        argmin = argmin[0]
                    label = '{} end position ({})'.format(exp_name, argmin)
                else:
                    label = '{} end position loss: {}'.format(exp_name, loss)

                plt.scatter(obs_hat[0], obs_hat[1], label=label)
                # plt.title('{} episode:{} run: {} context_idx: {} latent: {}'.format(exp_name, args.episode_idx, args.run_idx, ctx_idx, argmin.item())) # TODO put context idx and latent idx

            elif args.all_ctx:
                print('all ctx')
                xs = []
                ys = []
                latents = []
                vxs, vys = [], []
                for i in range(N_EPISODES):
                    datum = make_datum(episode, args.run_idx, i)
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
                    print(uniq_lat)
            elif args.all_lat:
                print('all lat')
                xs = []
                ys = []
                latents = []
                vxs, vys = [], []
                for i in range(N_EPISODES):
                    datum = make_datum(episode, args.run_idx, i)
                    datum = torchify(datum, 'cpu')

                    model_hat, z_e, emb, argmin = model.encode_decode(datum, set_argmin=[i])
                    obs_hat = model_hat.squeeze(0).cpu().detach().numpy()
                    xs.append(obs_hat[0])
                    ys.append(obs_hat[1])
                    theta_hat = obs_hat[2]
                    vx = np.cos(theta_hat) / scale
                    vy = np.sin(theta_hat) / scale
                    vxs.append(vx)
                    vys.append(vy)

                    plt.scatter(obs_hat[0], obs_hat[1], label='model end position ({})'.format(i), c=(0.0, i / N_EPISODES, i / N_EPISODES))

                    latents.append(argmin)

                # plt.title('{} episode:{} run: {} latents: {}'.format(exp_name, args.episode_idx, args.run_idx, uniq_lat))
                # plt.scatter(xs, ys, label='model end position')

                for i, (x, y, vx, vy) in enumerate(zip(xs, ys, vxs, vys)):
                    plt.arrow(x, y, vx, vy)

    plt.title('episode:{} run: {}'.format(args.episode_idx, args.run_idx)) # TODO put context idx and latent idx
    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)

    min_x = min(-1, np.min(obs_hat[0]))
    max_x = max(1, np.max(obs_hat[0]))
    min_y = min(-1, np.min(obs_hat[1]))
    max_y = max(1, np.max(obs_hat[1]))

    plt.title('rollouts for run {}'.format(args.run_idx))
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    plt.legend()
    if args.save_dir:
        plt.savefig(args.save_dir)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default=DEFAULT_LOCATION, help='Directories')
    parser.add_argument('-x', '--exp-dir', default=None, nargs='+')
    parser.add_argument('-e', '--episode-idx', default=0, type=int)
    parser.add_argument('-r', '--run-idx', default=0, type=int)
    parser.add_argument('-c', '--ctx-idx', type=int)
    parser.add_argument('-l', '--lat-idx', type=int)
    parser.add_argument('--all-ctx', action='store_true')
    parser.add_argument('--all-lat', action='store_true')
    parser.add_argument('--num-ctx', type=int, default=1)
    parser.add_argument('-s', '--save-dir', type=str)
    parser.add_argument('--all-episodes', action='store_true')
    args = parser.parse_args()

    dataset = EpisodicStartEndFovDataset(args.dataset, obs_fn=slide_box_state_to_xyt)

    # visualize(args)

