import os
import torch
import yaml

from varyingsim.models.transformer import make_encoder_oh
from varyingsim.models.mujoco_dynamics import MuJoCoDynamics, MuJoCoDynamicsFlat
from varyingsim.models.feed_forward import FeedForward, FeedForwardDatum
from varyingsim.models.osi import OSI, OSIMuJoCo
from varyingsim.models.continuous import SingleEncoder, SimpleDecoder, ContinualLatent, MultipleEncoder, RNNEncoder
from varyingsim.models.vq_vae import VQVAE
from varyingsim.models.proto import GenericEncoder, Proto
from varyingsim.models.variational import PEARLEncoder, VariationalLatent

# env imports
from varyingsim.envs.push_box_offset import PushBoxOffset
from varyingsim.envs.dummy_env import DummyEnv
from varyingsim.envs.slide_puck import SlidePuck
from varyingsim.envs.cartpole import CartpoleEnv
from varyingsim.envs.hopper import HopperEnv
from varyingsim.envs.swimmer import SwimmerEnv
from varyingsim.envs.humanoid import HumanoidEnv


from varyingsim.util.view import obs_to_relative_torch, push_box_state_to_xyt, \
    slide_box_state_to_xyt, slide_box_state_to_xyt_velocity, slide_box_state_to_xycs, \
    hopper_state_to_obs, swimmer_state_to_obs, slide_box_state_to_xy, push_box_state_to_xycs

# dataset imports
from varyingsim.datasets.toy_dataset import ToyDataset
from varyingsim.datasets.fov_dataset import EpisodicStartEndFovDataset, SmoothFovDataset, SmoothFovDatasetGenerator, StartEndDataset,  EpisodicStartEndDataset
from varyingsim.datasets.relabeled_dataset import RelabeledEpisodicFovDataset
from varyingsim.datasets.robot_push_dataset import RobotPushDataset
from varyingsim.datasets.ur_robot_push_dataset import URRobotPushDataset

def parse_env(options):
    if options.env == 'PushBoxOffset':
        return PushBoxOffset()
    if options.env == 'DummyEnv':
        return DummyEnv(options.env_nu, options.env_nq, options.env_nv, n_fov=4)
    if options.env == 'SlidePuck':
        return SlidePuck()
    if options.env == 'CartPole':
        return CartpoleEnv(mode=CartpoleEnv.SWINGUP)
    if options.env == 'Hopper':
        return HopperEnv()
    if options.env == 'Swimmer':
        return SwimmerEnv()
    if options.env == 'Humanoid':
        return HumanoidEnv()
    else:
        raise Exception('unrecognized env ' + options.env)

def parse_model(options, model_options, env):
    if options.use_obs_fn and not type(env) == DummyEnv:
        if options.env == 'PushBoxOffset':
            d_obs = 4 # TODO: handle xyt later
        if options.env == 'SlidePuck':
            if 'obs_fn' in dir(options):
                if options.obs_fn == 'slide_box_state_to_xyt':
                    d_obs = 3
                elif options.obs_fn == 'slide_box_state_to_xyt_velocity':
                    d_obs = 6
                elif options.obs_fn == 'slide_box_state_to_xycs':
                    d_obs = 4
                elif options.obs_fn == 'slide_box_state_to_xy':
                    d_obs = 2
                else:
                    d_obs = 3
            else:
                d_obs = 3
        elif options.env == 'Hopper':
            d_obs = 11
        elif options.env == 'Swimmer':
            d_obs = 8
        elif options.env == 'Humanoid':
            d_obs = 47
    else:
        d_obs = len(env.reset())

    print('d_obs', d_obs)

    if model_options['model'] == 'OSI':
        context_size = options.context_size
        d_fov = model_options['fov_dim']
        d_in = d_obs + env.model.nu
        
        model = FeedForward(d_in + d_fov, d_obs, model_options['hidden_dynamics']).to(options.device)

        mujoco_model = MuJoCoDynamics(env, model).to(options.device)
        osi_model = OSI(context_size, d_in, d_fov, model_options['d_share'], 
            model_options['hidden_shared'], model_options['hidden_osi']).to(options.device)

        if 'relative_obs' in model_options and model_options['relative_obs']: 
            obs_transform = obs_to_relative_torch
        else:
            obs_transform = obs_to_relative_torch

        osi_mujoco = OSIMuJoCo(env, osi_model, mujoco_model, context_size, device=options.device, obs_transform=obs_transform)
        
        return osi_mujoco
    elif model_options['model'] == 'VQVAE':
        context_size = options.context_size
        d_act = env.model.nu
        
        d_in = 2 * d_obs + d_act 
        d = model_options['embedding_vector_dim']
        k = model_options['discrete_latent_dim']
        vq_coef = model_options['vq_coef']
        commit_coef = model_options['commit_coef']
        encoder_hidden_sizes = model_options['encoder_hidden_sizes']
        decoder_hidden_sizes = model_options['decoder_hidden_sizes']
        ee_coef = float(model_options['ee_coef'])
        eq_coef = float(model_options['eq_coef'])
        
        encoder = SingleEncoder(env, context_size, d_in, d, encoder_hidden_sizes, device=options.device).to(options.device)
        decoder = SimpleDecoder(env, context_size, d + d_obs + d_act, d_obs, decoder_hidden_sizes, device=options.device).to(options.device)
        model = VQVAE(env, context_size, encoder, decoder, k, d, vq_coef=vq_coef, commit_coef=commit_coef,
                        ee_coef=ee_coef, eq_coef=eq_coef, device=options.device).to(options.device)
        return model
    elif model_options['model'] == 'FeedForward':
        d_act = env.model.nu
        d_in = d_obs + d_act
        model = FeedForwardDatum(env, d_in, d_obs, model_options['hidden_sizes'],
            device=options.device).to(options.device)
        return model
    elif model_options['model'] == 'FeedForwardFov':
        d_act = env.model.nu
        d_fov = env.n_fov
        d_in = d_obs + d_act + d_fov
        print('d_obs', d_obs, 'd_act', d_act, 'd_fov', d_fov)
        model = FeedForwardDatum(env, d_in, d_obs, model_options['hidden_sizes'], device=options.device, include_fov=True).to(options.device)
        return model
    elif model_options['model'] == 'ContinuousLatent':
        context_size = options.context_size
        d_act = env.model.nu
        d_in = 2 * d_obs + d_act 
        d = model_options['embedding_vector_dim']
        encoder_hidden_sizes = model_options['encoder_hidden_sizes']
        decoder_hidden_sizes = model_options['decoder_hidden_sizes']
        ee_coef = float(model_options['ee_coef'])
        combine_method = model_options['combine_method']
        recurrent = model_options['recurrent'] if 'recurrent' in model_options else False
        transformer = model_options['transformer'] if 'transformer' in model_options else False
        
        if recurrent:
            print('recurrent!')
            encoder = RNNEncoder(env, context_size, d_in, d, encoder_hidden_sizes, device=options.device).to(options.device)
        elif transformer:
            print('transformer')
            heads = model_options['heads']
            encoder = make_encoder_oh(d_in, d_model=encoder_hidden_sizes[0], h=heads, d_emb=d, device=options.device).to(options.device)
        else:
            print('regular encoder')
            encoder = MultipleEncoder(env, context_size, d_in, d, encoder_hidden_sizes, device=options.device, combine_method=combine_method).to(options.device)

        decoder = SimpleDecoder(env, context_size, d + d_obs + d_act, d_obs, decoder_hidden_sizes, device=options.device).to(options.device)
        model = ContinualLatent(env, context_size, encoder, decoder, device=options.device, ee_coef=ee_coef).to(options.device)

        return model
    elif model_options['model'] == 'Proto':
        context_size = options.context_size
        d_act = env.model.nu
        d_in = 2 * d_obs + d_act 
        d = model_options['embedding_vector_dim']
        k = model_options['discrete_latent_dim']
        temp = model_options['temp']
        proto_coef = model_options['proto_coef']
        encoder_hidden_sizes = model_options['encoder_hidden_sizes']
        decoder_hidden_sizes = model_options['decoder_hidden_sizes']
        latent_type = model_options['latent_type']
        tau = model_options['tau']
        base_coef = model_options['base_coef']
        single_entropy_coef = model_options['single_entropy_coef']
        batch_entropy_coef = model_options['batch_entropy_coef']

        if 'use_predictor' in model_options:
            use_predictor = model_options['use_predictor']
        else:
            use_predictor = True

        encoder = GenericEncoder(env, context_size, d_in, d, encoder_hidden_sizes, device=options.device).to(options.device)

        if latent_type in ['q']:
            decoder = SimpleDecoder(env, context_size, k + d_obs + d_act, d_obs, decoder_hidden_sizes, device=options.device).to(options.device)
        else:
            decoder = SimpleDecoder(env, context_size, d + d_obs + d_act, d_obs, decoder_hidden_sizes, device=options.device).to(options.device)
        
        model = Proto(env, context_size, encoder, decoder, k, d, temp, proto_coef, tau,
            device=options.device, latent_type=latent_type, base_coef=base_coef,
            single_entropy_coef=single_entropy_coef, batch_entropy_coef=batch_entropy_coef,
            use_predictor=use_predictor).to(options.device)

        if options.load_model_location is not None:
            model_path = os.path.join(options.load_model_location, options.model_name)
            with open(model_path, 'rb') as f:
                model_state = torch.load(f)
            model.load_state_dict(model_state)

        if options.freeze_encoder:
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False

        return model
    elif model_options['model'] == 'Variational':
        context_size = options.context_size
        d_act = env.model.nu
        
        d_in = 2 * d_obs + d_act 
        d = model_options['embedding_vector_dim']
        encoder_hidden_sizes = model_options['encoder_hidden_sizes']
        decoder_hidden_sizes = model_options['decoder_hidden_sizes']
        z_dim = model_options['z_dim']
        beta = float(model_options['beta'])
        
        encoder = SingleEncoder(env, context_size, d_in, d, encoder_hidden_sizes, device=options.device).to(options.device)
        decoder = SimpleDecoder(env, context_size, z_dim + d_obs + d_act, d_obs, decoder_hidden_sizes, device=options.device).to(options.device)
        model = VariationalLatent(env, context_size, encoder, decoder, d, z_dim, beta=beta, device=options.device).to(options.device)
        return model
    # elif model_options['model'] == 'LVEBM':
    #     context_size = options.context_size
    #     d_act = env.model.nu
        
    #     z_dim = model_options['z_dim']

    #     d_in = d_obs + d_act + z_dim
    #     model_hidden_sizes = model_options['model_hidden_sizes']

    #     optim_type = model_options['optim_type']

    #     # for sample
    #     n_rand = model_options['n_rand']
        
    #     # for cem
    #     n_sample = model_options['n_sample']
    #     n_elite = model_options['n_elite']
    #     n_iter = model_options['n_iter']

    #     # for sgd and Langevin dynamics
    #     lr = float(model_options['gd_lr'])

    #     # TODO: K zs for targeting instead of just one

    #     energy_model = FeedForward(d_in, d_obs, model_hidden_sizes).to(options.device)
    #     model = LatentVariableEBM(env, context_size, z_dim, energy_model, LatentVariableEBM.optim_z_sample,
    #         device=options.device, N_rand=n_rand, n_sample=n_sample, n_elite=n_elite, n_iter=n_iter, optim_type=optim_type, lr=lr).to(options.device)
        
    #     return model
    # elif model_options['model'] == 'CVAE':
    #     context_size = options.context_size
    #     d_act = env.model.nu
        
    #     z_dim = model_options['z_dim']
    #     hidden_z_size = model_options['hidden_z_size']

    #     d_in = d_obs + d_act + z_dim
    #     encoder_hidden_sizes = model_options['encoder_hidden_sizes']
    #     decoder_hidden_sizes = model_options['decoder_hidden_sizes']
        
    #     n_rand = model_options['n_rand']
        
    #     n_sample = model_options['n_sample']
    #     n_elite = model_options['n_elite']
    #     n_iter = model_options['n_iter']

    #     beta = model_options['beta']

    #     encoder_model = FeedForward(2 * d_obs + d_act, hidden_z_size, encoder_hidden_sizes).to(options.device)
    #     energy_model = FeedForward(d_in, d_obs, decoder_hidden_sizes).to(options.device)

    #     model = CVAE(env, context_size, z_dim, encoder_model, energy_model, None, beta,
    #         hidden_z_size, device=options.device, n_sample=n_sample, n_elite=n_elite, n_iter=n_iter).to(options.device)
    #     return model
    elif model_options['model'] == 'PEARL':
        context_size = options.context_size
        d_act = env.model.nu
        
        d_in = 2 * d_obs + d_act 
        d = model_options['embedding_vector_dim']
        encoder_hidden_sizes = model_options['encoder_hidden_sizes']
        decoder_hidden_sizes = model_options['decoder_hidden_sizes']
        z_dim = model_options['z_dim']
        beta = float(model_options['beta'])
        
        mu_hidden = () # (128, 128)
        sigma_hidden = ()# (128, 128)

        encoder = FeedForward(2 * d_obs + d_act, d, encoder_hidden_sizes).to(options.device)
        decoder = SimpleDecoder(env, context_size, z_dim + d_obs + d_act, d_obs, decoder_hidden_sizes, device=options.device).to(options.device)
        model = PEARLEncoder(env, context_size, encoder, decoder, d, z_dim, mu_hidden, sigma_hidden, beta=beta, device=options.device).to(options.device)
        return model
    else:
        raise Exception("model \'{}\' not implemented".format(model_options['model']))


def parse_dataset(options):
    H = options.context_size # need context even for non-context algos to have right obs
    val_dataset = None
    test_datasets = {}

    if options.use_obs_fn:
        if options.env == 'PushBoxOffset':
            if options.obs_fn is None:
                obs_fn = push_box_state_to_xycs
        elif options.env == 'SlidePuck':
            if options.obs_fn == 'slide_box_state_to_xyt':
                obs_fn = slide_box_state_to_xyt
            elif options.obs_fn == 'slide_box_state_to_xyt_velocity':
                obs_fn = slide_box_state_to_xyt_velocity
            elif options.obs_fn == 'slide_box_state_to_xycs':
                obs_fn = slide_box_state_to_xycs
            elif options.obs_fn == 'slide_box_state_to_xy':
                obs_fn = slide_box_state_to_xy
            else:
                obs_fn = None
        elif options.env == 'Hopper':
            obs_fn = hopper_state_to_obs
        elif options.env == 'Swimmer':
            obs_fn = swimmer_state_to_obs
        else:
            obs_fn = None
    else:
        obs_fn = None

    if options.dataset_type == 'StartEndDataset':
        train_dataset = StartEndDataset(options.train_dataset_location, H)
        for k, v in options.test_datasets:
            test_dataset = StartEndDataset(v, H)
            test_datasets[k] = test_dataset
    elif options.dataset_type == 'SmoothFovDataset':
        train_dataset = SmoothFovDataset(options.train_dataset_location, H, obs_skip=options.obs_skip,
            include_full=options.include_full, pad_till=options.pad_till)
        test_dataset = SmoothFovDataset(options.test_dataset_location, H, obs_skip=options.obs_skip,
            include_full=options.include_full, pad_till=options.pad_till)
    elif options.dataset_type == 'EpisodicStartEndDataset':
        
        train_dataset = EpisodicStartEndDataset(options.train_dataset_location, obs_fn=obs_fn,
            include_full=options.include_full)
        test_dataset = EpisodicStartEndDataset(options.test_dataset_location, obs_fn=obs_fn,
            include_full=options.include_full)
    elif options.dataset_type == 'EpisodicStartEndFovDataset':
        if options.use_obs_fn:
            if options.env == 'PushBox':
                obs_fn = push_box_state_to_xyt
            elif options.env == 'PushBoxOffset':
                obs_fn = push_box_state_to_xycs
            elif options.env == 'SlidePuck':
                if options.obs_fn == 'slide_box_state_to_xyt':
                    obs_fn = slide_box_state_to_xyt
                elif options.obs_fn == 'slide_box_state_to_xy':
                    obs_fn = slide_box_state_to_xy
        else:
            obs_fn = None
        train_dataset = EpisodicStartEndFovDataset(options.train_dataset_location, obs_fn=obs_fn)
        val_dataset = EpisodicStartEndFovDataset(options.val_dataset_location, obs_fn=obs_fn)
        for k, v in zip(options.test_names, options.test_datasets):
            test_dataset = EpisodicStartEndFovDataset(v, obs_fn=obs_fn)
            test_datasets[k] = test_dataset

    elif options.dataset_type == 'ToyDataset':
        train_dataset = ToyDataset(options.train_dataset_location)
        for k, v in zip(options.test_names, options.test_datasets):
            test_dataset = ToyDataset(v)
            test_datasets[k] = test_dataset

    elif options.dataset_type == 'RelabeledEpisodicFovDataset':
        train_dataset = RelabeledEpisodicFovDataset(options.train_dataset_location, obs_fn=obs_fn, context_size=options.context_size)
        val_dataset = RelabeledEpisodicFovDataset(options.val_dataset_location, obs_fn=obs_fn, context_size=options.context_size)
        for k, v in zip(options.test_names, options.test_datasets):
            test_dataset = RelabeledEpisodicFovDataset(v, obs_fn=obs_fn, context_size=options.context_size)
            test_datasets[k] = test_dataset

    elif options.dataset_type == 'RobotPushDataset':
        num_ctx_fn = lambda x: options.context_size # TODO: try out poisson too
        train_dataset = RobotPushDataset(options.train_dataset_location, num_ctx_fn)

        for k, v in zip(options.test_names, options.test_datasets):
            test_dataset = RobotPushDataset(v, num_ctx_fn)
            test_datasets[k] = test_dataset
    
    elif options.dataset_type == 'URRobotPushDataset':
        num_ctx_fn = lambda x: options.context_size # TODO: try out poisson too
        train_dataset = URRobotPushDataset(options.train_dataset_location, num_ctx_fn)
        val_dataset = URRobotPushDataset(options.val_dataset_location, num_ctx_fn)
        for k, v in zip(options.test_names, options.test_datasets):
            test_dataset = URRobotPushDataset(v, num_ctx_fn)
            test_datasets[k] = test_dataset

    if not options.test_datasets and options.train_ratio and options.train_ratio != 1.0:
        n = len(train_dataset)
        n_train = int(n * options.train_ratio)
        n_test = n - n_train
        print('splitting dataset in to train/test', n_train, n_test)
        train_dataset, test_dataset = random_split(train_dataset, (n_train, n_test))
        test_datasets = {'test': test_dataset}

    return train_dataset, test_datasets, val_dataset

def load_everything(exp_dir, best=True):
    # exp_loc = args.experiment
    param_loc = os.path.join(exp_dir, 'params.yaml')
    if best:
        model_loc = os.path.join(exp_dir, 'model_best_val_loss.pickle')
    else:
        model_loc = os.path.join(exp_loc, 'model.pickle')

    model_opt_loc = os.path.join(exp_dir, 'model_options.yaml')

    with open(param_loc, 'r') as f:
        params = yaml.load(f)
    with open(model_opt_loc, 'r') as f:
        model_options = yaml.load(f)
    with open(model_loc, 'rb') as f:
        model_params = torch.load(f)
        env = parse_env(params)
        model = parse_model(params, model_options, env)
        model.load_state_dict(model_params)

    train_dataset, test_sets, val_set = parse_dataset(params)

    return model, train_dataset, test_sets, val_set