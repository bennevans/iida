
# create a general way to connect algorithms and environments for training

from curses import start_color
from torch.utils.data import DataLoader
import numpy as np
import torch
from varyingsim.util.trajectory import rollout_algo
import os
import pickle
from tqdm import tqdm
from varyingsim.util.buffers import KeyedReservoirBuffer
import time
from torch.optim.lr_scheduler import StepLR
import ctypes

from copy import deepcopy

def torchify(datum, device='cpu'):
    ret = {}
    for k, v in datum.items():
        if type(v) == np.ndarray:
            ret[k] = torch.from_numpy(v).float().to(device)
        elif type(v) == torch.Tensor:
            ret[k] = v.to(device)
        else:
            ret[k] = v
    return ret

def select_start_end(datum, n, device):
    B, R, d_obs = datum['obs'].shape
    sample_idxs = torch.randint(R, (B, n), device=device)
    s = datum['obs'].to(device).gather(1, sample_idxs.view(B, n, 1).repeat(1, 1, datum['obs'].shape[2])).to(device)
    a = datum['act'].to(device).gather(1, sample_idxs.view(B, n, 1).repeat(1, 1, datum['act'].shape[2])).to(device)
    sp = datum['obs_prime'].to(device).gather(1, sample_idxs.view(B, n, 1).repeat(1, 1, datum['obs_prime'].shape[2])).to(device)
    fov = datum['fov'].to(device).gather(1, sample_idxs.view(B, n, 1).repeat(1, 1, datum['fov'].shape[2])).to(device)

    return s, a, sp, fov

def select_traj(datum, n, device):
    B, R, T, d_obs = datum['obs'].shape
    if 'lens' in datum.keys():
        batch_s, batch_a, batch_sp, batch_f = [], [], [], []
        for states, acts, state_primes, fovs in zip(datum['obs'], datum['act'], datum['obs_prime'], datum['fov']):
            all_states, all_acts, all_state_primes, all_fovs = [], [], [], []
            for i, lens in  enumerate(datum['lens']):
                l = lens[0].item()
                all_states.append(states[i][:l])
                all_acts.append(acts[i][:l])
                all_state_primes.append(state_primes[i][:l])
                all_fovs.append(fovs[i][:l])
            all_s = torch.cat(all_states)
            all_a = torch.cat(all_acts)
            all_sp = torch.cat(all_state_primes)
            all_f = torch.cat(all_fovs)

            T = all_s.shape[0]
            sample_idxs = torch.randint(T, (n,), device=device)
            batch_s.append(all_s[sample_idxs])
            batch_a.append(all_a[sample_idxs])
            batch_sp.append(all_sp[sample_idxs])
            batch_f.append(all_f[sample_idxs])

        s = torch.stack(batch_s).to(device)
        a = torch.stack(batch_a).to(device)
        sp = torch.stack(batch_sp).to(device)
        fov = torch.stack(batch_f).to(device)
    else:
        sample_idxs = torch.randint(R * T, (B, n), device=device)
        s = datum['obs'].to(device).view(B, R*T, -1).gather(1, sample_idxs.view(B, n, 1).repeat(1, 1, datum['obs'].shape[-1])).to(device)
        a = datum['act'].to(device).view(B, R*T, -1).gather(1, sample_idxs.view(B, n, 1).repeat(1, 1, datum['act'].shape[-1])).to(device)
        sp = datum['obs_prime'].to(device).view(B, R*T, -1).gather(1, sample_idxs.view(B, n, 1).repeat(1, 1, datum['obs_prime'].shape[-1])).to(device)
        fov = datum['fov'].to(device).view(B, R*T, -1).gather(1, sample_idxs.view(B, n, 1).repeat(1, 1, datum['fov'].shape[-1])).to(device)
    return s, a, sp, fov

def select_datas(datum, n=1, device='cpu'):
    # n_obs = len(datum['obs'].shape)
    # if n_obs == 3:
    return select_start_end(datum, n, device)
    # elif n_obs == 4:
    # return select_traj(datum, n, device)

def create_train_point(datum, n_batch=1, n_context=1, device='cpu', same_ctx=False):
    s, a, sp, fov = select_datas(datum, n=n_batch, device=device)
    ctx_s, ctx_a, ctx_sp, ctx_fov = select_datas(datum, n=n_batch * n_context, device=device)
    B, R, _ = s.shape

    if same_ctx:
        ret = dict(
            obs=s.view(B * n_batch, -1),
            act=a.view(B * n_batch, -1),
            obs_prime=sp.view(B * n_batch, -1),
            context_obs=s.view(B * n_batch, 1, -1).repeat(1, n_context, 1),
            context_act=a.view(B * n_batch, 1, -1).repeat(1, n_context, 1),
            context_obs_prime=sp.view(B * n_batch, 1, -1).repeat(1, n_context, 1),
            fov=fov.view(B * n_batch, -1),
        )
    else:
        ret = dict(
            obs=s.view(B * n_batch, -1),
            act=a.view(B * n_batch, -1),
            obs_prime=sp.view(B * n_batch, -1),
            context_obs=ctx_s.view(B * n_batch, n_context, -1),
            context_act=ctx_a.view(B * n_batch, n_context, -1),
            context_obs_prime=ctx_sp.view(B * n_batch, n_context, -1),
            fov=fov.view(B * n_batch, -1),
        )

    return ret

def get_next(options, dataset_iter, loader):
    try:
        data = next(dataset_iter)
    except StopIteration:
        dataset_iter = iter(loader)
        data = next(dataset_iter)

    data = clean_data(options, data)
    return data, dataset_iter

def clean_data(options, data):
    if 'context_obs' not in data:
        data = create_train_point(data, n_batch=options.n_batch, n_context=options.context_size, device=options.device, same_ctx=options.same_ctx)
    data = torchify(data, device=options.device)
    return data

def evaluate(model, dataset, name, seed_fn, options):

    start_eval_time = time.time()

    loader = DataLoader(dataset, batch_size=options.val_batch_size, shuffle=True,
        num_workers=options.num_workers, worker_init_fn=seed_fn)

    total_loss = 0.0
    n_iters = 0

    new_infos = {}

    start_loop_time = time.time()
    for datum in loader:
        start_data_l = time.time()
        datum = clean_data(options, datum)
        clean = time.time()
        loss, info = model.loss(datum, return_info=True, train=False)
        losst = time.time()
        for k,v in info.items():
            key = name + '_' + k
            if key in new_infos:
                new_infos[key] += v
            else:
                new_infos[key] = v

        total_loss += loss.item()
        n_iters += 1
        other = time.time()
        # print('total', other - start_data_l)
        # print('clean', clean - start_data_l)
        # print('loss', losst - clean)
        # print('other', other - losst)
        # import ipdb; ipdb.set_trace()
    

    for k in new_infos:
        new_infos[k] = new_infos[k] / n_iters

    end_loop_time = time.time()

    # print('total time', end_loop_time - start_eval_time)
    # print('dataloader', start_loop_time - start_eval_time)
    # print('loop', end_loop_time - start_loop_time)

    return total_loss / n_iters, new_infos

def update_best(best_info, name, avg_loss, iteration):
    if avg_loss < best_info['best_' + name + '_loss']:
        best_info['best_' + name + '_loss'] = avg_loss
        best_info['best_' + name + '_iter'] = iteration
        return True
    return False

def train(dataset, options, model, exp_path, sess, wandb, test_datasets=None, val_dataset=None):

    start_time = time.time()

    def gen_seed_workers(offset=0):
        custom_base_seed = ctypes.c_uint32(hash(str(options.seed))).value
        def seed_workers(worker_id):
            seed = custom_base_seed + worker_id + offset
            np.random.seed(seed)
            torch.manual_seed(seed)
        return seed_workers

    seed_workers_train = gen_seed_workers()
    seed_workers_val = gen_seed_workers(1)

    loader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True,
                num_workers=options.num_workers, worker_init_fn=seed_workers_train)
    val_loader = DataLoader(val_dataset, batch_size=options.val_batch_size, shuffle=True,
        num_workers=options.num_workers, worker_init_fn=seed_workers_val)

    if test_datasets:
        test_dataset_list = []
        dataset_names = []
        for i, (dataset_name, dataset) in enumerate(test_datasets.items()):
            test_dataset_list.append(dataset)
            dataset_names.append(dataset_name)
        n_test = len(dataset_names)


    # TODO: add parse optim function to run_exp
    if options.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=options.lr)
    elif options.optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
    else:
        raise Exception('unknown optimzier {}'.format(options.optim))
        
    if options.lr_step_size:
        scheduler = StepLR(optim, step_size=options.lr_step_size, gamma=options.lr_gamma)

    train_losses = []
    test_losses = []
    val_losses = []
    infos = []
    best_infos = []

    best_info = {
        'best_validation_loss': float('inf'),
        'best_validation_iter': 0,
        'iteration': 0
    }

    for name in dataset_names:
        best_info['best_' + name + '_loss'] = float('inf')
        best_info['best_' + name + '_iter'] = 0

    iteration = 0

    # TODO: if it's too slow
    # best_models = {name:deepcopy(model) for name in ['val'] + dataset_names}

    start_loop = time.time()
    print('before loop time', start_loop - start_time)
    for epoch in range(options.epochs):
        for i, data in enumerate(loader):
            inside_loop = time.time()
            model.train()

            data = clean_data(options, data)
            optim.zero_grad()

            loss, info = model.loss(data, True)
            loss.backward()
            optim.step()

            try:
                model.update()
            except AttributeError:
                pass

            train_losses.append(loss.item())
            infos.append(info)

            before_test = time.time()
            if iteration % options.test_iter == 0:
                log_info = dict(epoch=epoch, iteration=iteration, loss=loss.item())

                train_info = {}
                for k, v in infos[-1].items():
                    train_info['train_' + k] = v
                log_info.update(train_info)

                model.eval()

                test_losses = {}

                for i in range(n_test):
                    name = dataset_names[i]
                    test_dataset = test_dataset_list[i]
                    avg_test_loss, avg_test_info = evaluate(model, test_dataset, name, seed_workers_val, options)

                    log_info.update(avg_test_info)
                    test_losses[name] = avg_test_loss

                    better = update_best(best_info, name, avg_test_loss, iteration)
                    if better:
                        print('saving - best {} loss, so far!'.format(name))
                        with open(os.path.join(exp_path, 'model_best_{}_loss.pickle'.format(name)), 'wb') as f:
                            model.save(f)
                    log_info.update(best_info)

                print('epoch: {} i: {} train_loss: {} '.format(epoch, iteration, loss.item()))
                for k, v in log_info.items():
                    if k.endswith("base_loss"):
                        print("{}: {}".format(k, v), end=' ')
                print()

                # evaluate on the entire val set and save the model if it's better
                name = 'validation'

                avg_val_loss, avg_val_info = evaluate(model, val_dataset, name, seed_workers_val, options)

                log_info.update(avg_val_info)

                print('epoch: {} i: {} train_loss: {} '.format(epoch, iteration, loss.item()))
                for k, v in log_info.items():
                    if k.endswith("base_loss"):
                        print("{}: {}".format(k, v), end=' ')
                print()
                
                better = update_best(best_info, name, avg_val_loss, iteration)
                if better:
                    for k, v in test_losses.items():
                        best_info['{}_at_best_val'.format(k)] = v

                    print('saving - best val loss, so far!')
                    with open(os.path.join(exp_path, 'model_best_val_loss.pickle'), 'wb') as f:
                        model.save(f)
                
                log_info.update(best_info)

                best_infos.append(best_info)

                if not options.no_wandb:
                    wandb.log(log_info)
            after_test = time.time()

            iteration += 1

            # print('total loop time:', after_test - inside_loop)
            # print('time before test:', before_test - inside_loop)
            # print('test time:', after_test - before_test)

        if options.lr_step_size:
            scheduler.step()

        if iteration % options.save_iter == 0:
            # save every epoch now
            print('saving!')
            with open(os.path.join(exp_path, 'model_{}_{}.pickle'.format(epoch, iteration)), 'wb') as f:
                model.save(f)

    train_info = dict(train_losses=train_losses, infos=infos, best_infos=best_infos)
    return model, train_info
