
import torch
import numpy as np
from mujoco_py import functions

def rollout_algo_single(algo, datum, end=None):
    """
        TODO: assumes numpy in and out, but can do torch later
        TODO: batch rollouts
    """
    obs_0, acts, fovs = datum['obs'], [datum['act']], [datum['fov']]
    cont_obss, cont_acts = [datum['context_obs']], [datum['context_act']]
    T = len(acts)
    obs = torch.from_numpy(obs_0).float()
    act = torch.from_numpy(acts[0]).float()
    fov = torch.from_numpy(fovs[0]).float()
    cont_obs = torch.from_numpy(cont_obss[0]).float()
    cont_act = torch.from_numpy(cont_acts[0]).float()
    
    obs_hats = []
    obs_hats.append(obs)
    errs = []

    end = T if end is None else end

    dev = algo.device
    algo.device = 'cpu'
    algo.to('cpu')

    for t in range(end-1):
        dat = dict(obs=obs, act=act, fov=fov, context_obs=cont_obs,  context_act=cont_act)
        qpos_prime, qvel_prime = algo.forward(dat)
        act = torch.from_numpy(acts[t]).float()
        fov = torch.from_numpy(fovs[t]).float()
        cont_obs = torch.from_numpy(cont_obss[t]).float()
        cont_act = torch.from_numpy(cont_acts[t]).float()
        obs = torch.cat([qpos_prime, qvel_prime], dim=-1).cpu()
        obs_hats.append(obs)

    traj = torch.stack(obs_hats).detach().cpu().numpy()
    algo.device = dev
    return traj


def fix_context(arr, context_size):
    context_shape = np.array(arr.shape)
    context_shape[-2] = context_size - context_shape[-2] 
    if arr.shape[-2] == 0:
        return np.zeros(context_shape)
    elif arr.shape[-2] < context_size:
        return np.concatenate([np.zeros(context_shape), arr], axis=-2)
    else:
        return arr

def update_context(arr, new):
    return np.concatenate([arr[1:], np.expand_dims(new, 0)], axis=-2)

def rollout_algo(algo, datum, acts, fovs, end=None):
    """
        TODO: assumes numpy in and out, but can do torch later
        TODO: batch rollouts
    """
    obs_0 = datum['obs']
    cont_obss, cont_acts = [fix_context(datum['context_obs'], algo.context_size)], [fix_context(datum['context_act'], algo.context_size)]
    T = len(acts)
    obs = torch.from_numpy(obs_0).float()
    act = torch.from_numpy(acts[0]).float()
    fov = torch.from_numpy(fovs[0]).float()
    cont_obs = torch.from_numpy(cont_obss[0]).float()
    cont_act = torch.from_numpy(cont_acts[0]).float()
    
    obs_hats = []
    obs_hats.append(obs)
    errs = []

    end = T if end is None else end

    dev = algo.device
    algo.device = 'cpu'
    algo.to('cpu')

    for t in range(end-1):
        dat = dict(obs=obs, act=act, fov=fov, context_obs=cont_obs,  context_act=cont_act)
        qpos_prime, qvel_prime = algo.forward(dat)
        act = torch.from_numpy(acts[t]).float()
        fov = torch.from_numpy(fovs[t]).float()
        cont_obs = torch.from_numpy(cont_obss[t]).float()
        cont_act = torch.from_numpy(cont_acts[t]).float()
        obs = torch.cat([qpos_prime, qvel_prime], dim=-1).cpu()
        obs_hats.append(obs)

        new_context_obs = update_context(cont_obss[t], obs.detach().numpy())
        new_context_act = update_context(cont_acts[t], act.detach().numpy())
        cont_obss.append(new_context_obs)
        cont_acts.append(new_context_act)


    traj = torch.stack(obs_hats).detach().cpu().numpy()
    algo.device = dev
    return traj

def get_traj_pol(env, i, policy_fn, set_fov, traj_len, ret_info=False):

    obs_traj = []
    act_traj = []
    fov_traj = []
    rew_traj = []
    fov_memory = dict()
    act_memory = dict()
    memories = []
    
    set_fov(env, i, 0, fov_memory)
    env.sim.set_constants()

    obs = env.reset()

    for t in range(traj_len + 1):
        fov = env.get_fovs()
        act = policy_fn(obs, i, t, act_memory)

        obs_traj.append(obs)
        act_traj.append(act)
        fov_traj.append(fov)
        obs, rew, done, info = env.step(act)
        rew_traj.append(rew)
        if fov_memory:
            memories.append(fov_memory)

        if done:
            break
    if not ret_info:
        return np.array(obs_traj), np.array(act_traj), np.array(fov_traj), np.array(rew_traj)
    return np.array(obs_traj), np.array(act_traj), np.array(fov_traj), np.array(rew_traj), memories

def toucher_in_concact(env):
    box_id = env.model.geom_name2id('box')
    pusher_id = env.model.geom_name2id('pusher')
    for i in range(env.sim.data.ncon):
        if (env.sim.data.contact[i].geom1 == box_id and env.sim.data.contact[i].geom2 == pusher_id) or \
            (env.sim.data.contact[i].geom2 == box_id and env.sim.data.contact[i].geom1 == pusher_id):            
            return True
    return False 

def get_traj_pol_contact(env, i, policy_fn, set_fov, traj_len):
    obs = env.reset()
    obs_traj = []
    act_traj = []
    fov_traj = []
    rew_traj = []
    fov_memory = dict()
    act_memory = dict()
    in_con = []
    for t in range(traj_len + 1):
        set_fov(env, i, t, fov_memory)
        fov = env.get_fovs()
        act = policy_fn(obs, i, t, act_memory)

        obs_traj.append(obs)
        act_traj.append(act)
        fov_traj.append(fov)
        obs, rew, done, info = env.step(act)
        rew_traj.append(rew)
        in_con.append(toucher_in_concact(env))

        if done:
            break

    contacts, = np.where(in_con)
    if len(contacts) > 0:
        first_concact = contacts[0]
        last_contact = contacts[-1]
        return np.array(obs_traj)[first_concact:last_contact], \
            np.array(act_traj)[first_concact:last_contact], \
            np.array(fov_traj)[first_concact:last_contact], \
            np.array(rew_traj)[first_concact:last_contact]
    else:
        return None

    # return np.array(obs_traj), np.array(act_traj), np.array(fov_traj), np.array(rew_traj)

def get_traj_acts(env, obs, i, acts, set_fov):

        qpos, qvel = obs[:env.model.nq], obs[env.model.nq:]
        env.set_state(qpos, qvel)
        obs_traj = []
        act_traj = []
        fov_traj = []
        rew_traj = []

        traj_len = len(acts)
        for t in range(traj_len):
            set_fov(env, i, t)
            fov = env.get_fovs()
            act = acts[t]

            obs_traj.append(obs)
            act_traj.append(act)
            fov_traj.append(fov)
            obs, rew, done, info = env.step(act)
            rew_traj.append(rew)

            if done:
                break

        return np.array(obs_traj), np.array(act_traj), np.array(fov_traj), np.array(rew_traj)
    