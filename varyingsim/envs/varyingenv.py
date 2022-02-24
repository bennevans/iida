
import numpy as np
from gym import utils
from varyingsim.envs import mujoco_env
from mujoco_py import functions

import torch.nn.functional as F

# TODO: reset factors of variation to intiial values

class VaryingEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, xml_file, frame_skip, include_fov=False, set_param_fn=None): 
        self.include_fov = include_fov
        self.t = -1
        self.episode = -1
        self.set_param_fn = set_param_fn
        
        mujoco_env.MujocoEnv.__init__(self, xml_file, frame_skip)
        utils.EzPickle.__init__(self)

    def step(self, a):
        if self.set_param_fn:
            self.set_param_fn(self, self.episode, self.t)
        
        # did this for some reason. something about updating the kinematic chain
        # it causes issues with the PushBoxCircle. disabling for now

        # state = self.sim.get_state()
        # functions.mj_setConst(self.model, self.sim.data)
        # self.sim.set_state(state)

        self.t += 1
        prev_obs = self._get_obs()
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        reward = self.get_reward(prev_obs, obs, a)
    
        return obs, reward, False, dict(t=self.t)

    def get_fovs(self):
        """
            returns a list of the vactors of variation
        """
        raise Exception('must override get_fovs')
    
    def get_fov_names(self):
        raise Exception('must override get_fov_names')

    def get_fov_idx(self, name):
        return self.get_fov_names().index(name)

    def get_fov(self, name):
        return self.get_fovs()[self.get_fov_idx(name)]

    def get_fov_dict(self):
        return {k: v for k, v in zip(self.get_fov_names(), self.get_fovs())}

    def flatten(self, fov_dict):
        return [fov_dict[k] for k in self.get_fov_names()]

    def set_fovs(self, fovs):
        if type(fovs) == list:
            if len(fovs) != self.n_fov:
                raise Exception("must pass in correct number of fovs. got {} expected {}".format(len(fovs), self.n_fov))
            for name, val in zip(self.get_fov_names(), fovs):
                if val:
                    self.set_fov(name, val)
        elif type(fovs) == dict:
            for fov_name, fov in fovs.items():
                self.set_fov(fov_name, fov)
        else:
            raise Exception('fov must be list or dict')

    @property
    def n_fov(self):
        return len(self.get_fov_names())

    def get_reward(self, prev_obs, obs, act):
        return 0.0

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        ret = np.concatenate([qpos.flat, qvel.flat])
        if self.include_fov:
            fov = self.get_fovs()
            ret = np.concatenate([ret, fov])
        return ret

    def reset_model(self):
        self.episode += 1
        self.set_state(
            self.init_qpos, 
            self.init_qvel
        )
        self.t = -1
        return self._get_obs()

    def set_fov(self, name, val):
        fov_names = self.get_fov_names()
        if name in fov_names:
            set_fn = getattr(self, 'set_' +name)
            set_fn(val)
        else:
            raise Exception('invalid name: {}. expected one of {}'.format(name, fov_names))

    def loss(self, obs_hat, obs):
        """
            loss function defined by the environment. defaults to mse if not overwritten,
            but should write custom loss if mix of angles and positions
        """
        loss = F.mse_loss(obs_hat, obs)
        return loss, dict(base_loss=loss.item())