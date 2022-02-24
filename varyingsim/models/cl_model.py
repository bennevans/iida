
from varyingsim.envs.cartpole import CartpoleEnv
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from varyingsim.envs.dummy_env import DummyEnv
from varyingsim.envs.cartpole import CartpoleEnv

class ContinualLearningModel(nn.Module):
    def __init__(self, env, context_size, device='cpu', obs_transform=None):
        super(ContinualLearningModel, self).__init__()
        self.env = env
        self.context_size = context_size
        self.device = device

    def forward(self, datum, ret_extra=False):
        """
            datum - dicitonary with obs, act, context_obs, context_act, and obs_prime
            returns qpos, qvel TODO: make just obs everywhere
        """
        raise Exception("not implemented")

    def loss(self, datum, return_info=False, wandb_obj=None, train=True, ret_extra=False):
        if ret_extra:
            recon, extra = self(datum, ret_extra=True)
        else:
            recon = self(datum)

        ret = self.env.loss(recon, datum['obs_prime'])
        if return_info:
            if ret_extra:
                return ret, extra
            return ret
        else:
            if ret_extra:
                return ret[0], extra
            return ret[0]

    def save(self, f):
        torch.save(self.state_dict(), f)