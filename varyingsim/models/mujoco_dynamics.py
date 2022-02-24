

import torch
import torch.nn as nn
from varyingsim.util.mujoco import MuJoCoDelta
from varyingsim.models.cl_model import ContinualLearningModel
from varyingsim.util.view import obs_to_relative_torch
from copy import deepcopy

class MuJoCoDynamics(nn.Module):
    def __init__(self, env, model):
        super(MuJoCoDynamics, self).__init__()
        self.env = env
        self.model = model
        self.delta_layer = MuJoCoDelta(env)
    
    # TODO: one input or separate?
    def forward(self, qpos, qvel, ctrl, extra=None, obs=None):
        """
            x - [qpos, qvel, ctrl]
            return [qpos_prime, qvel_prime]
        """

        if obs is None:
            obs = torch.cat([qpos, qvel], dim=-1)

        if extra is not None:
            x = torch.cat([obs, ctrl, extra], dim=-1)
        else:
            x = torch.cat([obs, ctrl], dim=-1)

        delta = self.model(x)
        # qpos, qvel = torch.split(x, self.env.model.nq, dim=-1)
        delta_qpos, delta_qvel = torch.split(delta, self.env.model.nq, dim=-1)
        new_qpos, new_qvel = self.delta_layer(qpos, qvel, delta_qpos, delta_qvel)
        # return torch.cat([new_qpos, new_qvel], dim=-1)
        return new_qpos, new_qvel

class RelativeCLMuJoCo(ContinualLearningModel):
    """
        this is specific to our push box environment
    """

    def __init__(self, model, env, context_size, device):
        super(RelativeCLMuJoCo, self).__init__(env, context_size, device)
        self.model = model
        self.delta_layer = MuJoCoDelta(env)
    
    def forward(self, datum):
        # Change input to relative
        obs_relative, M, M_inv = obs_to_relative_torch(datum["obs"], device=self.device, return_Ms=True)
        context_obs_relative = obs_to_relative_torch(datum["context_obs"], device=self.device, return_Ms=False)

        datum_relative = deepcopy(datum)
        datum['obs'] = obs_relative
        datum['context_obs'] = context_obs_relative
            
        delta = self.model(datum)
        delta_qpos, delta_qvel = torch.split(delta, self.env.model.nq, dim=-1)
    
        # assume deltas are relative, convert back to world
        qpos, qvel = obs[..., :9], obs[..., 9:]

        new_qpos, new_qvel = self.delta_layer(qpos, qvel, delta_qpos, delta_qvel)
        return torch.cat([new_qpos, new_qvel], dim=-1)


class MuJoCoDynamicsFlat(MuJoCoDynamics):
    def __init__(self, env, model):
        super(MuJoCoDynamicsFlat, self).__init__(env, model)
    
    def forward(self, x):
        """
            x - [qpos, qvel, ctrl]
        """
        qpos, qvel, ctrl = torch.split([self.env.model.nq, self.env.model.nv, self.model.nu], dim=-1)
        new_qpos, new_qvel = super().forward(qpos, qvel, ctrl)
        return torch.cat([new_qpos, new_qvel], dim=-1)
