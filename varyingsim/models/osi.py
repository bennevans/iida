
import torch
import torch.nn as nn
import torch.nn.functional as F
from varyingsim.models.feed_forward import FeedForward
from varyingsim.models.cl_model import ContinualLearningModel
import numpy as np

class OSI(nn.Module):
    # TODO: pass in shapes, activations, etc? or just the modules

    def __init__(self, h, d_in, d_param, d_share, d_hidden_shared, d_hidden_osi):
        """
            h - the time window
        """
        super(OSI, self).__init__()

        self.h = h
        self.d_in = d_in
        self.d_param = d_param
        self.d_share = d_share
        self.d_hidden_shared = d_hidden_shared
        self.d_hidden_osi = d_hidden_osi

        self.shared_encoder = FeedForward(d_in, d_share, d_hidden_shared)
        self.osi = FeedForward(d_share * h, d_param, d_hidden_osi)

    def forward(self, x):
        """
            x - (N, h, d_in)
            out - (N, d_param)
        """
        N = x.shape[0]
        xs = x.view(N * self.h, self.d_in)
        feat = self.shared_encoder(xs)
        feats = feat.view(N, self.h * self.d_share) # TODO: need to verify we're not mixing batches
        return self.osi(feats)

class OSIModel(ContinualLearningModel):
    # TODO: pass in shapes, activations, etc? or just the modules

    def __init__(self, env, h, d_in, d_param, d_share, d_hidden_shared, d_hidden_osi, device='cpu'):
        """
            h - the time window
        """
        super(OSIModel, self).__init__(env, h, device=device)

        self.h = h
        self.d_in = d_in
        self.d_param = d_param
        self.d_share = d_share
        self.d_hidden_shared = d_hidden_shared
        self.d_hidden_osi = d_hidden_osi

        self.shared_encoder = FeedForward(d_in, d_share, d_hidden_shared)
        self.osi = FeedForward(d_share * h, d_param, d_hidden_osi)

    def forward(self, datum):
        """
            x - (N, h, d_in)
            out - (N, d_param)
        """
        N = x.shape[0]
        xs = x.view(N * self.h, self.d_in)
        feat = self.shared_encoder(xs)
        feats = feat.view(N, self.h * self.d_share) # TODO: need to verify we're not mixing batches
        return self.osi(feats)

        obs = datum['obs'].to(self.device)
        act = datum['act'].to(self.device)
        context_obs = datum['context_obs'].to(self.device)
        context_act = datum['context_act'].to(self.device)
        if len(datum['obs'].shape) == 1:
            batch_size = 1
        else:
            batch_size = datum['obs'].shape[0]

        print('osimujoco', act)
        obs = self.obs_transform(obs, act, self.device)

        x = torch.cat([obs, act], dim=-1)
        qpos, qvel = torch.split(obs, self.env.model.nq, dim=-1)

        context_obs = self.obs_transform(context_obs, context_act, self.device)

        x_hist = torch.cat([context_obs, context_act], dim=-1).to(self.device)

        if x_hist.shape[-2] < self.context_size:
            # pad with zeros
            shape = np.array(x_hist.shape)
            shape[-2] = self.context_size - shape[-2] 
            x_hist = torch.cat([torch.zeros(*shape, device=self.device), x_hist], dim=-2)

        x = x.to(self.device)
        x_hist = x_hist.to(self.device).view(batch_size, self.context_size, -1)

        fov = self.osi_model(x_hist).squeeze(0)

        
class OSIMuJoCo(ContinualLearningModel):
    def __init__(self, env, osi_model, mujoco_model, context_size, device='cpu', obs_transform=None):
        super(OSIMuJoCo, self).__init__(env, context_size, device, obs_transform)
        self.osi_model = osi_model
        self.mujoco_model = mujoco_model

    def forward(self, datum):
        obs = datum['obs'].to(self.device)
        act = datum['act'].to(self.device)
        context_obs = datum['context_obs'].to(self.device)
        context_act = datum['context_act'].to(self.device)
        if len(datum['obs'].shape) == 1:
            batch_size = 1
        else:
            batch_size = datum['obs'].shape[0]

        print('osimujoco', act)
        obs = self.obs_transform(obs, act, self.device)

        x = torch.cat([obs, act], dim=-1)
        qpos, qvel = torch.split(obs, self.env.model.nq, dim=-1)

        context_obs = self.obs_transform(context_obs, context_act, self.device)

        x_hist = torch.cat([context_obs, context_act], dim=-1).to(self.device)

        if x_hist.shape[-2] < self.context_size:
            # pad with zeros
            shape = np.array(x_hist.shape)
            shape[-2] = self.context_size - shape[-2] 
            x_hist = torch.cat([torch.zeros(*shape, device=self.device), x_hist], dim=-2)

        x = x.to(self.device)
        x_hist = x_hist.to(self.device).view(batch_size, self.context_size, -1)

        fov = self.osi_model(x_hist).squeeze(0)
        return self.mujoco_model(qpos, qvel, act, fov, obs=obs)



        



