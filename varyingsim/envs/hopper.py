from varyingsim.envs.varyingenv import VaryingEnv
import numpy as np
from gym import utils

import torch
import torch.nn.functional as F

from varyingsim.util.dataset import sample_fovs

class HopperEnv(VaryingEnv):
    def __init__(self):
        VaryingEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

        self.torso_body_idx = self.model.body_name2id('torso')
        self.thigh_body_idx = self.model.body_name2id('thigh')
        self.leg_body_idx = self.model.body_name2id('leg')
        self.foot_body_idx = self.model.body_name2id('foot')

        self.torso_geom_idx = self.model.body_geomadr[self.torso_body_idx]
        self.thigh_geom_idx = self.model.body_geomadr[self.thigh_body_idx]
        self.leg_geom_idx = self.model.body_geomadr[self.leg_body_idx]
        self.foot_geom_idx = self.model.body_geomadr[self.foot_body_idx]

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        if 'torso_body_idx' in dir(self):
            info = {'fovs': self.get_fov_dict()}
        else:
            info = {}
        return ob, reward, done, info

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    # getters
    def get_torso_mass(self):
        return self.model.body_mass[self.torso_body_idx]

    def get_thigh_mass(self):
        return self.model.body_mass[self.thigh_body_idx]

    def get_leg_mass(self):
        return self.model.body_mass[self.leg_body_idx]
    
    def get_foot_mass(self):
        return self.model.body_mass[self.foot_body_idx]

    def get_torso_length(self):
        return self.model.geom_size[self.torso_geom_idx, 0]

    def get_thigh_length(self):
        return self.model.geom_size[self.thigh_geom_idx, 0]

    def get_leg_length(self):
        return self.model.geom_size[self.leg_geom_idx, 0]

    def get_foot_length(self):
        return self.model.geom_size[self.foot_geom_idx, 0]

    # setters
    def set_torso_mass(self, val):
        self.model.body_mass[self.torso_body_idx] = val

    def set_thigh_mass(self, val):
        self.model.body_mass[self.thigh_body_idx] = val

    def set_leg_mass(self, val):
        self.model.body_mass[self.leg_body_idx] = val
    
    def set_foot_mass(self, val):
        self.model.body_mass[self.foot_body_idx] = val

    def set_torso_length(self, val):
        self.model.geom_size[self.torso_geom_idx, 0] = val

    def set_thigh_length(self, val):
        self.model.geom_size[self.thigh_geom_idx, 0] = val

    def set_leg_length(self, val):
        self.model.geom_size[self.leg_geom_idx, 0] = val

    def set_foot_length(self, val):
        self.model.geom_size[self.foot_geom_idx, 0] = val
    
    def get_fovs(self):
        return [self.get_torso_mass(), self.get_thigh_mass(), self.get_leg_mass(), self.get_foot_mass(), 
            self.get_torso_length(), self.get_thigh_length(), self.get_leg_length(), self.get_foot_length()]

    def get_fov_names(self):
        return ['torso_mass', 'thigh_mass', 'leg_mass',  'foot_mass',
                'torso_length', 'thigh_length', 'leg_length', 'foot_length']

    def loss(self, obs_hat, obs):
        zy_hat = obs_hat[..., :2]
        thigh_leg_foot_hat = obs_hat[..., 2:5]
        vel_hat = obs_hat[..., 5:]
        cs_hat = torch.cat([torch.cos(thigh_leg_foot_hat), torch.sin(thigh_leg_foot_hat)], dim=-1)

        zy = obs[..., :2]
        thigh_leg_foot = obs[..., 2:5]
        vel = obs[..., 5:]
        cs = torch.cat([torch.cos(thigh_leg_foot), torch.sin(thigh_leg_foot)], dim=-1)

        zy_loss = F.mse_loss(zy_hat, zy)
        cs_loss = F.mse_loss(cs_hat, cs)
        vel_loss = F.mse_loss(vel_hat, vel)

        loss = zy_loss + cs_loss + vel_loss
        return loss, dict(zy_loss=zy_loss.item(), cs_loss=cs_loss.item(), vel_loss=vel_loss.item(), base_loss=loss.item())

class HopperStreamingEnv(HopperEnv):
    def __init__(self, fov_config, seed=None):
        # TODO: could do a more generic sampler function
        """
            fov_config - a dict of form {fov_name: {low: <low>, high: <high>}}
        """
        if seed:
            self.seed(seed)
        super().__init__()
        self.fov_config = fov_config

    def new_fov(self):
        fovs = sample_fovs(self.fov_config)
        self.set_fovs(fovs)
        self.sim.set_constants()

    def reset_model(self):
        self.new_fov()
        return super().reset_model()