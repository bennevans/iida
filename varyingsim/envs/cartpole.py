
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from varyingsim.envs.varyingenv import VaryingEnv
import torch
import torch.nn.functional as F

# TODO: restructure environment into a folder
# TODO: set which fovs are given in obs?
# TODO: random reset

class CartpoleEnv(VaryingEnv):

    UPRIGHT = 'UPRIGHT'
    SWINGUP = 'SWINGUP'

    def __init__(self, mode=UPRIGHT, include_fov=False, set_param_fn=None, T=1000): 
        self.mode = mode
        self.rotation = 0.0
        self.T = T
        VaryingEnv.__init__(self, 'cartpole.xml', 4, include_fov=include_fov, set_param_fn=set_param_fn)

    def set_cart_mass(self, mass):
        self.model.body_mass[1] = mass

    def get_cart_mass(self):
        return self.model.body_mass[1]

    def set_pole_mass(self, mass):
        self.model.body_mass[2] = mass

    def get_pole_mass(self):
        return self.model.body_mass[2]
    
    def set_end_mass(self, mass):
        self.model.body_mass[3] = mass

    def get_end_mass(self):
        return self.model.body_mass[3]

    def set_rotation(self, rotation):
        self.rotation = rotation

    def get_rotation(self):
        return self.rotation

    def get_wind_x(self):
        return self.model.opt.wind[0]

    def get_wind_z(self):
        return self.model.opt.wind[2]

    def set_wind_x(self, val):
        self.model.opt.wind[0] = val

    def set_wind_z(self, val):
        self.model.opt.wind[2] = val
    
    def set_slider_damping(self, val):
        self.model.dof_damping[0] = val

    def set_hinge_damping(self, val):
        self.model.dof_damping[1] = val

    def get_slider_damping(self):
        return self.model.dof_damping[0]
    
    def get_hinge_damping(self):
        return self.model.dof_damping[1]

    def get_fovs(self):
        return [self.get_cart_mass(), self.get_pole_mass(), self.get_end_mass(), self.get_rotation(),
        self.get_wind_x(), self.get_wind_z(), self.get_slider_damping(), self.get_hinge_damping()]
    
    def get_fov_names(self):
        return ['cart_mass', 'pole_mass', 'end_mass', 'rotation', 'wind_x', 'wind_z', 'slider_damping', 'hinge_damping']

    def _get_obs(self):
        qpos = self.sim.data.qpos.flat
        qvel = self.sim.data.qvel.flat

        if self.rotation != 0.0:
            R = np.array([[np.cos(self.rotation), -np.sin(self.rotation)], [np.sin(self.rotation), np.cos(self.rotation)]]) 
            
            qpos = R @ qpos
            qvel = R @ qvel

        ret = np.concatenate([qpos, qvel])
        if self.include_fov:
            fov = self.get_fovs()
            ret = np.concatenate([ret, fov])
        return ret

    def step(self, a):
        if self.set_param_fn:
            self.set_param_fn(self, self.episode, self.t)

        self.t += 1
        prev_obs = self._get_obs()
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()

        if self.mode is self.UPRIGHT:
            reward = 1.0
            notdone = np.isfinite(obs).all() and (np.abs(obs[1]) <= 0.2)
            done = (not notdone) or (self.t > self.T)
        elif self.mode is self.SWINGUP:
            smooth_rew = np.cos(obs[1])
            bonus_rew = (np.abs(obs[1]) <= 0.2)
            reward = smooth_rew + bonus_rew
            # print(smooth_rew, bonus_rew)
            done = self.t > self.T

        return obs, reward, done, dict(t=self.t)
    
    def reset_model(self):
        self.episode += 1
        if self.mode is self.UPRIGHT:
            qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
            qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
            self.set_state(qpos, qvel)
        elif self.mode is self.SWINGUP:
            qpos = np.array([0, np.pi])+ self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
            qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
            self.set_state(qpos, qvel)

        self.t = -1
        return self._get_obs()

    def loss(self, obs_hat, obs):
        x_hat = obs_hat[..., 0]
        theta_hat = obs_hat[..., 1]
        vel_hat = obs_hat[..., 2:]

        x = obs[..., 0]
        theta = obs[..., 1]
        vel = obs[..., 2:]

        cs_hat = torch.cat([torch.cos(theta_hat), torch.sin(theta_hat)], dim=-1)
        cs = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)

        x_loss = F.mse_loss(x_hat, x)
        cs_loss = F.mse_loss(cs_hat, cs)
        vel_loss = F.mse_loss(vel_hat, vel)

        loss = x_loss + cs_loss + vel_loss
        return loss, dict(x_loss=x_loss.item(), cs_loss=cs_loss.item(), vel_loss=vel_loss.item(), base_loss=loss.item())