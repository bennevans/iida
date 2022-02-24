from varyingsim.envs.varyingenv import VaryingEnv
import numpy as np
from gym import utils

class SwimmerEnv(VaryingEnv):
    def __init__(self):
        VaryingEnv.__init__(self, 'swimmer.xml', 4)
        utils.EzPickle.__init__(self)

        self.torso_body_idx = self.model.body_name2id('torso')
        self.mid_body_idx = self.model.body_name2id('mid')
        self.back_body_idx = self.model.body_name2id('back')
        self.torso_geom_idx = self.model.body_geomadr[self.torso_body_idx]
        self.mid_geom_idx = self.model.body_geomadr[self.mid_body_idx]
        self.back_geom_idx = self.model.body_geomadr[self.back_body_idx]

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()

    # getters
    def get_torso_mass(self):
        return self.model.body_mass[self.torso_body_idx]

    def get_mid_mass(self):
        return self.model.body_mass[self.mid_body_idx]

    def get_back_mass(self):
        return self.model.body_mass[self.back_body_idx]

    def get_torso_length(self):
        return self.model.geom_size[self.torso_geom_idx, 0]

    def get_mid_length(self):
        return self.model.geom_size[self.mid_geom_idx, 0]

    def get_back_length(self):
        return self.model.geom_size[self.back_geom_idx, 0]

    # setters
    def set_torso_mass(self, val):
        self.model.body_mass[self.torso_body_idx] = val

    def set_mid_mass(self, val):
        self.model.body_mass[self.mid_body_idx] = val

    def set_back_mass(self, val):
        self.model.body_mass[self.back_body_idx] = val

    def set_torso_length(self, val):
        self.model.geom_size[self.torso_geom_idx, 0] = val

    def set_mid_length(self, val):
        self.model.geom_size[self.mid_geom_idx, 0] = val

    def set_back_length(self, val):
        self.model.geom_size[self.back_geom_idx, 0] = val

    def get_fovs(self):
        return [self.get_torso_mass(), self.get_mid_mass(), self.get_back_mass(), 
            self.get_torso_length(), self.get_mid_length(), self.get_back_length()]

    def get_fov_names(self):
        return ['torso_mass', 'mid_mass', 'back_mass', 'torso_length', 'mid_length', 'back_length']


    