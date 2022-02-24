import numpy as np
from gym import utils
from varyingsim.envs.varyingenv import VaryingEnv


class AntEnv(VaryingEnv):
    def __init__(self):
        VaryingEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

        self.wind_speed = 4.0
        self.set_wind_direction(0.0)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_wind_direction(self):
        return self.wind_dir
    
    def set_wind_direction(self, val):
        self.wind_dir = val
        wind_x, wind_y = self.wind_speed * np.cos(val), self.wind_speed * np.sin(val)
        self.model.opt.wind[0] = wind_x
        self.model.opt.wind[1] = wind_y

    def get_fovs(self):
        return [self.get_wind_direction()]
    
    def get_fov_names(self):
        return ['wind_direction']