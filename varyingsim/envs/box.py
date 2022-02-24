
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class BoxEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, include_fov=True, set_param_fn=None, rand_reset=False): 
        self.include_fov = include_fov
        self.t = -1
        self.set_param_fn = set_param_fn
        self.rand_reset = rand_reset

        mujoco_env.MujocoEnv.__init__(self, '/home/benevans/projects/varyingsim/varyingsim/assets/box.xml', 2)
        utils.EzPickle.__init__(self)

    def step(self, a):
        if self.set_param_fn:
            self.set_param_fn(self, self.t)

        self.t += 1
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward = 0.0
    
        return ob, reward, False, dict(t=self.t)

    def set_mass(self, mass):
        self.model.body_mass[1] = mass
    
    def set_box_friction(self, friction):
        self.model.geom_friction[1, 0] = friction    
    
    def set_floor_friction(self, friction):
        self.model.geom_friction[0, 0] = friction

    def set_actuator_gear(self, gear):
        self.model.actuator_gear[:, 0] = gear

    def get_mass(self):
        return self.model.body_mass[1]

    def get_box_friction(self):
        return self.model.geom_friction[1, 0]    
    
    def get_floor_friction(self):
        return self.model.geom_friction[0, 0]

    def get_actuator_gear(self):
        return self.model.actuator_gear[:, 0]

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        ret = np.concatenate([qpos.flat, qvel.flat])
        if self.include_fov:
            bf = self.get_box_friction()
            ff = self.get_floor_friction()
            mass = self.get_mass()
            gear = self.get_actuator_gear()
            ret = np.concatenate([ret, [bf], [ff], [mass], gear])
        return ret

    def reset_model(self):
        if self.rand_reset:
            qpos_start = self.init_qpos.copy()
            qvel_start = self.init_qvel.copy()
            qpos_start[:2] += np.random.randn(2) / 2.0
            qvel_start[:2] += np.random.randn(2) / 10.0
            self.set_state(
                qpos_start, 
                qvel_start
            )
        else:
            self.set_state(
                self.init_qpos, 
                self.init_qvel
            )
        self.t = -1
        return self._get_obs()