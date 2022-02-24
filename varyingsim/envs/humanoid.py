from varyingsim.envs.varyingenv import VaryingEnv
import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from varyingsim.envs.varyingenv import VaryingEnv

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class HumanoidEnv(VaryingEnv):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
        utils.EzPickle.__init__(self)

        self.left_thigh_body_idx = self.model.body_name2id('left_thigh')
        self.right_thigh_body_idx = self.model.body_name2id('right_thigh')
        self.left_foot_body_idx = self.model.body_name2id('left_foot')
        self.right_foot_body_idx = self.model.body_name2id('right_foot')
        self.left_upper_arm_body_idx = self.model.body_name2id('left_upper_arm')
        self.right_upper_arm_body_idx = self.model.body_name2id('right_upper_arm')
        self.left_lower_arm_body_idx = self.model.body_name2id('left_lower_arm')
        self.right_lower_arm_body_idx = self.model.body_name2id('right_lower_arm')
    
        self.left_upper_arm_geom_idx = self.model.geom_name2id('left_uarm1')
        self.right_upper_arm_geom_idx = self.model.geom_name2id('right_uarm1')
        self.left_lower_arm_geom_idx = self.model.geom_name2id('left_larm')
        self.right_lower_arm_geom_idx = self.model.geom_name2id('right_larm')
        self.left_hand_geom_idx = self.model.geom_name2id('left_hand')
        self.right_hand_geom_idx = self.model.geom_name2id('right_hand')
    

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat,
                               data.qvel.flat])

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20

    # getters
    def get_thigh_offset(self):
        return self.model.body_pos[self.left_thigh_body_idx][1]
    
    def get_foot_offset(self):
        return self.model.body_pos[self.left_foot_body_idx][2]
    
    def get_upper_arm_length(self):
        return 2 * np.sqrt(self.model.geom_size[self.left_upper_arm_geom_idx][1]**2 / 3.0)
    
    def get_lower_arm_length(self):
        return 2 * np.sqrt(self.model.geom_size[self.left_lower_arm_geom_idx][1]**2 / 3.0)
    
    def get_upper_arm_mass(self):
        return self.model.body_mass[self.left_upper_arm_body_idx]
    
    def get_lower_arm_mass(self):
        return self.model.body_mass[self.left_lower_arm_body_idx]

    # setters
    def set_thigh_offset(self, val):
        self.model.body_pos[self.left_thigh_body_idx][1] = val
        self.model.body_pos[self.right_thigh_body_idx][1] = -val
    
    def set_foot_offset(self, val):
        self.model.body_pos[self.left_foot_body_idx][2] = val
        self.model.body_pos[self.right_foot_body_idx][2] = val
    
    def set_upper_arm_length(self, val):
        geom_size = np.sqrt(3.0 / 4.0 * val**2)
        l_pos = val + 0.02
        u_pos = val / 2.0

        self.model.geom_size[self.left_upper_arm_geom_idx][1] = geom_size
        self.model.geom_size[self.right_upper_arm_geom_idx][1] = geom_size
        
        # arms kinda float above the shoulder, but that's okay for now
        self.model.geom_pos[self.left_upper_arm_geom_idx] = np.array([u_pos, u_pos, -u_pos]).reshape(-1)
        self.model.geom_pos[self.right_upper_arm_geom_idx] = np.array([u_pos, -u_pos, -u_pos]).reshape(-1)

        self.model.body_pos[self.left_lower_arm_body_idx] = np.array([l_pos, l_pos, -l_pos]).reshape(-1)
        self.model.body_pos[self.right_lower_arm_body_idx] = np.array([l_pos, -l_pos, -l_pos]).reshape(-1)
        
    def set_lower_arm_length(self, val):
        geom_size = np.sqrt(3.0 / 4.0 * val**2)
        pos = val + 0.02
        self.model.geom_size[self.left_upper_arm_geom_idx][1] = geom_size
        self.model.geom_size[self.right_upper_arm_geom_idx][1] = geom_size
        self.model.geom_pos[self.left_hand_geom_idx] = np.array([pos, -pos, pos]).reshape(-1)
        self.model.geom_pos[self.right_hand_geom_idx] = np.array([pos, pos, pos]).reshape(-1)
    
    def set_upper_arm_mass(self, val):
        self.model.body_mass[self.left_upper_arm_body_idx] = val
    
    def set_lower_arm_mass(self, val):
        self.model.body_mass[self.left_lower_arm_body_idx] = val

    def get_fovs(self):
        return [self.get_thigh_offset(), self.get_foot_offset(), self.get_upper_arm_length(), \
            self.get_lower_arm_length(), self.get_upper_arm_mass(), self.get_lower_arm_mass()]
    
    def get_fov_names(self):
        return ['thigh_offset', 'foot_offset', 'upper_arm_length', 'lower_arm_length',\
            'upper_arm_mass', 'lower_arm_mass']