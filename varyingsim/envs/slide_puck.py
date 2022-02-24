
from varyingsim.envs.varyingenv import VaryingEnv
from varyingsim.util.view import euler_from_quaternion, euler_to_quaternion
import numpy as np

class SlidePuck(VaryingEnv):

    def __init__(self, include_fov=False, set_param_fn=None, rand_reset=False, reset_list=None):
        self.d_pos = np.array([0, 0.])
        self.push_start = 0.0
        self.push_done = False
        self.rand_reset = rand_reset
        self.reset_list = reset_list
        self.min_puck_rad = 0.025
        self.max_puck_rad = 0.2
        self.k = None
        self.r = None
        self.reset_idx = None

        VaryingEnv.__init__(self, 'slide_puck.xml', 1, include_fov=include_fov, set_param_fn=set_param_fn)

        self.floor_geom_idx = self.model.geom_name2id('plane')
        self.puck_geom_idx = self.model.geom_name2id('puck')
        self.pusher_geom_idx = self.model.geom_name2id('pusher')
        self.puck_body_idx = self.model.body_name2id('puck')

    def get_floor_friction(self):
        return self.model.geom_friction[self.floor_geom_idx, 0]

    def get_puck_mass(self):
        return self.model.body_mass[self.puck_body_idx]

    def get_puck_friction(self):
        return self.model.geom_friction[self.puck_geom_idx, 0]

    def get_friction(self):
        return self.get_puck_friction()

    def get_pusher_friction(self):
        return self.model.geom_friction[self.pusher_geom_idx, 0]

    def get_puck_size(self):
        return self.model.geom_size[self.puck_geom_idx, 0]

    def get_table_tilt(self):
        return euler_from_quaternion(*self.model.geom_quat[self.floor_geom_idx])

    def get_table_tilt_x(self):
        return self.get_table_tilt()[0]

    def get_table_tilt_y(self):
        return self.get_table_tilt()[1]
    
    def get_wind_x(self):
        return self.model.opt.wind[0]

    def get_wind_y(self):
        return self.model.opt.wind[1]

    def get_damping(self):
        return self.model.dof_damping[0]

    def set_floor_friction(self, val):
        self.model.geom_friction[self.floor_geom_idx, 0] = val

    def set_puck_mass(self, val):
        self.model.body_mass[self.puck_body_idx] = val

    def set_puck_friction(self, val):
        self.model.geom_friction[self.puck_geom_idx, 0] = val

    def set_friction(self, val):
        self.set_puck_friction(val)
        self.set_floor_friction(val)

    def set_pusher_friction(self, val):
        self.model.geom_friction[self.pusher_geom_idx, 0] = val

    def set_puck_size(self, val):
        self.model.geom_size[self.puck_geom_idx, 0] = val

    def set_table_tilt(self, x, y, z):
        quat = euler_to_quaternion(x, y, z)
        self.model.geom_quat[self.floor_geom_idx] = quat

    def set_table_tilt_x(self, val):
        self.set_table_tilt(val, self.get_table_tilt_y(), 0)

    def set_table_tilt_y(self, val):
        self.set_table_tilt(self.get_table_tilt_x(), val, 0)
    
    def set_wind_x(self, val):
        self.model.opt.wind[0] = val

    def set_wind_y(self, val):
        self.model.opt.wind[1] = val

    def set_damping(self, val):
        self.model.dof_damping[:6] = val

    def get_fovs(self):
        return [self.get_puck_size(), self.get_puck_mass(), self.get_friction(), self.get_pusher_friction(), 
            self.get_table_tilt_x(), self.get_table_tilt_y(), self.get_wind_x(), self.get_wind_y(), self.get_damping()]
    
    def get_fov_names(self):
        return ['puck_size', 'puck_mass', 'friction', 'pusher_friction', 
                'table_tilt_x', 'table_tilt_y', 'wind_x', 'wind_y', 'damping']

    def do_simulation(self, ctrl, n_frames):
        start_x = ctrl[0]
        start_y = ctrl[1]
        push_angle = ctrl[2]
        push_vel = max(ctrl[3], 1e-3)
        restart_push = ctrl[4]
        
        push_dist = self.max_puck_rad
        push_time = push_dist / push_vel


        if restart_push:
            # recalculate dx, dy, just apply it 
            self.d_pos = -push_vel * np.array([np.cos(push_angle), np.sin(push_angle)]) / self.frame_skip
            self.push_start = self.sim.data.time

        for _ in range(n_frames):
            qpos = self.sim.data.qpos
            qvel = self.sim.data.qvel
            push_xy = self.sim.data.body_xpos[3][0:2] # xy in global coordinates
            x_global, y_global = push_xy

            if restart_push:
                # set to initial position
                qpos[7:9] = np.array([start_x, start_y])
                restart_push = False

            self.push_done = push_time < (self.sim.data.time - self.push_start)
            # TODO change this to be true when the push is done or when the puck has stopped?
            # could also do a puck stopped flag

            if not self.push_done:
                qvel[6:8] = self.d_pos
            else:
                qvel[6:8] = [0., 0.]

            self.set_state(qpos, qvel)
            self.sim.step()
    
    def reset_model(self):
        self.push_done = False

        init_qpos = np.copy(self.init_qpos)
        
        if self.rand_reset:
            
            # if self.k is not None:
            #     print('k, r', self.k, self.r)
            #     rand_x = self.reset_list[self.r][self.k][0]
            #     rand_y = self.reset_list[self.r][self.k][1]
            #     rand_angle = self.reset_list[self.r][self.k][2]
            #     # rand_x = self.reset_list[self.k][self.r][0]
            #     # rand_y = self.reset_list[self.k][self.r][1]
            #     # rand_angle = self.reset_list[self.k][self.r][2]
            if self.reset_list and self.reset_idx < len(self.reset_list):
                # print('ep', self.episode)
                # print('reset_idx', self.reset_idx)
                # print('reset vals', self.reset_list[self.reset_idx])
                rand_x = self.reset_list[self.reset_idx][0]
                rand_y = self.reset_list[self.reset_idx][1]
                rand_angle = self.reset_list[self.reset_idx][2]
            else:
                rand_x = np.random.random()- 0.5
                rand_y = np.random.random()- 0.5
                rand_angle = np.random.random() * 2 * np.pi
                
            init_qpos[0] = np.copy(rand_x)
            init_qpos[1] = np.copy(rand_y)
            init_qpos[3:7] = np.copy(euler_to_quaternion(0, 0, rand_angle))
            # import time; time.sleep(0.1)
        self.set_state(
            init_qpos, 
            self.init_qvel
        )

        self.t = -1
        self.episode += 1
        return self._get_obs()
    def _get_obs(self):
        obs = super()._get_obs()
        return np.concatenate([obs, [self.push_done]], axis=0)

# some set_fov functions
def sin_com_slow(env, i, t):
    my_t = i * 3000 + t # TODO: general way to find the episode length / true current time
    com = np.sin(my_t / (3000.0 * 10.0) * 2 *np.pi ) / 5.0
    env.set_com_offset(com)