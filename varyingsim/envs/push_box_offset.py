
from varyingsim.envs.varyingenv import VaryingEnv
from varyingsim.util.view import euler_from_quaternion, euler_to_quaternion
import numpy as np

class PushBoxOffset(VaryingEnv):

    def __init__(self, include_fov=False, set_param_fn=None, rand_reset=True, reset_list=None):
        self.d_pos = np.array([0, 0.])
        self.push_start = 0.0
        self.push_done = False
        self.rand_reset = rand_reset
        self.reset_list = reset_list

        VaryingEnv.__init__(self, 'push_box_offset.xml', 4, include_fov=include_fov, set_param_fn=set_param_fn)

    def get_floor_friction(self):
        return self.model.geom_friction[0, 0]

    def get_box_mass(self):
        return self.model.body_mass[1]

    def get_box_friction(self):
        return self.model.geom_friction[1, 0]

    def get_pusher_friction(self):
        return self.model.geom_friction[2, 0]

    def get_com_offset(self):
        return self.model.body_pos[2, 1]

    def get_box_size(self):
        return self.model.geom_size[1, 0]

    def set_floor_friction(self, val):
        self.model.geom_friction[0, 0] = val

    def set_box_mass(self, val):
        self.model.body_mass[1] = val

    def set_box_friction(self, val):
        self.model.geom_friction[1, 0] = val

    def set_pusher_friction(self, val):
        self.model.geom_friction[2, 0] = val

    def set_com_offset(self, val): 
        self.model.body_pos[2, 1] = val

    def set_box_size(self, val):
        self.model.geom_size[1, :] = val

    def get_fovs(self):
        return [self.get_com_offset(), self.get_box_mass(), self.get_box_friction(), 
            self.get_floor_friction(), self.get_pusher_friction(), self.get_box_size()]
    
    def get_fov_names(self):
        return ['com_offset', 'box_mass', 'box_friction', 'floor_friction', 'pusher_friction', 'box_size']

    def do_simulation(self, ctrl, n_frames):
        start_x = ctrl[0]
        start_y = ctrl[1]
        push_angle = ctrl[2]
        push_vel = ctrl[3]
        push_time = ctrl[4]
        restart_push = ctrl[5]

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
            if self.reset_list:
                print(self.episode, self.reset_list[self.episode])
                rand_x = self.reset_list[self.episode][0]
                rand_y = self.reset_list[self.episode][1]
                rand_angle = self.reset_list[self.episode][2]
            else:
                rand_x = np.random.random()- 0.5
                rand_y = np.random.random()- 0.5
                rand_angle = np.random.random() * 2 * np.pi
                
            init_qpos[0] = rand_x
            init_qpos[1] = rand_y
            init_qpos[3:7] = euler_to_quaternion(0, 0, rand_angle)

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