
import mujoco_py
import torch
import torch.nn as nn
import torch.nn.functional as F
from varyingsim.util.view import qmul

class QPOSType:
    ADD_REGULAR = 'ADD_REGULAR'
    ADD_QUAT = 'ADD_QUAT'

    def __init__(self, add_type, size):
        self.type = add_type
        self.size = size

        if type == self.ADD_QUAT and size != 4:
            raise Exception('quaternion size must be 4!')
    
    def __repr__(self):
        return self.type + " " + str(self.size)

class MuJoCoDelta(nn.Module):
    
    def __init__(self, env):
        super(MuJoCoDelta, self).__init__()
        # first get the properties of qpos, qvel to see if we have rotation -> quaternion math
        self.env = env
        self.add_types = []
        for name in env.model.joint_names:
            jnt_id = env.model.joint_name2id(name)
            jnt_type = env.model.jnt_type[jnt_id]
            if jnt_type == mujoco_py.const.JNT_FREE:
                self.add_types.append(QPOSType(QPOSType.ADD_REGULAR, 3))
                self.add_types.append(QPOSType(QPOSType.ADD_QUAT, 4))
            elif jnt_type == mujoco_py.const.JNT_BALL:
                self.add_types.append(QPOSType(QPOSType.ADD_QUAT, 4))
            elif jnt_type == mujoco_py.const.JNT_SLIDE:
                self.add_types.append(QPOSType(QPOSType.ADD_REGULAR, 1))
            elif jnt_type == mujoco_py.const.JNT_HINGLE:
                self.add_types.append(QPOSType(QPOSType.ADD_REGULAR, 1))
        
        total_size = 0
        for add_type in self.add_types:
            total_size += add_type.size
        assert total_size == self.env.model.nq

    def forward(self, qpos, qvel, delta_qpos, delta_qvel):
        new_qvel = qvel + delta_qvel

        new_qpos = torch.zeros_like(qpos)
        idx = 0

        # TODO: not sure if pytorch handles indexing weirdly for gradeients? need to stack?
        for add_type in self.add_types:
            if add_type.type == add_type.ADD_REGULAR:
                new_qpos[..., idx:idx + add_type.size] = qpos[..., idx:idx + add_type.size] + delta_qpos[..., idx:idx + add_type.size]
            else:
                norm_fact = torch.norm(delta_qpos[..., idx:idx + add_type.size], dim=-1).unsqueeze(-1)
                delta_normalized = delta_qpos[..., idx:idx + add_type.size] / norm_fact
                new_qpos[..., idx:idx + add_type.size] = qmul(qpos[..., idx:idx + add_type.size], delta_normalized)

            idx += add_type.size
        return new_qpos, new_qvel

    def mse_loss(self, qpos_hat, qvel_hat, qpos, qvel):
        y_hat = torch.cat([qpos_hat, qvel_hat], dim=-1)
        y = torch.cat([qpos, qvel], dim=-1)
        return F.mse_loss(y_hat, y)

    def quat_loss(self, qpos_hat, qvel_hat, qpos, qvel):
        return 0.0

# class MuJoCoType:
#     def __init__(self, env, joint_name, qpos=True, qvel=True):
#         self.env = env
#         self.joint_name = joint_name
#         self.qpos = qpos
#         self.qvel = qvel

# class MuJoCoTypeSpec:
#     def __init__(self, env, spec_list):
#         self.spec_list = spec_list
#         self.add_types = []
#         self.env = env

#         for spec in self.spec_list:
#             jnt_id = env.model.joint_name2id(spec.joint_name)
#             jnt_type = env.model.jnt_type[jnt_id]
#             if jnt_type == mujoco_py.const.JNT_FREE:
#                 self.add_types.append(QPOSType(QPOSType.ADD_REGULAR, 3))
#                 self.add_types.append(QPOSType(QPOSType.ADD_QUAT, 4))
#             elif jnt_type == mujoco_py.const.JNT_BALL:
#                 self.add_types.append(QPOSType(QPOSType.ADD_QUAT, 4))
#             elif jnt_type == mujoco_py.const.JNT_SLIDE:
#                 self.add_types.append(QPOSType(QPOSType.ADD_REGULAR, 1))
#             elif jnt_type == mujoco_py.const.JNT_HINGLE:
#                 self.add_types.append(QPOSType(QPOSType.ADD_REGULAR, 1))
        
#         self.total_size = 0
#         for add_type in self.add_types:
#             self.total_size += add_type.size
#         assert self.total_size == self.env.model.nq
    
#     @property
#     def dim(self):
#         return self.total_size

#     @classmethod
#     def all(cls, env):
#         spec_list = []
#         for name in env.model.joint_names:
#             spec_list.append(MuJoCoType(env, name))
#         return cls(env, spec_list)
    
#     @classmethod
#     def names(cls, env, names):
#         spec_list = []
#         for name in names:
#             spec_list.append(MuJoCoType(env, name))
#         return cls(env, spec_list)

# class MuJoCoSubDelta(nn.Module):

#     def __init__(self, env, in_spec, out_spec):
#         super(MuJoCoSubDelta, self).__init__()
#         self.env = env
#         self.in_spec = in_spec
#         self.out_spec = out_spec

#         # TODO: make sure out spec is a subset of in spec

#     def forward(self, input, delta):
#         assert input.shape[-1] == self.d_in
#         assert delta.shape[-1] == self.d_out

#         qpos_parts = {}
#         qvel_parts = {}

#         for spec in self.in_spec.spec_list:
#             if spec.qpos:
#                 qpos_parts[spec.joint_name] = 



#         new_qvel = qvel + delta_qvel

#         new_qpos = torch.zeros_like(qpos)
#         idx = 0

#         # TODO: not sure if pytorch handles indexing weirdly for gradeients? need to stack?
#         for add_type in self.add_types:
#             if add_type.type == add_type.ADD_REGULAR:
#                 new_qpos[..., idx:idx + add_type.size] = qpos[..., idx:idx + add_type.size] + delta_qpos[..., idx:idx + add_type.size]
#             else:
#                 norm_fact = torch.norm(delta_qpos[..., idx:idx + add_type.size], dim=-1).unsqueeze(-1)
#                 delta_normalized = delta_qpos[..., idx:idx + add_type.size] / norm_fact
#                 new_qpos[..., idx:idx + add_type.size] = qmul(qpos[..., idx:idx + add_type.size], delta_normalized)

#             idx += add_type.size
#         return new_qpos, new_qvel


#     @property
#     def d_in(self):
#         return self.in_spec.dim

#     @property
#     def d_out(self):
#         return self.out_spec.dim