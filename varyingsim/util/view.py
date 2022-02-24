import numpy as np
import quaternion
import torch
import torchgeometry

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    taken from https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    
    original_shape = q.shape
    
    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def obs_to_relative(obs, act):
    """
        assumes numpy input
        outputs an observation that is position-independent, all relative to the pusher
    """
    qpos, qvel = obs[:9], obs[9:]
    xyz_box = qpos[:3]
    xyz_pusher = np.concatenate([qpos[-2:], np.array([0.1])]) # the height of the pusher
    relative_xyz = xyz_box - xyz_pusher

    angle = act[1]

    # TODO can do this without the inverse.
    M = np.array([  [np.cos(angle), -np.sin(angle), 0, xyz_pusher[0] - 0.2], # because qpos is relative and we start at (-0.2, 0, 0.1)
                    [np.sin(angle),  np.cos(angle), 0, xyz_pusher[1]],
                    [0,              0,             1, xyz_pusher[2]],
                    [0,              0,             0, 1]])
    box_homog = np.concatenate([xyz_box, np.array([1.0])])

    M_inv = np.linalg.inv(M)

    coords_relative = M_inv @ box_homog

    q_rot = quaternion.from_euler_angles(0, 0, angle)
    orig_quat = np.quaternion(*obs[3:7])
    quat_relative = orig_quat * q_rot
    # note we're excluding the pusher pos now that everything is relative
    return np.concatenate([coords_relative, quaternion.as_float_array([quat_relative])[0], qvel])
    
def get_transform(obs, device='cpu'):
    if len(obs.shape) == 1:
        obs_batch = False
        obs = obs.unsqueeze(0)
    else:
        obs_batch = True
  
    qpos, qvel = obs[..., :9], obs[..., 9:]
    box_quat = obs[..., 3:7]

    all_angles = torchgeometry.quaternion_to_angle_axis(box_quat)
    M = torchgeometry.angle_axis_to_rotation_matrix(all_angles)
    M_inv = torch.inverse(M)
    return M, M_inv, all_angles

def global_to_local_obs(obs, M_inv, all_angles, device='cpu'):
    if len(obs.shape) == 1:
        obs_batch = False
        obs = obs.unsqueeze(0)
    else:
        obs_batch = True
  
    qpos, qvel = obs[..., :9], obs[..., 9:]
    box_quat = obs[..., 3:7]
    xyz_box = qpos[..., :3]
    xy_pusher = qpos[:, -2:]
    xy_pusher[:, 0] = xy_pusher[:, 0] - (torch.ones(qpos.shape[0], 1, device=device) * 0.2) # qpos -> global
    xyz_pusher = torch.cat([xy_pusher, torch.ones(qpos.shape[0], 1, device=device) * 0.1], dim=1) # the height of the pusher
    relative_box = xyz_box - xyz_box
    relative_pusher = xyz_box - xyz_pusher

    vel_box = qvel[..., :3]
    angle_vel_box = qvel[..., 3:6]
    xy_pusher_vel = qvel[..., 6:]


    box_homog = torch.cat([relative_box, torch.ones(qpos.shape[0], 1, device=device)], dim=1).unsqueeze(-1)
    box_vel_homog = torch.cat([vel_box, torch.ones(qpos.shape[0], 1, device=device)], dim=1).unsqueeze(-1)
    box_angle_vel_homog = torch.cat([angle_vel_box, torch.ones(qpos.shape[0], 1, device=device)], dim=1).unsqueeze(-1)
    pusher_homog = torch.cat([relative_pusher, torch.ones(qpos.shape[0], 1, device=device)], dim=1).unsqueeze(-1)
    pusher_vel_homog = torch.cat([xy_pusher_vel, torch.zeros(qpos.shape[0], 1, device=device), torch.ones(qpos.shape[0], 1, device=device)], dim=1).unsqueeze(-1)

    box_coords_relative = torch.bmm(M_inv, box_homog).squeeze(-1)[..., :3]
    box_vel_relative = torch.bmm(M_inv, box_vel_homog).squeeze(-1)[..., :3]
    box_angle_vel_relative = torch.bmm(M_inv, box_angle_vel_homog).squeeze(-1)[..., :3]
    pusher_relative = torch.bmm(M_inv, pusher_homog).squeeze(-1)[..., :2]
    pusher_vel_relative = torch.bmm(M_inv, pusher_vel_homog).squeeze(-1)[..., :2]

    q_rot = torchgeometry.angle_axis_to_quaternion(-all_angles)
    orig_quat = obs[..., 3:7]
    quat_relative = qmul(orig_quat, q_rot)

    x = torch.cat([box_coords_relative, quat_relative, pusher_relative, \
            box_vel_relative, box_angle_vel_relative, pusher_vel_relative], dim=1)

    if obs_batch:
        return x
    else:
        return x.squeeze(0)

def obs_to_relative_torch(obs, device='cpu'):
    _, M_inv, all_angles = get_transform(obs, device=device)
    return global_to_local_obs(obs, M_inv, all_angles, device=device)

def euler_from_quaternion(w, x, y, z):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def z_from_quat(quats):
    # B x 4 array of quaternions
    w = quats[..., 0]
    x = quats[..., 1]
    y = quats[..., 2]
    z = quats[..., 3]

    a = 2 * (w * z + x * y)
    b = 1 - 2 * (y * y + z * z)
    return np.arctan2(a, b)

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qw, qx, qy, qz]

def push_box_state_to_xyt(state):
    xy = state[..., :2]
    orient = state[..., 3:7]
    theta = z_from_quat(orient)
    return np.concatenate([xy, np.expand_dims(theta, 1)], axis=1)

def push_box_state_to_xycs(state):
    xy = state[..., :2]
    orient = state[..., 3:7]
    theta = z_from_quat(orient)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.concatenate([xy, np.expand_dims(cos_theta, -1),  np.expand_dims(sin_theta, -1)], axis=-1)

def slide_box_state_to_xyt(state):
    xy = state[..., :2]
    orient = state[..., 3:7]
    theta = z_from_quat(orient)
    return np.concatenate([xy, np.expand_dims(theta, 1)], axis=1)

def slide_box_state_to_xy(state):
    return state[..., :2]

def slide_box_state_to_xycs(state):
    xy = state[..., :2]
    orient = state[..., 3:7]
    theta = z_from_quat(orient)
    print(xy.shape)
    cs = np.stack([np.cos(theta), np.sin(theta)])
    print(cs.shape)
    return np.concatenate([xy, cs], axis=1)

def slide_box_state_to_xyt_velocity(state):
    xy = state[..., :2]
    orient = state[..., 3:7]
    theta = z_from_quat(orient)
    dpos = state[..., 9:11]
    dtheta = state[..., 14]
    return np.concatenate([xy, np.expand_dims(theta, 1), dpos, np.expand_dims(dtheta, 1)], axis=1)

def hopper_state_to_obs(state):
    return state[..., 1:]

def swimmer_state_to_obs(state):
    return state[..., 2:]