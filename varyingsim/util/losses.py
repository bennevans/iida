import torch
import torch.nn.functional as F

def qpos_box_loss(qpos_hat, qvel_hat, qpos, qvel):
    return F.mse_loss(qpos_hat[..., :7].cpu(), qpos[..., :7].cpu())

def qpos_qvel_box_loss(qpos_hat, qvel_hat, qpos, qvel, pos_vel_weight=0.5):
    
    return pos_vel_weight * F.mse_loss(qpos_hat[..., :7].cpu(), qpos[..., :7].cpu()) + \
    (1 - pos_vel_weight) * F.mse_loss(qvel_hat[..., :6].cpu(), qvel[..., :6].cpu())

def dist_rot_loss(qpos_hat, qvel_hat, qpos, qvel):
    xyz_hat = qpos_hat[..., :3].cpu()
    quat_hat = qpos_hat[..., 3:7].cpu()
    xyz = qpos[..., :3].cpu()
    quat = qpos[..., 3:7].cpu()
    dist_loss = F.mse_loss(xyz_hat, xyz)
    rot_loss = 2 * torch.mean(torch.arccos(torch.abs(torch.sum(quat_hat *quat, dim=-1))))
    print(dist_loss.item(), rot_loss.item())
    return dist_loss + rot_loss
