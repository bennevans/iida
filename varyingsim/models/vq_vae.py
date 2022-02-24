
from varyingsim.models.cl_model import ContinualLearningModel
from varyingsim.models.nearest_embed import NearestEmbed
from varyingsim.models.feed_forward import FeedForward

import torch
import torch.nn.functional as F

class VQVAE(ContinualLearningModel):
    def __init__(self, env, context_size, encoder, decoder, k, d, vq_coef=1.0, commit_coef=0.25, ee_coef=0.0, eq_coef=0.0, device='cpu', obs_transform=None):
        super(VQVAE, self).__init__(env, context_size, device, obs_transform)
        self.encoder = encoder # takes in history and outputs a vector
        self.decoder = decoder
        self.k = k
        self.d = d
        self.emb = NearestEmbed(k, d)
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.ee_coef = ee_coef
        self.eq_coef = eq_coef

    def forward(self, datum):
        recon, z_e, emb, argmin = self.encode_decode(datum)
        return recon

    def encode_decode(self, datum, set_argmin=None):
        z_e = self.encoder(datum)

        if set_argmin:
            argmin = set_argmin
            z_q = self.emb.weight[:, argmin].detach().T
            emb = self.emb.weight[:, argmin].T
        else:
            z_q, argmin = self.emb(z_e, weight_sg=True)
            emb, _ = self.emb(z_e.detach())
        return self.decoder(datum, z_q), z_e, emb, argmin

    def loss(self, datum, return_info=False):
        recon, z_e, emb, argmin = self.encode_decode(datum)


        # mse_loss = F.mse_loss(recon, datum['obs_prime'])
        base_loss, base_info = super().loss(datum, True)
        vq_loss = F.mse_loss(emb, z_e.detach())
        commit_loss = F.mse_loss(z_e, emb.detach())
        ee_loss = calc_ee_loss(self.encoder, datum)

        total_loss = base_loss + self.vq_coef * vq_loss + self.commit_coef * commit_loss + self.ee_coef * ee_loss

        if return_info:
            info = dict(argmin=argmin.detach().cpu().numpy(),
                vq_loss=vq_loss.item(), commit_loss=commit_loss.item(), ee_loss=ee_loss.item())
            info.update(base_info)
            return total_loss, info
        return total_loss
    
    def save(self, f):
        torch.save(self.state_dict(), f)
