from varyingsim.models.cl_model import ContinualLearningModel
import torch
import torch.nn.functional as F
from dcem import dcem
from varyingsim.models.lv_ebm import LatentVariableEBM
import torch.nn as nn

class CVAE(LatentVariableEBM):
    def __init__(self, env, context_size, z_dim, encoder_model, decoder_model, energy_optim, beta, hidden_z_size, device='cpu', N_rand=10, n_sample=200, n_elite=None, n_iter=10):
        super(CVAE, self).__init__(env, context_size, z_dim, decoder_model, energy_optim, device=device,
            N_rand=N_rand, n_sample=n_sample, n_elite=n_elite, n_iter=n_iter)
        
        self.encoder_model = encoder_model
        self.hidden_z_size = hidden_z_size

        self.beta = beta

        self.fc_mu = nn.Linear(hidden_z_size, z_dim)
        self.fc_sigma = nn.Linear(hidden_z_size, z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode_decode(self, datum, use_mean=False):

        x = torch.cat([datum['obs'], datum['act'], datum['obs_prime']], dim=-1)
        x_no_other = torch.cat([datum['obs'], datum['act']], dim=-1)
        h = self.encoder_model(x)
        mu = self.fc_mu(h)
        logsigma = self.fc_sigma(h)
 
        if use_mean:
            z = mu
        else:
            z = self.reparameterize(mu, logsigma)

        z_star, info = self.optim_dcem(datum) 

        return self.evaluate(x_no_other, z_star), self.evaluate(x_no_other, z), mu, logsigma

    def forward(self, datum, train=True):
        if train:
            return self.encode_decode(datum)[0]
        else:
            return self.encode_decode(datum)[1]

    def loss(self, datum, return_info=False, wandb_obj=None, train=True):        
        recon_star, recon, mu, logsigma = self.encode_decode(datum)

        if train:
            base_loss, xy_loss, theta_loss = self.recon_loss(datum, recon)
        else:
            base_loss, xy_loss, theta_loss = self.recon_loss(datum, recon_star)

        kl_loss = torch.mean(-0.5 * torch.sum(1 + logsigma - mu ** 2 - logsigma.exp(), dim = 1), dim = 0)

        loss = base_loss + self.beta * kl_loss
        if return_info:
            info = {'kl_loss': kl_loss.item()}
            info.update({'xy_loss': xy_loss.item(), 'theta_loss': theta_loss.item(), 'base_loss': base_loss.item()})
            return loss, info
        return loss
