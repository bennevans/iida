import torch
import torch.nn as nn
import torch.nn.functional as F
from varyingsim.models.cl_model import ContinualLearningModel
from varyingsim.models.feed_forward import FeedForward
from varyingsim.util.model import reparameterize

class VariationalLatent(ContinualLearningModel):
    def __init__(self, env, context_size, encoder, decoder, latent_size, hidden_z_size, beta=1.0, device='cpu', obs_transform=None):
        super(VariationalLatent, self).__init__(env, context_size, device, obs_transform)
        self.encoder = encoder # takes in history and outputs a vector 
        self.decoder = decoder

        self.latent_size = latent_size
        self.hidden_z_size = hidden_z_size

        self.fc_mu = nn.Linear(latent_size, hidden_z_size)
        self.fc_sigma = nn.Linear(latent_size, hidden_z_size)

        self.beta = beta

    def encode_decode(self, datum, use_mean=False):
        h = self.encoder(datum)
        mu = self.fc_mu(h)
        logsigma = self.fc_sigma(h)

        if use_mean:
            z = mu
        else:
            z = reparameterize(mu, logsigma)

        return self.decoder(datum, z), mu, logsigma

    def forward(self, datum):
        return self.encode_decode(datum)[0]
    
    def loss(self, datum, return_info=False, wandb_obj=None):
        base_loss, base_info = super().loss(datum, True, wandb_obj=wandb_obj)
        
        recon, mu, logsigma = self.encode_decode(datum)

        kl_loss = torch.mean(-0.5 * torch.sum(1 + logsigma - mu ** 2 - logsigma.exp(), dim = 1), dim = 0)

        loss = base_loss + self.beta * kl_loss
        if return_info:
            info = {'kl_loss': kl_loss.item()}
            info.update(base_info)
            return loss, info
        return loss

# Taken from PEARL
def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=1)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=1)
    return mu, sigma_squared

def compute_kl_div(mus, vars):
    ''' compute KL( q(z|c) || r(z) ) '''
    b, d = mus.shape
    device = mus.device
    prior = torch.distributions.Normal(torch.zeros(d, device=device), torch.ones(d, device=device))
    posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(mus), torch.unbind(vars))]
    kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
    kl_div_sum = torch.sum(torch.stack(kl_divs))
    return kl_div_sum

class PEARLEncoder(ContinualLearningModel):

    def __init__(self, env, context_size, encoder, decoder, d_h, d_z, mu_hidden, sigma_hidden, beta=1.0, device='cpu', obs_transform=None):
        super(PEARLEncoder, self).__init__(env, context_size, device, obs_transform)

        self.encoder = encoder
        self.decoder = decoder
        self.mu_hidden = mu_hidden
        self.sigma_hidden = sigma_hidden
        self.beta = beta

        self.mu_network = FeedForward(d_h, d_z, mu_hidden)
        self.sigma_network = FeedForward(d_h, d_z, sigma_hidden)

    def forward(self, datum, ret_extra=False):

        s = datum['context_obs'].to(self.device)
        a = datum['context_act'].to(self.device)
        sp = datum['context_obs_prime'].to(self.device)
        x = torch.cat([s, a, sp], dim=-1).float()

        z_all = self.encoder(x)
        mu = self.mu_network(z_all)
        sigma = F.softplus(self.sigma_network(z_all))
        combined_mu, combined_sigma = _product_of_gaussians(mu, sigma)
        sampled_latents = reparameterize(combined_mu, combined_sigma)

        y_hat = self.decoder(datum, sampled_latents)

        if ret_extra:
            return y_hat, {'mu': combined_mu, 'var': combined_sigma, 'latents': sampled_latents}
        return y_hat
        

    def loss(self, datum, return_info=False, wandb_obj=None, train=False, ret_extra=True):
        (loss, info), extra = super().loss(datum, return_info=True, ret_extra=True)
        kl_loss = compute_kl_div(extra['mu'], extra['var'])

        new_loss = loss + self.beta * kl_loss

        if return_info:
            info['kl_loss'] = kl_loss
            return new_loss, info

        return new_loss
