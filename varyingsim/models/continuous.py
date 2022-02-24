
from varyingsim.models.transformer import EncoderLayer
from varyingsim.models.cl_model import ContinualLearningModel
from varyingsim.models.nearest_embed import NearestEmbed
from varyingsim.models.feed_forward import FeedForward

import torch
import torch.nn.functional as F
import numpy as np

class SingleEncoder(ContinualLearningModel):

    def __init__(self, env, context_size, d_in, d_out, hidden_sizes, device='cpu'):
        super(SingleEncoder, self).__init__(env, context_size, device=device)
        self.model = FeedForward(d_in, d_out, hidden_sizes)

    def forward(self, datum):
        s = datum['context_obs'].to(self.device)
        a = datum['context_act'].to(self.device)
        sp = datum['context_obs_prime'].to(self.device)
        x = torch.cat([s, a, sp], dim=-1).float()
        # TODO: change this to have a "combine" method? (average, concat, something else)
        return torch.mean(self.model(x), dim=1) # average pool all results

class MultipleEncoder(ContinualLearningModel):

    def __init__(self, env, context_size, d_in, d_out, hidden_sizes, device='cpu', combine_method='mean'):
        super(MultipleEncoder, self).__init__(env, context_size, device=device)
        self.model = FeedForward(d_in, d_out, hidden_sizes)
        self.combine_method = combine_method

        if self.combine_method == 'max':
            self.combine_fn = self.max_combine
        elif self.combine_method == 'mean':
            self.combine_fn = self.mean_combine

    def max_combine(self, x):
        return torch.max(x, dim=1)[0]

    def mean_combine(self, x):
        return torch.mean(x, dim=1)

    def forward(self, datum):
        s = datum['context_obs'].to(self.device)
        a = datum['context_act'].to(self.device)
        sp = datum['context_obs_prime'].to(self.device)
        x = torch.cat([s, a, sp], dim=-1).float()
        outputs = self.model(x)
        return self.combine_fn(outputs)

class RNNEncoder(ContinualLearningModel):

    def __init__(self, env, context_size, d_in, d_out, hidden_sizes, device='cpu', combine_method='mean'):
        super(RNNEncoder, self).__init__(env, context_size, device=device)

        self.encoder = torch.nn.LSTM(d_in, hidden_sizes[0], len(hidden_sizes), batch_first=True)
        self.linear = torch.nn.Linear(hidden_sizes[0], d_out)

    def forward(self, datum):
        s = datum['context_obs'].to(self.device)
        a = datum['context_act'].to(self.device)
        sp = datum['context_obs_prime'].to(self.device)
        x = torch.cat([s, a, sp], dim=-1).float()
        _, (h, c) = self.encoder(x)
        last_hidden = h[-1]
        output = self.linear(last_hidden)
        return output

class TransformerEncoder(ContinualLearningModel):
    def __init__(self, env, context_size, d_in, d_out, d_inner, n_layers, n_head, d_k, d_v, device='cpu', dropout=0.1):
        super(TransformerEncoder, self).__init__(env, context_size, device=device)
        self.layers = torch.nn.ModuleList([
            EncoderLayer(d_in, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)
        ])
    
    def forward(self, datum):
        s = datum['context_obs'].to(self.device)
        a = datum['context_act'].to(self.device)
        sp = datum['context_obs_prime'].to(self.device)
        x = torch.cat([s, a, sp], dim=-1).float()

        for enc_layer in self.layers:
            x, _ = enc_layer(x)

class SimpleDecoder(ContinualLearningModel):
    def __init__(self, env, context_size, d_in, d_out, hidden_sizes, device='cpu'):
        super(SimpleDecoder, self).__init__(env, context_size, device=device)
        self.model = FeedForward(d_in, d_out, hidden_sizes)
    
    def forward(self, datum, latent):
        obs = datum['obs'].to(self.device)
        act = datum['act'].to(self.device)

        x = torch.cat([obs, act, latent], dim=-1)
        return self.model(x)

class ContinualLatent(ContinualLearningModel):
    """
        A simple class that combines an encoder and decoder to produce an output
        Can return the latent zs used in the forward pass
    """
    def __init__(self, env, context_size, encoder, decoder, ee_coef=0.0, device='cpu', obs_transform=None):
        super(ContinualLatent, self).__init__(env, context_size, device, obs_transform)
        self.encoder = encoder # takes in history and outputs a vector
        self.decoder = decoder
        self.ee_coef = ee_coef

    def forward(self, datum, ret_extra=False):
        z = self.encoder(datum)
        if ret_extra:
            return self.decoder(datum, z), {'latents': z}
        return self.decoder(datum, z)
    
    def loss(self, datum, return_info=False, wandb_obj=None, train=True):
        base_loss, base_info = super().loss(datum, True, wandb_obj=wandb_obj)

        loss = base_loss
        if return_info:
            return loss, base_info
        return loss

