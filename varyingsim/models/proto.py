
from varyingsim.models.cl_model import ContinualLearningModel
from varyingsim.models.nearest_embed import NearestEmbed
from varyingsim.models.feed_forward import FeedForward

import torch
import torch.nn.functional as F

from copy import deepcopy
from varyingsim.util.model import copy_weights, entropy, entropy_from_log, weight_init

import wandb

# TODO: proto vqvae?

class GenericEncoder(ContinualLearningModel):

    def __init__(self, env, context_size, d_in, d_out, hidden_sizes, device='cpu'):
        super(GenericEncoder, self).__init__(env, context_size, device=device)
        self.model = FeedForward(d_in, d_out, hidden_sizes)
        self.apply(weight_init)

    def forward(self, s, a, sp):
        s = s.to(self.device)
        a = a.to(self.device)
        sp = sp.to(self.device)
        x = torch.cat([s, a, sp], dim=-1).float()
        # TODO: change this to have a "combine" method? (average, concat, something else)
        return self.model(x)

class Proto(ContinualLearningModel):
    def __init__(self, env, context_size, encoder, decoder, num_protos, proto_dim, temp, proto_coef, tau,
            device='cpu', obs_transform=None, num_iters=3, latent_type='soft', base_coef=1.0,
            single_entropy_coef=0.0, batch_entropy_coef=0.0, use_predictor=True):
        super(Proto, self).__init__(env, context_size, device, obs_transform)
        self.encoder = encoder # takes in history and outputs a vector
        self.encoder_target = deepcopy(self.encoder)
        self.decoder = decoder
        self.protos = torch.nn.Linear(proto_dim, num_protos, bias=False)
        self.temp = temp
        self.proto_coef = proto_coef
        self.tau = tau
        self.num_iters = num_iters
        self.latent_type = latent_type
        self.use_predictor = use_predictor
        if use_predictor:
            self.predictor = torch.nn.Sequential(torch.nn.Linear(proto_dim, 256), 
                                            torch.nn.ReLU(), torch.nn.Linear(256, proto_dim))
        else:
            self.predictor = torch.nn.Sequential()
            
        self.base_coef = base_coef
        self.single_entropy_coef = single_entropy_coef
        self.batch_entropy_coef = batch_entropy_coef

        self.apply(weight_init)
        self.num_infs = 0

    def forward(self, datum):
        recon, extra = self.encode_decode(datum)
        return recon

    def encode_decode(self, datum, set_argmin=None):
        s = datum['obs']
        a = datum['act']
        sp = datum['obs_prime']

        context_s = datum['context_obs']
        context_a = datum['context_act']
        context_sp = datum['context_obs_prime']

        z = self.encoder(s, a, sp)
        z = self.predictor(z)
        with torch.no_grad():
            context_z = self.encoder_target(context_s, context_a, context_sp).squeeze(1)

        C = self.protos.weight.data.clone()
        C = F.normalize(C, dim=1, p=2)
        self.protos.weight.data.copy_(C)
        
        z = F.normalize(z, dim=1, p=2)
        context_z = F.normalize(context_z, dim=1, p=2)

        scores_z = self.protos(z)
        log_p_s = F.log_softmax(scores_z / self.temp, dim=1)

        with torch.no_grad():
            scores_context_z = self.protos(context_z)
            q = self.sinkhorn(scores_context_z)

        if self.latent_type == 'soft':
            latent = q @ self.protos.weight
        elif self.latent_type == 'q':
            latent = q
        else:
            latent = context_z

        return self.decoder(datum, latent), dict(context_z=context_z, log_p_s=log_p_s, q=q, z=z, latent=latent)

    def sinkhorn(self, scores):
        def remove_infs(x):
            m = x[torch.isfinite(x)].max().item()
            x[torch.isinf(x)] = m
            return x

        Q = scores / self.temp
        Q -= Q.max()

        Q = torch.exp(Q).T
        Q = remove_infs(Q)
        Q /= Q.sum()

        r = torch.ones(Q.shape[0], device=Q.device) / Q.shape[0]
        c = torch.ones(Q.shape[1], device=Q.device) / Q.shape[1]
        for it in range(self.num_iters):
            u = Q.sum(dim=1)
            self.num_infs += torch.isinf(u).sum()
            u = remove_infs(r / u)
            Q *= u.unsqueeze(dim=1)
            Q *= (c / Q.sum(dim=0)).unsqueeze(dim=0)
        Q = Q / Q.sum(dim=0, keepdim=True)
        return Q.T

    def loss(self, datum, return_info=False, wandb_obj=None):
        recon, extra = self.encode_decode(datum)
        log_p_s, q = extra['log_p_s'], extra['q']

        if wandb_obj:
            images = {
                'p_image': wandb.Image(log_p_s.exp().detach().cpu().numpy()),
                'q_image': wandb.Image(q.detach().cpu().numpy())
            }
            wandb_obj.log(images)

        base_loss, base_info = super().loss(datum, True)
        proto_loss = -(q * log_p_s).sum(dim=1).mean()

        single_entropy_loss = entropy_from_log(log_p_s)
        batch_entropy_loss = entropy_from_log(log_p_s.exp().mean(dim=0))

        total_loss =    self.base_coef * base_loss + self.proto_coef * proto_loss + \
                        self.single_entropy_coef * single_entropy_loss + self.batch_entropy_coef * batch_entropy_loss

        if return_info:
            p_ent = entropy_from_log(log_p_s).item()
            q_ent = entropy(q).item()

            info = dict(proto_loss=proto_loss.item(), p_entropy=p_ent,
                q_entropy=q_ent,single_entropy_loss=single_entropy_loss,
                batch_entropy_loss=batch_entropy_loss, num_infs=self.num_infs)
            info.update(base_info)
            return total_loss, info
        return total_loss

    def update(self):
        copy_weights(self.encoder, self.encoder_target, self.tau)