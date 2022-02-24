from torch import optim
from varyingsim.models.cl_model import ContinualLearningModel
import torch
import torch.nn.functional as F
from dcem import dcem

class LatentVariableEBM(ContinualLearningModel):
    def __init__(self, env, context_size, z_dim, model, energy_optim, device='cpu', N_rand=10, n_sample=200, n_elite=None, n_iter=10,
        use_self=False, optim_type='cem', lr=1e-3):
        super(LatentVariableEBM, self).__init__(env, context_size, device)
        
        self.z_dim = z_dim
        # energy model
        self.model = model

        # a function / class that minimizes the energy as a functoin of z
        # self.energy_optim = energy_optim 

        self.N_rand = N_rand
        self.n_sample = n_sample
        self.n_elite = n_elite if n_elite else z_dim * 2
        self.n_iter = n_iter
        self.use_self = use_self
        self.optim_type = optim_type
        self.lr = lr

        if self.optim_type == 'cem':
            self.optim = self.optim_dcem
        elif self.optim_type == 'sample':
            self.optim = self.optim_z_sample
        elif self.optim_type == 'sgd':
            self.optim = self.optim_sgd
        elif self.optim_type == 'langevin':
            self.optim = self.optim_langevin
    
    def forward(self, datum, train=True):
        z, info = self.optim(datum)
        x = torch.cat([datum['obs'], datum['act']], dim=-1)
        return self.evaluate(x, z)

    # @staticmethod
    def optim_z_sample(self, datum):

        batch_X = torch.cat([datum['context_obs'], datum['context_act']], dim=-1)
        batch_Y = datum['context_obs_prime']

        B, n_context, dx = batch_X.shape
        z = torch.randn((self.N_rand * B, 1, self.z_dim), device=self.device)
        x_rep = batch_X.repeat(self.N_rand, 1, 1)
        y_rep = batch_Y.repeat(self.N_rand, 1, 1)
        z_rep = z.repeat(1, n_context, 1)
        x = torch.cat([x_rep, z_rep], dim=-1)
        s_hat = self.model(x)
        energies = F.mse_loss(y_rep, s_hat, reduction='none').sum(dim=[-2,-1]).view(self.N_rand, B)
        smallest = energies.argmin(dim=0)
        ret_z = z.view(self.N_rand, B, -1)[smallest, torch.arange(B)]
        return ret_z, energies[smallest, torch.arange(B)]
    
    def optim_dcem(self, datum):

        if self.use_self:
            batch_X = torch.unsqueeze(torch.cat([datum['obs'], datum['act']], dim=-1), 0)
            batch_Y = torch.unsqueeze(datum['obs_prime'], 0)
        else:
            batch_X = torch.cat([datum['context_obs'], datum['context_act']], dim=-1)
            batch_Y = datum['context_obs_prime']
        B, n_context, dx = batch_X.shape

        def f(z):
            x_rep = batch_X.repeat(1, self.n_sample, 1)
            y_rep = batch_Y.repeat(1, self.n_sample, 1)
            # z_rep = z.repeat(1, n_context, 1)
            z_rep = z.tile((1, n_context)).view(B, self.n_sample * n_context, -1)
            x = torch.cat([x_rep, z_rep], dim=-1)
            s_hat = self.model(x)
            energies = F.mse_loss(y_rep, s_hat, reduction='none').view(B, self.n_sample, n_context, -1).sum(dim=[-2, -1])
            return energies

        init_mu = torch.zeros(B, self.z_dim, device=self.device)
        init_sigma = torch.ones(B, self.z_dim, device=self.device)
        with torch.no_grad():
            min_z = dcem(f=f, nx=self.z_dim, n_batch=B, init_mu=init_mu, init_sigma=init_sigma, n_sample=self.n_sample, n_elite=self.n_elite, n_iter=self.n_iter, device=self.device)

        return min_z, None
    
    def optim_sgd(self, datum):
        # for now, do one z per element in the batch
        batch_X = torch.cat([datum['context_obs'], datum['context_act']], dim=-1)
        batch_Y = datum['context_obs_prime']
        B, n_context, dx = batch_X.shape

        # for now, start z at same place and just optimize down the hill.
        # could also sample in a region and then optimize
        z = torch.zeros((B, self.z_dim), device=self.device, requires_grad=True)

        z_optim = torch.optim.SGD([z], lr=self.lr, momentum=0.0)

        def f(z):
            z_rep = z.tile((1, n_context)).view(B, n_context, -1)
            x = torch.cat([batch_X, z_rep], dim=-1)
            s_hat = self.model(x)
            energies = F.mse_loss(batch_Y, s_hat, reduction='none').view(B, n_context, -1).sum(dim=-1)
            return energies

        for i in range(self.n_iter):
            z_optim.zero_grad()
            energies = f(z)
            loss = energies.sum()
            loss.backward()
            z_optim.step()

        return z, None
    
    def optim_langevin(self, datum):
        # for now, do one z per element in the batch
        batch_X = torch.cat([datum['context_obs'], datum['context_act']], dim=-1)
        batch_Y = datum['context_obs_prime']
        B, n_context, dx = batch_X.shape

        # for now, start z at same place and just optimize down the hill.
        # could also sample in a region and then optimize
        z = torch.zeros((B, self.z_dim), device=self.device, requires_grad=True)

        z_optim = torch.optim.SGD([z], lr=self.lr, momentum=0.0)

        def f(z):
            z_rep = z.tile((1, n_context)).view(B, n_context, -1)
            x = torch.cat([batch_X, z_rep], dim=-1)
            s_hat = self.model(x)
            energies = F.mse_loss(batch_Y, s_hat, reduction='none').view(B, n_context, -1).sum(dim=-1)
            return energies

        for i in range(self.n_iter):
            z_optim.zero_grad()
            energies = f(z)
            loss = energies.sum()
            loss.backward()

            z_optim.step()

        return z, None

    def evaluate(self, x, z):
        # takes in x = (s ; a) and latent z and outputs \hat{s'}
        x = torch.cat([x, z.detach()], dim=1)
        return self.model(x)