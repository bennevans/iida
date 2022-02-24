import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
from varyingsim.models.cl_model import ContinualLearningModel

class FeedForward(nn.Module):
    def __init__(self, d_in, d_out, hidden_sizes, activation=nn.ReLU):
        super(FeedForward, self).__init__()
        if len(hidden_sizes) == 0:
            self.model = nn.Linear(d_in, d_out)
        else:
            modules = [nn.Linear(d_in, hidden_sizes[0])]
            for i in range(len(hidden_sizes) - 1):
                modules.append(activation())
                modules.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            modules.append(activation())    
            modules.append(nn.Linear(hidden_sizes[-1], d_out))

            self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

class FeedForwardDatum(ContinualLearningModel):
    def __init__(self, env, d_in, d_out, hidden_sizes, activation=nn.ReLU, device='cpu', include_fov=False):
        super(FeedForwardDatum, self).__init__(env, 1, device=device)
        self.model = FeedForward(d_in, d_out, hidden_sizes, activation=activation)
        self.include_fov=include_fov
    
    def forward(self, datum):
        obs = datum['obs']
        act = datum['act']
        fov = datum['fov']

        if self.include_fov:
            x = torch.cat([obs, act, fov], dim=-1).to(self.device)
        else:
            x = torch.cat([obs, act], dim=-1).to(self.device)
        
        return self.model(x)
