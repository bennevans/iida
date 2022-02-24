import torch
from torch.distributions import Categorical

def copy_weights(model_from, model_to, tau):
    """
        does exponential moving average update for model
    """
    for param, target_param in zip(model_from.parameters(), model_to.parameters()):
        target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)

def weight_init(m):
    """Custom weight init for Linear layers."""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def entropy(probs):
    return Categorical(probs=probs).entropy().mean()

def entropy_from_log(log_probs):
    return Categorical(logits=log_probs).entropy().mean()

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu