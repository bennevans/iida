import numpy as np
import torch.nn.functional as F

class DummyModel():
    def __init__(self, nu, nq, nv):
        self.nu = nu
        self.nq = nq
        self.nv = nv

class DummyEnv():
    def __init__(self, nu, nq, nv, n_fov=1):
        self.model = DummyModel(nu, nq, nv)
        self.n_fov = n_fov
    
    def reset(self):
        return np.zeros(self.model.nq + self.model.nv)

    def loss(self, y_hat, y):
        base_loss = F.mse_loss(y_hat, y)
        return base_loss, {'base_loss': base_loss.item()}