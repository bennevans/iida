
from varyingsim.policies.policy import Policy
from varyingsim.util.trajectory import get_traj_acts

import numpy as np

class MPPIEnvPolicy(Policy):
    # TODO: mppi pytorch policy

    def __init__(self, env_constr, K, H, sigma, set_fov, temp=1.0, term_fn=None):
        self.env = env_constr() # takes obs, ctrl outputs next_obs 
        self.K = K
        self.H = H
        self.sigma = sigma
        self.mu = np.zeros(self.sigma.shape[0])
        self.temp = temp
        self.set_fov = set_fov
        self.term_fn = term_fn

        self.control_sequence = np.zeros((self.H, self.env.model.nu))
        self.control_perturbed = np.zeros((self.K, self.H, self.env.model.nu))
        self.traj_rews = np.zeros(self.K)

    def get_action(self, obs, t, i=0):
        perturbations = np.random.multivariate_normal(self.mu, self.sigma, size=(self.K, self.H))
        self.control_perturbed[:] = self.control_sequence + perturbations

        # TODO: only works if actionspace bounds are specified
        self.control_perturbed[:] = np.clip(self.control_perturbed[:], self.env.action_space.low, self.env.action_space.high)

        # TODO: parallel rollouts
        
        obs_0 = obs
        for k in range(self.K):
            acts = self.control_perturbed[k]
            self.env.t = t
            ret = get_traj_acts(self.env, obs_0, i, acts, self.set_fov)
            obs, act, fov, rew = ret
            if self.term_fn is None:
                self.traj_rews[k] = np.sum(rew) # TODO: discounted sum
            else:
                self.traj_rews[k] = np.sum(rew) + self.term_fn(obs[-1])

        beta = np.min(self.traj_rews)

        exp_rews = np.exp((self.traj_rews - beta ) / self.temp)
        norm = np.sum(exp_rews)
        weights = exp_rews / norm
        # TODO: make numpy / efficient

        self.control_sequence[:] = np.zeros((self.H, self.env.model.nu))
        for k in range(self.K):
            self.control_sequence += weights[k] * self.control_perturbed[k]
        
        act = self.control_sequence[0]
        self.control_sequence = np.concatenate([self.control_sequence[1:], self.control_sequence[-1:]])
        self.traj_rews = np.zeros(self.K)

        return act

    def reset(self):
        self.control_sequence = np.zeros((H, env.model.nu))

if __name__ == '__main__':
    from varyingsim.envs.cartpole import CartpoleEnv

    def set_fov(env, i, t):
        # end_mass = 1 + np.sin(t / 10.0)
        # end_mass = 0.0
        # env.set_end_mass(end_mass)
        # env.set_pole_mass(0.01)
        pass
    
    def construct_fn():
        return CartpoleEnv(mode=CartpoleEnv.SWINGUP, set_param_fn=set_fov, T=250)
    
    env = construct_fn()
    K = 32
    H = 128
    sigma = np.eye(env.model.nu) * 1.0
    temp = 0.3

    policy = MPPIEnvPolicy(construct_fn, K, H, sigma, set_fov, temp=temp)

    obs = env.reset()
    for t in range(1000):
        act = policy.get_action(obs, t)
        obs, rew, done, info = env.step(act)
        env.render()
        if done:
            break
            obs = env.reset()

