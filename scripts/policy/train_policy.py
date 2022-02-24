from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.policies.gaussian_linear import LinearPolicy
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
import mjrl.envs
import time as timer

from varyingsim.envs.cartpole import CartpoleEnv
import gym

SEED = 500

# T = 500
# job_name = 'training_cartpole'
# ENV_NAME = 'VaryingCartPole-v0'
# gym.envs.register(id=ENV_NAME, 
#     entry_point='varyingsim.envs.cartpole:CartpoleEnv', 
#     max_episode_steps=T,
#     kwargs={'mode': 'SWINGUP', 'T': T})

T = 1000
job_name = 'training_swimmer'
ENV_NAME = 'VaryingSwimmer-v0'
gym.envs.register(id=ENV_NAME, 
    entry_point='varyingsim.envs.swimmer:SwimmerEnv', 
    max_episode_steps=T)
step_size = 0.1
gamma = 0.995
num_traj = 10
num_iter = 50
init_log_std = -0.5
e = GymEnv(ENV_NAME)
policy = LinearPolicy(e.spec, init_log_std=init_log_std)

# T = 1000
# job_name = 'training_hopper'
# ENV_NAME = 'VaryingHopper-v0'
# gym.envs.register(id=ENV_NAME, 
#     entry_point='varyingsim.envs.hopper:HopperEnv', 
#     max_episode_steps=T)

e = GymEnv(ENV_NAME)
# policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED, init_log_std=init_log_std)
# baseline = QuadraticBaseline(e.spec)
baseline = MLPBaseline(e.spec)
agent = NPG(e, policy, baseline, normalized_step_size=step_size, seed=SEED, save_logs=True)

ts = timer.time()
train_agent(job_name=job_name,
            agent=agent,
            seed=SEED,
            niter=num_iter,
            gamma=gamma,
            gae_lambda=0.97,
            num_cpu=30,
            sample_mode='trajectories',
            num_traj=num_traj,
            save_freq=10,
            evaluation_rollouts=5)
print("time taken = %f" % (timer.time()-ts))
e.visualize_policy(policy, num_episodes=5, horizon=e.horizon, mode='evaluation')
