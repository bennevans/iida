from gym.envs import register
import yaml
import os

register(id='VaryingCartPole-v0', 
    entry_point='varyingsim.envs.cartpole:CartpoleEnv', 
    max_episode_steps=500,
    kwargs={'mode': 'SWINGUP', 'T': 500})

register(id='VaryingHopper-v0', 
    entry_point='varyingsim.envs.hopper:HopperEnv', 
    max_episode_steps=1000)

register(id='VaryingSwimmer-v0', 
    entry_point='varyingsim.envs.swimmer:SwimmerEnv', 
    max_episode_steps=1000)

register(id='VaryingHumanoid-v0', 
    entry_point='varyingsim.envs.humanoid:HumanoidEnv', 
    max_episode_steps=1000)

register(id='AntWind-v0', 
    entry_point='varyingsim.envs.ant:AntEnv', 
    max_episode_steps=1000)

# streaming envs
config_path = os.path.join(os.path.split(__file__)[0], 'fov_configs')
with open(os.path.join(config_path, 'hopper.yaml'), 'r') as f:
    hopper_config = yaml.load(f, Loader=yaml.SafeLoader)

register(id='VaryingStreamingHopper-v0', 
    entry_point='varyingsim.envs.hopper:HopperStreamingEnv',
    max_episode_steps=1000, kwargs=dict(fov_config=hopper_config))