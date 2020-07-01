import gym
from gym.envs.registration import registry, make, spec
from .hopper import DartHopperEnv

def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


# ------------dart-------------

# register(
#     id="HumanoidSwimmerEnv-v1",
#     entry_point="my_pybullet_envs:HumanoidSwimmerEnv",
#     max_episode_steps=240,
# )

register(
    id='DartHopper-v1',
    entry_point='my_pydart_envs:DartHopperEnv',
    max_episode_steps=500,
)
