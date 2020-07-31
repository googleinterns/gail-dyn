import gym
from gym.envs.registration import registry, make, spec
from .hopper_env import HopperURDFEnv
from .hopper_env_MB import HopperURDFEnvMB
from .laikago_env import LaikagoBulletEnv
from .hopper_env_conf_policy import HopperConFEnv


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


# ------------bullet-------------

register(
    id="HopperURDFEnv-v1",
    entry_point="my_pybullet_envs:HopperURDFEnv",
    max_episode_steps=500,
)

register(
    id="HopperURDFEnv-v2",
    entry_point="my_pybullet_envs:HopperURDFEnvMB",
    max_episode_steps=500,
)

register(
    id="HopperConFEnv-v1",
    entry_point="my_pybullet_envs:HopperConFEnv",
    max_episode_steps=500,
)

register(
    id="LaikagoBulletEnv-v1",
    entry_point="my_pybullet_envs:LaikagoBulletEnv",
    max_episode_steps=500,
)

def getList():
    btenvs = [
        "- " + spec.id
        for spec in gym.envs.registry.all()
        if spec.id.find("Bullet") >= 0
    ]
    return btenvs
