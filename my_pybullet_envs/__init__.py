#  Copyright 2020 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import gym
from gym.envs.registration import registry, make, spec
from .hopper_env import HopperURDFEnv
from .hopper_env_MB import HopperURDFEnvMB
from .laikago_env import LaikagoBulletEnv
from .laikago_env_v2 import LaikagoBulletEnvV2
from .laikago_env_ori import LaikagoBulletEnvOri
from .hopper_env_conf_policy import HopperConFEnv
from .laikago_env_conf_policy import LaikagoConFEnv
from .laikago_env_actf_policy import LaikagoActFEnv
from .laikago_env_actf_policy_v2 import LaikagoActFEnvV2
# from .laikago_env_conf_reverse_env import LaikagoConFEnvRev
# from .laikago_env_conf_debug_simple import LaikagoConFEnvDebug


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

register(
    id="LaikagoBulletEnv-v2",
    entry_point="my_pybullet_envs:LaikagoBulletEnvV2",
    max_episode_steps=500,
)

# register(
#     id="LaikagoBulletEnv-v2",
#     entry_point="my_pybullet_envs:LaikagoBulletEnvOri",
#     max_episode_steps=500,
# )

register(
    id="LaikagoConFEnv-v1",
    entry_point="my_pybullet_envs:LaikagoConFEnv",
    max_episode_steps=500,
)

register(
    id="LaikagoActFEnv-v1",
    entry_point="my_pybullet_envs:LaikagoActFEnv",
    max_episode_steps=500,
)

register(
    id="LaikagoActFEnv-v2",
    entry_point="my_pybullet_envs:LaikagoActFEnvV2",
    max_episode_steps=500,
)

# register(
#     id="LaikagoConFEnv-v2",
#     entry_point="my_pybullet_envs:LaikagoConFEnvRev",
#     max_episode_steps=500,
# )
#
# register(
#     id="LaikagoConFEnv-v3",
#     entry_point="my_pybullet_envs:LaikagoConFEnvDebug",
#     max_episode_steps=500,
# )

def getList():
    btenvs = [
        "- " + spec.id
        for spec in gym.envs.registry.all()
        if spec.id.find("Bullet") >= 0
    ]
    return btenvs
