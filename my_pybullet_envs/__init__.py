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
