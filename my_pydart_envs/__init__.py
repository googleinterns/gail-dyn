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
