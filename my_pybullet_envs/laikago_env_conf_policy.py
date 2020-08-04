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

from .laikago import LaikagoBullet

from pybullet_utils import bullet_client
import pybullet
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math
import torch
from gan import utils

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class LaikagoConFEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 render=True,
                 init_noise=True,
                 act_noise=True,
                 obs_noise=True,
                 control_skip=10,
                 using_torque_ctrl=True,

                 max_tar_vel=2.5,
                 energy_weight=0.05,
                 jl_weight=0.5,
                 ab=0,
                 q_pen_weight=0.3,

                 train_dyn=True,  # if false, fix dyn and train motor policy
                 behavior_dir="trained_models_laika_bullet_12/ppo",
                 behavior_env_name="LaikagoBulletEnv-v1",
                 dyn_dir="",
                 dyn_env_name="LaikagoConFEnv-v1",
                 dyn_iter=None,

                 cuda_env=True
                 ):

        self.render = render
        self.init_noise = init_noise
        self.obs_noise = obs_noise
        self.act_noise = act_noise
        self.control_skip = int(control_skip)
        self._ts = 1. / 500.

        self.max_tar_vel = max_tar_vel
        self.energy_weight = energy_weight
        self.jl_weight = jl_weight
        self.ab = ab
        self.q_pen_weight = q_pen_weight

        self.train_dyn = train_dyn
        self.cuda_env = cuda_env

        if self.render:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self.np_random = None
        self.robot = LaikagoBullet(init_noise=self.init_noise,
                                   time_step=self._ts,
                                   np_random=self.np_random)
        self.seed(0)  # used once temporarily, will be overwritten outside though superclass api
        self.viewer = None
        self.timer = 0

        if self.train_dyn:
            self.dyn_actor_critic = None
            # load fixed behavior policy
            self.go_actor_critic, _, \
            self.recurrent_hidden_states, \
            self.masks = utils.load(
                behavior_dir, behavior_env_name, self.cuda_env, None
            )
        else:
            if dyn_iter:
                dyn_iter = int(dyn_iter)
            # train motor policy
            self.go_actor_critic = None
            # load fixed dynamics model
            self.dyn_actor_critic, _, \
            self.recurrent_hidden_states, \
            self.masks = utils.load(
                dyn_dir, dyn_env_name, self.cuda_env, dyn_iter
            )

        self.reset_counter = 50  # do a hard reset first
        self.obs = []
        self.behavior_obs_len = None
        self.behavior_act_len = None
        self.reset()  # and update init obs

        if self.train_dyn:
            assert self.behavior_act_len == len(self.robot.ctrl_dofs)
            self.action_dim = 12  # see beginning of step() for comment
            self.obs_dim = self.behavior_act_len + self.behavior_obs_len  # see beginning of update_obs() for comment
        else:
            self.action_dim = len(self.robot.ctrl_dofs)
            self.obs_dim = len(self.obs)

        self.act = [0.0] * self.action_dim
        self.action_space = gym.spaces.Box(low=np.array([-1.] * self.action_dim),
                                           high=np.array([+1.] * self.action_dim))
        obs_dummy = np.array([1.12234567] * self.obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf * obs_dummy, high=np.inf * obs_dummy)

    def reset(self):

        if self.reset_counter < 50:
            self.reset_counter += 1
            self.robot.soft_reset(self._p)
        else:
            self.reset_counter = 0

            self._p.resetSimulation()
            self._p.setTimeStep(self._ts)
            self._p.setGravity(0, 0, -10)
            self._p.setPhysicsEngineParameter(numSolverIterations=100)
            # self._p.setPhysicsEngineParameter(restitutionVelocityThreshold=0.000001)

            # self.floor_id = self._p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0.0], useFixedBase=1)
            # self._p.changeDynamics(self.floor_id, -1, contactDamping=150.0, contactStiffness=600.0)
            # # self._p.changeDynamics(self.floor_id, -1, contactStiffness=3.0)

            self.robot.reset(self._p)
            # # should be after reset!
            # for ind in self.robot.feet:
            #     self._p.changeDynamics(self.robot.go_id, ind, contactDamping=150.0, contactStiffness=600.0)
            #     # self._p.changeDynamics(self.robot.go_id, ind, contactStiffness=3.0)

        self._p.stepSimulation()
        self.timer = 0
        self.update_extended_observation()

        return self.obs

    def step(self, a):

        if self.train_dyn:
            # TODO: currently for laika, env_action is 12D, 4 feet 3D without wrench
            env_action = a
            robo_action = self.obs[self.behavior_obs_len:]
        else:
            robo_action = a
            env_pi_obs = np.concatenate((self.obs, robo_action))

            env_pi_obs_nn = utils.wrap(env_pi_obs, is_cuda=self.cuda_env)
            with torch.no_grad():
                _, env_action_nn, _, self.recurrent_hidden_states = self.dyn_actor_critic.act(
                    env_pi_obs_nn, self.recurrent_hidden_states, self.masks, deterministic=False
                )
            env_action = utils.unwrap(env_action_nn, is_cuda=self.cuda_env)

        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        x_0 = root_pos[0]

        robo_action = np.clip(robo_action, -1.0, 1.0)
        if self.act_noise:
            robo_action = utils.perturb(robo_action, 0.05, self.np_random)

        for _ in range(self.control_skip):
            self.robot.apply_action(robo_action)
            self.apply_scale_clip_conf_from_pi(env_action)
            self._p.stepSimulation()
            if self.render:
                time.sleep(self._ts * 0.5)
            self.timer += 1
        self.update_extended_observation()

        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        x_1 = root_pos[0]

        reward = self.ab  # alive bonus
        tar = np.minimum(self.timer / 500, self.max_tar_vel)
        reward += np.minimum((x_1 - x_0) / (self.control_skip * self._ts), tar) * 4.0
        # print("v", (x_1 - x_0) / (self.control_skip * self._ts), "tar", tar)
        reward += -self.energy_weight * np.square(robo_action).sum()
        # print("act norm", -self.energy_weight * np.square(a).sum())

        q, dq = self.robot.get_q_dq(self.robot.ctrl_dofs)
        pos_mid = 0.5 * (self.robot.ll + self.robot.ul)
        q_scaled = 2 * (q - pos_mid) / (self.robot.ul - self.robot.ll)
        joints_at_limit = np.count_nonzero(np.abs(q_scaled) > 0.97)
        reward += -self.jl_weight * joints_at_limit
        # print("jl", -self.jl_weight * joints_at_limit)

        reward += -np.minimum(np.sum(np.abs(dq)) * 0.02, 5.0)  # almost like /23
        reward += -np.minimum(np.sum(np.square(q - self.robot.init_q)) * self.q_pen_weight, 5.0)  # almost like /23
        # print(np.abs(dq))
        # print("vel pen", -np.minimum(np.sum(np.abs(dq)) * 0.03, 5.0))
        # print("pos pen", -np.minimum(np.sum(np.abs(q - self.robot.init_q)) * self.q_pen_weight, 5.0))
        # print("pos pen", -np.minimum(np.sum(np.square(q - self.robot.init_q)) * self.q_pen_weight, 5.0))

        y_1 = root_pos[1]
        reward += -y_1 * 0.5
        # print("dev pen", -y_1*0.5)
        height = root_pos[2]
        reward += np.minimum(height - 0.3, 0.2) * 12
        # print("height", height)

        # in_support = self.robot.is_root_com_in_support()

        # print("______")
        # print(in_support)

        # print("h", height)
        # print("dq.", np.abs(dq))
        # print((np.abs(dq) < 50).all())

        # body = [0, 4, 8, 12, -1, 1, 5, 9, 13]
        # body_floor = False
        # for link in body:
        #     cps = self._p.getContactPoints(self.robot.go_id, self.floor_id, link, -1)
        #     if len(cps) > 0:
        #         print("body hit floor")
        #         body_floor = True
        #         break

        # print("------")
        rpy = self._p.getEulerFromQuaternion(self.obs[9:13])
        not_done = (abs(y_1) < 5.0) and (height > 0.1) and (height < 1.0) and (rpy[2] > 0.1)
        # not_done = True

        return self.obs, reward, not not_done, {}

    def apply_scale_clip_conf_from_pi(self, con_f):

        approx_mass = 26.0
        max_fz = approx_mass * 9.81 * 2  # 2mg        # TODO

        for foot_ind, link in enumerate(self.robot.feet):
            this_con_f = con_f[foot_ind*3: (foot_ind+1)*3]
            # first dim represents fz
            fz = np.interp(this_con_f[0], [-0.1, 5], [0.0, max_fz])
            # second dim represents fx
            fx = np.interp(this_con_f[1], [-2, 2], [-1.8*fz, 1.8*fz])
            # fx = np.sign(fx) * np.maximum(np.abs(fx) - 0.6 * max_fz, 0.0)   # mu<=1.2
            # third dim represents fy
            fy = np.interp(this_con_f[2], [-2, 2], [-1.8*fz, 1.8*fz])
            # fy = np.sign(fy) * np.maximum(np.abs(fy) - 0.6 * max_fz, 0.0)   # mu<=1.2

            utils.apply_external_world_force_on_local_point(self.robot.go_id, link,
                                                            [fx, fy, fz],
                                                            [0, 0, 0],
                                                            self._p)

    def get_dist(self):
        return self.robot.get_link_com_xyz_orn(-1)[0][0]

    def update_extended_observation(self):
        self.obs = self.robot.get_robot_observation()

        self.obs = np.concatenate((self.obs, [np.minimum(self.timer / 500, self.max_tar_vel)]))  # TODO

        if self.obs_noise:
            self.obs = utils.perturb(self.obs, 0.1, self.np_random)

        if self.train_dyn:
            self.behavior_obs_len = len(self.obs)

            obs_nn = utils.wrap(self.obs, is_cuda=self.cuda_env)
            with torch.no_grad():
                _, action_nn, _, self.recurrent_hidden_states = self.go_actor_critic.act(
                    obs_nn, self.recurrent_hidden_states, self.masks, deterministic=False  # TODO, det pi
                )
            action = utils.unwrap(action_nn, is_cuda=self.cuda_env)

            self.behavior_act_len = len(action)

            self.obs = np.concatenate((self.obs, action))

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s

    def cam_track_torso_link(self):
        distance = 5
        yaw = 0
        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        self._p.resetDebugVisualizerCamera(distance, yaw, -20, [root_pos[0], 0, 0.4])
