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
from gan import utils

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class LaikagoBulletEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 render=True,
                 init_noise=True,
                 act_noise=True,
                 obs_noise=True,
                 control_skip=10,
                 using_torque_ctrl=True,

                 no_dq=False,
                 max_tar_vel=2.5,
                 energy_weight=0.1,
                 jl_weight=0.5,
                 ab=5.0,
                 q_pen_weight=0.5,
                 dq_pen_weight=0.02,
                 vel_r_weight=4.0,

                 soft_floor_env=False,
                 low_power_env=False,
                 randomization_train=False,
                 randomforce_train=False
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
        self.dq_pen_weight = dq_pen_weight
        self.vel_r_weight = vel_r_weight

        self.soft_floor_env = soft_floor_env
        self.low_power_env = low_power_env
        self.randomization_train = randomization_train
        self.randomforce_train = randomforce_train

        if self.render:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self.np_random = None
        self.robot = LaikagoBullet(init_noise=self.init_noise,
                                   time_step=self._ts,
                                   np_random=self.np_random,
                                   no_dq=no_dq)
        self.seed(0)  # used once temporarily, will be overwritten outside though superclass api
        self.viewer = None
        self.timer = 0

        self.floor_id = None

        self.reset_counter = 50  # do a hard reset first
        self.init_state = None
        obs = self.reset()  # and update init obs

        self.action_dim = len(self.robot.ctrl_dofs)
        self.act = [0.0] * len(self.robot.ctrl_dofs)
        self.action_space = gym.spaces.Box(low=np.array([-1.] * self.action_dim),
                                           high=np.array([+1.] * self.action_dim))
        self.obs_dim = len(obs)
        obs_dummy = np.array([1.12234567] * self.obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf * obs_dummy, high=np.inf * obs_dummy)

    def reset(self):

        if self.reset_counter < 50:
            self.reset_counter += 1

            self._p.restoreState(self.init_state)
            self.robot.soft_reset(self._p)
        else:
            self.reset_counter = 0

            self._p.resetSimulation()
            self._p.setTimeStep(self._ts)
            self._p.setGravity(0, 0, -10)
            self._p.setPhysicsEngineParameter(numSolverIterations=100)
            # self._p.setPhysicsEngineParameter(restitutionVelocityThreshold=0.000001)

            self.floor_id = self._p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0.0], useFixedBase=1)

            self.robot.reset(self._p)
            self.init_state = self._p.saveState()

            # self._p.changeDynamics(self.floor_id, -1, contactDamping=150.0, contactStiffness=400.0)
            # for ind in self.robot.feet:
            #     self._p.changeDynamics(self.robot.go_id, ind, contactDamping=150.0, contactStiffness=400.0)

        # should be after reset!
        if self.soft_floor_env:
            # TODO: for pi23
            damp = np.random.uniform(50.0, 75.0)
            stiff = np.random.uniform(75.0, 150.0)
            # # TODO: for pi12, 36
            # damp = 150.0
            # stiff = 400.0
            self._p.changeDynamics(self.floor_id, -1, contactDamping=damp, contactStiffness=stiff)
            # for ind in range(16):
            for ind in self.robot.feet:
                self._p.changeDynamics(self.robot.go_id, ind, contactDamping=damp, contactStiffness=stiff)

        # if self.low_power_env:
        #     # TODO: for pi23
        #     self.robot.max_forces = [30.0] * 3 + [15.0] * 3 + [30.0] * 6

        if self.randomization_train:
            damp = np.random.uniform(150.0, 1000.0)
            stiff = np.random.uniform(300.0, 3000.0)
            # damp = np.random.uniform(75.0, 150.0)
            # stiff = np.random.uniform(150.0, 300.0)
            self._p.changeDynamics(self.floor_id, -1, contactDamping=damp, contactStiffness=stiff)
            # for ind in range(16):
            for ind in self.robot.feet:
                self._p.changeDynamics(self.robot.go_id, ind, contactDamping=damp, contactStiffness=stiff)

        self._p.stepSimulation()

        self.timer = 0
        obs = self.get_extended_observation()

        return np.array(obs)

    def step(self, a):

        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        x_0 = root_pos[0]

        a = np.clip(a, -1.0, 1.0)
        if self.act_noise:
            a = utils.perturb(a, 0.05, self.np_random)

        if self.low_power_env:
            # TODO: for pi23
            _, dq = self.robot.get_q_dq(self.robot.ctrl_dofs)
            max_force_ratio = np.clip(2 - dq/3.0, 0, 1)
            # print(max_force_ratio)
            # self.robot.max_forces[3:6] = [30., 30, 30] * max_force_ratio[3:6]
            # self.robot.max_forces[6:9] = [30., 30, 30] * max_force_ratio[6:9]
            # self.robot.max_forces = ([30., 30, 30]*4) * max_force_ratio
            a *= max_force_ratio
            # a[3:6] *= 0.5

        for _ in range(self.control_skip):
            # action is in not -1,1
            if a is not None:
                self.act = a
                self.robot.apply_action(a)

            if self.randomforce_train:
                for foot_ind, link in enumerate(self.robot.feet):
                    # first dim represents fz
                    fz = np.random.uniform(-80, 80)
                    # second dim represents fx
                    fx = np.random.uniform(-80, 80)
                    # third dim represents fy
                    fy = np.random.uniform(-80, 80)

                    utils.apply_external_world_force_on_local_point(self.robot.go_id, link,
                                                                    [fx, fy, fz],
                                                                    [0, 0, 0],
                                                                    self._p)

            self._p.stepSimulation()
            if self.render:
                time.sleep(self._ts * 0.5)
            self.timer += 1

        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        x_1 = root_pos[0]
        self.velx = (x_1 - x_0) / (self.control_skip * self._ts)

        # reward = 0.0
        # tar = np.minimum(self.timer / 500, self.max_tar_vel)
        # reward += np.minimum(self.velx, tar) * self.vel_r_weight
        # # print("v", self.velx, "tar", tar)
        #
        # reward += np.maximum((self.max_tar_vel - tar) * self.vel_r_weight - 3.0, 0.0)     # alive bonus
        #
        # reward += -self.energy_weight * np.linalg.norm(a)
        # # print("act norm", -self.energy_weight * np.square(a).sum())
        #
        q, dq = self.robot.get_q_dq(self.robot.ctrl_dofs)
        # # print(np.max(np.abs(dq)))
        # pos_mid = 0.5 * (self.robot.ll + self.robot.ul)
        # q_scaled = 2 * (q - pos_mid) / (self.robot.ul - self.robot.ll)
        # joints_at_limit = np.count_nonzero(np.abs(q_scaled) > 0.97)
        # reward += -self.jl_weight * joints_at_limit
        # # print("jl", -self.jl_weight * joints_at_limit)
        #
        # reward += -np.minimum(np.linalg.norm(dq) * self.dq_pen_weight, 5.0)
        # reward += -np.minimum(np.linalg.norm(q - self.robot.init_q) * self.q_pen_weight, 5.0)
        # # print("vel pen", -np.minimum(np.sum(np.abs(dq)) * self.dq_pen_weight, 5.0))
        # # print("pos pen", -np.minimum(np.sum(np.square(q - self.robot.init_q)) * self.q_pen_weight, 5.0))

        reward = self.ab  # alive bonus
        tar = np.minimum(self.timer / 500, self.max_tar_vel)
        reward += np.minimum(self.velx, tar) * self.vel_r_weight
        # print("v", self.velx, "tar", tar)
        reward += -self.energy_weight * np.square(a).sum()
        # print("act norm", -self.energy_weight * np.square(a).sum())

        pos_mid = 0.5 * (self.robot.ll + self.robot.ul)
        q_scaled = 2 * (q - pos_mid) / (self.robot.ul - self.robot.ll)
        joints_at_limit = np.count_nonzero(np.abs(q_scaled) > 0.97)
        reward += -self.jl_weight * joints_at_limit
        # print("jl", -self.jl_weight * joints_at_limit)

        reward += -np.minimum(np.sum(np.abs(dq)) * self.dq_pen_weight, 5.0)
        reward += -np.minimum(np.sum(np.square(q - self.robot.init_q)) * self.q_pen_weight, 5.0)

        y_1 = root_pos[1]
        reward += -y_1 * 0.5
        # print("dev pen", -y_1*0.5)
        height = root_pos[2]

        in_support = self.robot.is_root_com_in_support()

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

        cps = self._p.getContactPoints(bodyA=self.robot.go_id, linkIndexA=0)
        body_in_contact = (len(cps) > 0)

        # print("------")
        obs = self.get_extended_observation()
        rpy = self._p.getEulerFromQuaternion(obs[8:12])

        # for training
        not_done = (np.abs(dq) < 90).all() and (height > 0.3) and (height < 1.0) and in_support and not body_in_contact
        # for data collection
        not_done = (np.abs(dq) < 90).all() and (height > 0.3) and (height < 1.0) and not body_in_contact
        # not_done = True

        return obs, reward, not not_done, {}

    def get_dist(self):
        return self.robot.get_link_com_xyz_orn(-1)[0][0]

    def get_ave_dx(self):
        return self.velx

    def get_extended_observation(self):
        obs = self.robot.get_robot_observation()

        # for foot in self.robot.feet:
        #     cps = self._p.getContactPoints(self.robot.go_id, self.floor_id, foot, -1)
        #     if len(cps) > 0:
        #         obs.extend([1.0])
        #     else:
        #         obs.extend([-1.0])

        # obs.extend([np.minimum(self.timer / 500, self.max_tar_vel)])  # TODO

        if self.obs_noise:
            obs = utils.perturb(obs, 0.1, self.np_random)

        # print(obs)
        return obs

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s

    def cam_track_torso_link(self):
        distance = 2
        yaw = 0
        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        distance -= root_pos[1]
        self._p.resetDebugVisualizerCamera(distance, yaw, -20, [root_pos[0], 0.0, 0.4])
