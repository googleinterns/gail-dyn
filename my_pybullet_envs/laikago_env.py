from .laikago import LaikagoBullet

from pybullet_utils import bullet_client
import pybullet
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math

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

                 max_tar_vel=2.5,
                 energy_weight=0.05,
                 jl_weight=0.5,
                 ab=0,
                 q_pen_weight=0.3

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

        if self.render:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self.np_random = None
        self.robot = LaikagoBullet(init_noise=self.init_noise,
                                   obs_noise=self.obs_noise,
                                   act_noise=self.act_noise,
                                   time_step=self._ts,
                                   np_random=self.np_random)
        self.seed(0)  # used once temporarily, will be overwritten outside though superclass api
        self.viewer = None
        self.timer = 0

        self.floor_id = None

        self.reset_counter = 200    # do a hard reset first
        obs = self.reset()    # and update init obs

        action_dim = len(self.robot.ctrl_dofs)
        self.act = [0.0] * len(self.robot.ctrl_dofs)
        self.action_space = gym.spaces.Box(low=np.array([-1.]*action_dim), high=np.array([+1.]*action_dim))
        obs_dim = len(obs)
        obs_dummy = np.array([1.12234567]*obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)

    def reset(self):

        if self.reset_counter < 200:
            self.reset_counter += 1
            self.robot.soft_reset(self._p)
        else:
            self.reset_counter = 0

            self._p.resetSimulation()
            self._p.setTimeStep(self._ts)
            self._p.setGravity(0, 0, -10)
            self._p.setPhysicsEngineParameter(numSolverIterations=100)
            # self._p.setPhysicsEngineParameter(restitutionVelocityThreshold=0.000001)

            self.floor_id = self._p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0.0], useFixedBase=1)
            # self._p.changeDynamics(self.floor_id, -1, lateralFriction=1.0)     # TODO
            # self._p.changeDynamics(self.floor_id, -1, restitution=.2)

            self.robot.reset(self._p)
            # # should be after reset!
            # for ind in range(self.robot.n_total_dofs):
            #     self._p.changeDynamics(self.robot.go_id, ind, lateralFriction=1.0)
            #     self._p.changeDynamics(self.robot.go_id, ind, restitution=.2)

        self._p.stepSimulation()

        # # self.robot.soft_reset(self._p)
        # q, dq = self.robot.get_q_dq(self.robot.ctrl_dofs)
        # print("dq after reset", dq)
        # input("press enter")

        self.timer = 0
        obs = self.get_extended_observation()

        return np.array(obs)

    def step(self, a):

        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        x_0 = root_pos[0]

        for _ in range(self.control_skip):
            # action is in not -1,1
            if a is not None:
                self.act = a
                self.robot.apply_action(a)
            self._p.stepSimulation()
            if self.render:
                time.sleep(self._ts * 1.5)
            self.timer += 1

        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        x_1 = root_pos[0]

        not_done = True
        reward = self.ab        # alive bonus
        tar = np.minimum(self.timer / 500, self.max_tar_vel)
        reward += np.minimum((x_1 - x_0) / (self.control_skip * self._ts), tar) * 4.0
        # print("v", (x_1 - x_0) / (self.control_skip * self._ts), "tar", tar)
        reward += -self.energy_weight * np.square(a).sum()
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
        reward += np.minimum(height-0.3, 0.2) * 12

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

        # print("------")
        not_done = (np.abs(dq) < 90).all() and in_support and (height > 0.3)

        return self.get_extended_observation(), reward, not not_done, {}

    def get_dist(self):
        return self.robot.get_link_com_xyz_orn(-1)[0][0]

    def get_extended_observation(self):
        obs = self.robot.get_robot_observation()

        for foot in self.robot.feet:
            cps = self._p.getContactPoints(self.robot.go_id, self.floor_id, foot, -1)
            if len(cps) > 0:
                obs.extend([1.0])
            else:
                obs.extend([-1.0])

        obs.extend([np.minimum(self.timer / 500, self.max_tar_vel)])   # TODO

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
        distance = 3
        yaw = 0
        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        self._p.resetDebugVisualizerCamera(distance, yaw, -20, [root_pos[0], 0, 0.4])
