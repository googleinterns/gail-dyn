from .hopper import HopperURDF

from pybullet_utils import bullet_client
import pybullet
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class HopperURDFEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 render=True,
                 init_noise=True,
                 act_noise=True,
                 obs_noise=True,
                 control_skip=10,
                 using_torque_ctrl=False
                 ):

        self.render = render
        self.init_noise = init_noise
        self.obs_noise = obs_noise
        self.act_noise = act_noise
        self.control_skip = int(control_skip)
        self._ts = 1. / 500.

        if self.render:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self.np_random = None
        self.robot = HopperURDF(init_noise=self.init_noise,
                                obs_noise=self.obs_noise,
                                act_noise=self.act_noise,
                                time_step=self._ts,
                                np_random=self.np_random)
        self.seed(0)  # used once temporarily, will be overwritten outside though superclass api
        self.viewer = None
        self.timer = 0

        self.floor_id = None

        obs = self.reset()    # and update init obs

        action_dim = len(self.robot.ctrl_dofs)
        self.act = [0.0] * len(self.robot.ctrl_dofs)
        self.action_space = gym.spaces.Box(low=np.array([-1.]*action_dim), high=np.array([+1.]*action_dim))
        obs_dim = len(obs)
        obs_dummy = np.array([1.12234567]*obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)

    def reset(self):
        self._p.resetSimulation()
        self._p.setTimeStep(self._ts)
        self._p.setGravity(0, 0, -10)
        self.timer = 0

        self._p.setPhysicsEngineParameter(numSolverIterations=100)
        # self._p.setPhysicsEngineParameter(restitutionVelocityThreshold=0.000001)

        self.floor_id = self._p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0.0], useFixedBase=1)
        self._p.changeDynamics(self.floor_id, -1, lateralFriction=1.0)     # TODO
        self._p.changeDynamics(self.floor_id, -1, restitution=.2)

        self.robot.reset(self._p)
        # # should be after reset!
        # for ind in range(self.robot.n_total_dofs):
        #     self._p.changeDynamics(self.robot.hopper_id, ind, lateralFriction=1.0)
        #     self._p.changeDynamics(self.robot.hopper_id, ind, restitution=.2)

        # self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, i)

        self._p.stepSimulation()

        obs = self.get_extended_observation()

        return np.array(obs)

    def step(self, a):

        x_0 = self._p.getJointState(self.robot.hopper_id, 0)[0]

        for _ in range(self.control_skip):
            # action is in not -1,1
            if a is not None:
                self.act = np.clip(a, -1.0, 1.0)
                self.robot.apply_action(self.act)
            self._p.stepSimulation()
            if self.render:
                time.sleep(self._ts * 0.5)
            self.timer += 1

        x_1 = self._p.getJointState(self.robot.hopper_id, 0)[0]

        reward = 2.0        # alive bonus
        reward += (x_1 - x_0) / (self.control_skip * self._ts)
        # print("v", (x_1 - x_0) / (self.control_skip * self._ts))
        reward += -0.1 * np.square(a).sum()
        # print("act norm", -0.1 * np.square(a).sum())

        q, _ = self.robot.get_q_dq(self.robot.ctrl_dofs)
        pos_mid = 0.5 * (self.robot.ll + self.robot.ul)
        q_scaled = 2 * (q - pos_mid) / (self.robot.ul - self.robot.ll)
        # print(q)
        # print(q_scaled)
        joints_at_limit = np.count_nonzero(np.abs(q_scaled) > 0.97)
        reward += -2.0 * joints_at_limit
        # print("jl", -1.0 * joints_at_limit)

        joints_state = self._p.getJointStates(self.robot.hopper_id, self.robot.ctrl_dofs)
        joints_dq = np.array(joints_state)[:, [1]]
        joints_dq = np.hstack(joints_dq.flatten())
        reward -= np.minimum(np.sum(np.abs(joints_dq)) * 0.02, 5.0)  # almost like /23
        # print("vel pen", np.minimum(np.sum(np.abs(joints_dq)) * 0.02, 5.0))

        height = self._p.getLinkState(self.robot.hopper_id, 2, computeForwardKinematics=1)[0][2]
        # ang = self._p.getJointState(self.robot.hopper_id, 2)[0]

        # print(joints_dq)
        # print(height)
        # print("ang", ang)
        not_done = (np.abs(joints_dq) < 50).all() and (height > .7) and (height < 1.8)

        return self.get_extended_observation(), reward, not not_done, {}

    def get_dist(self):
        return self._p.getJointState(self.robot.hopper_id, 0)[0]

    def get_extended_observation(self):
        return self.robot.get_robot_observation()

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
        torso_x = self._p.getLinkState(self.robot.hopper_id, 2, computeForwardKinematics=1)[0]
        self._p.resetDebugVisualizerCamera(distance, yaw, -20, torso_x)
