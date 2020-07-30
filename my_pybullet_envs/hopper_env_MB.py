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

import torch
from gan.wgan_models import Generator
from gan import utils

class HopperURDFEnvMB(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 render=True,
                 init_noise=True,
                 act_noise=True,
                 obs_noise=True,
                 control_skip=10,
                 using_torque_ctrl=True,
                 correct_obs_dx=True,        # if need to correct dx obs,
                 use_gen_dyn=False,
                 gen_dyn_path=None,
                 soft_floor_env=False,
                 low_torque_env=False
                 ):

        self.render = render
        self.init_noise = init_noise
        self.obs_noise = obs_noise
        self.act_noise = act_noise
        self.control_skip = int(control_skip)
        self._ts = 1. / 500.
        self.correct_obs_dx = correct_obs_dx
        self.use_gen_dyn = use_gen_dyn

        self.soft_floor_env = soft_floor_env
        self.low_torque_env = low_torque_env

        if self.render:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self.np_random = None
        self.robot = HopperURDF(init_noise=self.init_noise,
                                time_step=self._ts,
                                np_random=self.np_random)
        self.seed(0)  # used once temporarily, will be overwritten outside though superclass api
        self.viewer = None
        self.timer = 0

        self.floor_id = None

        self.obs = []
        self.reset()    # and update init obs

        self.action_dim = len(self.robot.ctrl_dofs)
        self.act = [0.0] * len(self.robot.ctrl_dofs)
        self.action_space = gym.spaces.Box(low=np.array([-1.]*self.action_dim), high=np.array([+1.]*self.action_dim))
        self.obs_dim = len(self.obs)
        obs_dummy = np.array([1.12234567]*self.obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)

        self.gen_dyn = None
        if self.use_gen_dyn:
            self.gen_dyn = Generator()
            self.gen_dyn.load_state_dict(torch.load(gen_dyn_path))
            self.gen_dyn.eval()

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
        self.robot.update_x(reset=True)
        # # should be after reset!

        if self.soft_floor_env:
            self._p.changeDynamics(self.floor_id, -1, contactDamping=100.0, contactStiffness=600.0)
            for ind in range(self.robot.n_total_dofs):
                self._p.changeDynamics(self.robot.hopper_id, ind, contactDamping=100.0, contactStiffness=600.0)

        if self.low_torque_env:
            self.robot.max_forces[2] = 200/1.6      # 1.6 for policy 4, 2.0 for policy 3

        #     self._p.changeDynamics(self.robot.hopper_id, ind, lateralFriction=1.0)
        #     self._p.changeDynamics(self.robot.hopper_id, ind, restitution=.2)

        # self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, i)

        self._p.stepSimulation()

        self.update_extended_observation()

        return self.obs

    def step(self, a):
        if self.act_noise and a is not None:
            a = utils.perturb(a, 0.05, self.np_random)

        if not self.use_gen_dyn:
            for _ in range(self.control_skip):
                # action is in not -1,1
                if a is not None:
                    self.act = np.clip(a, -1.0, 1.0)
                    self.robot.apply_action(self.act)
                self._p.stepSimulation()
                if self.render:
                    time.sleep(self._ts * 0.5)
                self.timer += 1
            self.robot.update_x()
            self.update_extended_observation()
            obs_unnorm = np.array(self.obs) / self.robot.obs_scaling
        else:

            # gen_input = list(self.obs) + list(a) + [0.0] * (11+3)

            gen_input = list(self.obs) + list(a) + list(np.random.normal(0, 1, 14))       # TODO
            gen_input = utils.wrap(gen_input, is_cuda=False)      # TODO
            gen_output = self.gen_dyn(gen_input)
            gen_output = utils.unwrap(gen_output, is_cuda=False)

            self.obs = gen_output
            obs_unnorm = np.array(self.obs) / self.robot.obs_scaling
            self.robot.last_x = self.robot.x
            self.robot.x += obs_unnorm[5] * (self.control_skip * self._ts)

            if self.render:
                all_qs = [self.robot.x] + list(obs_unnorm[:5])
                all_qs[1] -= 1.5
                for ind in range(self.robot.n_total_dofs):
                    self._p.resetJointState(self.robot.hopper_id, ind, all_qs[ind], 0.0)
                time.sleep(self._ts * 5.0)

            if self.obs_noise:
                self.obs = utils.perturb(self.obs, 0.1, self.np_random)
            self.timer += self.control_skip

        reward = 2.0        # alive bonus
        reward += self.get_ave_dx()
        # print("v", self.get_ave_dx())
        reward += -0.1 * np.square(a).sum()
        # print("act norm", -0.1 * np.square(a).sum())

        q = np.array(obs_unnorm[2:5])
        pos_mid = 0.5 * (self.robot.ll + self.robot.ul)
        q_scaled = 2 * (q - pos_mid) / (self.robot.ul - self.robot.ll)
        joints_at_limit = np.count_nonzero(np.abs(q_scaled) > 0.97)
        reward += -2.0 * joints_at_limit
        # print("jl", -2.0 * joints_at_limit)

        dq = np.array(obs_unnorm[8:11])
        reward -= np.minimum(np.sum(np.abs(dq)) * 0.02, 5.0)  # almost like /23
        # print("vel pen", np.minimum(np.sum(np.abs(dq)) * 0.02, 5.0))

        height = obs_unnorm[0]
        # ang = self._p.getJointState(self.robot.hopper_id, 2)[0]

        # print(joints_dq)
        # print(height)
        # print("ang", ang)
        not_done = (np.abs(dq) < 50).all() and (height > .3) and (height < 1.8)

        return self.obs, reward, not not_done, {}

    def get_dist(self):
        return self.robot.x

    def get_ave_dx(self):
        if self.robot.last_x:
            return (self.robot.x - self.robot.last_x) / (self.control_skip * self._ts)
        else:
            return 0.0

    def update_extended_observation(self):
        self.obs = self.robot.get_robot_observation()

        if self.correct_obs_dx:
            dx = self.get_ave_dx() * self.robot.obs_scaling[5]
            self.obs[5] = dx

        if self.obs_noise:
            self.obs = utils.perturb(self.obs, 0.1, self.np_random)

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
