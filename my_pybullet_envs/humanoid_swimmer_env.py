from .humanoid_swimmer import HumanoidSwimmer

from pybullet_utils import bullet_client
import pybullet
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math

from . import rigidBodySento as rb

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class HumanoidSwimmerEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 render=True,
                 init_noise=True,
                 control_skip=10,
                 using_torque_ctrl=False
                 ):

        self.render = render
        self.init_noise = init_noise
        self.control_skip = int(control_skip)
        self._ts = 1. / 480.

        if self.render:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self.np_random = None
        self.robot = HumanoidSwimmer(init_noise=self.init_noise, time_step=self._ts, np_random=self.np_random,
                                     using_torque_ctrl=using_torque_ctrl)
        self.seed(0)  # used once temporarily, will be overwritten outside though superclass api
        self.viewer = None
        self.timer = 0

        self.floor_id = None

        obs = self.reset()    # and update init obs

        action_dim = self.robot.n_dofs
        self.act = [0.0] * self.robot.n_dofs
        self.action_space = gym.spaces.Box(low=np.array([-1.]*action_dim), high=np.array([+1.]*action_dim))
        obs_dim = len(obs)
        obs_dummy = np.array([1.12234567]*obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)

    def reset(self):
        self._p.resetSimulation()
        self._p.setTimeStep(self._ts)
        self._p.setGravity(0, 0, -rb.GRAVITY)
        self.timer = 0

        self.floor_id = self._p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0.0], useFixedBase=1)
        self._p.changeDynamics(self.floor_id, -1, lateralFriction=1.0, restitution=0.5)

        _ = rb.create_primitive_shape(-1, self._p.GEOM_BOX, [4, 4, rb.WATER_SURFACE/2.0], [0, 0, 0.9, 0.3],
                                      False, [0, 0, rb.WATER_SURFACE / 2.0])
        self.robot.reset(self._p)

        # self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, i)

        self._p.stepSimulation()

        obs = self.get_extended_observation()

        return np.array(obs)

    def step(self, a):
        # TODO: student code starts here
        # do two things here: apply action "a" and simulate with fluid force for control_skip times
        # then calculate reward based on post-sim state for return

        reward = 0.0

        # end student code
        return self.get_extended_observation(), reward, False, {}

    def get_extended_observation(self):
        # TODO: student code starts here
        return [0.0]        # dummy, change this
        # end student code

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s
