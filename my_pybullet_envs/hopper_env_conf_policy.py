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

class HopperConFEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 render=True,
                 init_noise=True,
                 act_noise=True,
                 obs_noise=True,
                 control_skip=10,
                 using_torque_ctrl=True,
                 correct_obs_dx=True,        # if need to correct dx obs,

                 # soft_floor_env=False,
                 # low_torque_env=False,

                 behavior_dir="trained_models_hopper_bullet_3/ppo",
                 behavior_env_name="HopperURDFEnv-v1"
                 ):

        self.render = render
        self.init_noise = init_noise
        self.obs_noise = obs_noise
        self.act_noise = act_noise
        self.control_skip = int(control_skip)
        self._ts = 1. / 500.
        self.correct_obs_dx = correct_obs_dx

        # self.soft_floor_env = soft_floor_env
        # self.low_torque_env = low_torque_env

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

        # load fixed behavior policy
        self.hopper_actor_critic, _, \
            self.recurrent_hidden_states, \
            self.masks = utils.load(
                behavior_dir, behavior_env_name, False, None        # cpu load
            )

        # self.floor_id = None

        self.obs = []
        self.behavior_obs_len = None
        self.behavior_act_len = None
        self.reset()    # and update init obs

        self.action_dim = 4     # see beginning of step() for comment
        self.act = [0.0] * self.action_dim
        self.action_space = gym.spaces.Box(low=np.array([-1.]*self.action_dim), high=np.array([+1.]*self.action_dim))
        self.obs_dim = self.behavior_act_len+self.behavior_obs_len    # see beginning of update_obs() for comment
        assert self.behavior_act_len == len(self.robot.ctrl_dofs)
        obs_dummy = np.array([1.12234567]*self.obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf*obs_dummy, high=np.inf*obs_dummy)

        # self.gen_dyn = None
        # if self.use_gen_dyn:
        #     self.gen_dyn = Generator()
        #     self.gen_dyn.load_state_dict(torch.load(gen_dyn_path))
        #     self.gen_dyn.eval()

    def reset(self):
        self._p.resetSimulation()
        self._p.setTimeStep(self._ts)
        self._p.setGravity(0, 0, -10)
        self.timer = 0

        self._p.setPhysicsEngineParameter(numSolverIterations=100)
        # self._p.setPhysicsEngineParameter(restitutionVelocityThreshold=0.000001)

        # self.floor_id = self._p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0.0], useFixedBase=1)
        # self._p.changeDynamics(self.floor_id, -1, lateralFriction=1.0)     # TODO
        # self._p.changeDynamics(self.floor_id, -1, restitution=.2)

        self.robot.reset(self._p)
        self.robot.update_x(reset=True)
        # # should be after reset!

        # if self.soft_floor_env:
        #     self._p.changeDynamics(self.floor_id, -1, contactDamping=100.0, contactStiffness=600.0)
        #     for ind in range(self.robot.n_total_dofs):
        #         self._p.changeDynamics(self.robot.hopper_id, ind, contactDamping=100.0, contactStiffness=600.0)
        #
        # if self.low_torque_env:
        #     self.robot.max_forces[2] = 200/1.6      # 1.6 for policy 4, 2.0 for policy 3

        #     self._p.changeDynamics(self.robot.hopper_id, ind, lateralFriction=1.0)
        #     self._p.changeDynamics(self.robot.hopper_id, ind, restitution=.2)

        # self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, i)

        self._p.stepSimulation()

        self.update_extended_observation()

        return self.obs

    def step(self, con_f):
        # in hopper case, con_f is 4D, 2D contact force + 2D local force location
        # redundant (3D) but fine

        action = self.obs[self.behavior_obs_len:]
        action = np.clip(action, -1.0, 1.0)
        if self.act_noise:
            action = utils.perturb(action, 0.05, self.np_random)

        for _ in range(self.control_skip):
            self.robot.apply_action(action)
            self.apply_scale_clip_conf_from_pi(con_f)
            self._p.stepSimulation()
            if self.render:
                time.sleep(self._ts * 0.5)
            self.timer += 1
        self.robot.update_x()
        self.update_extended_observation()

        height = self._p.getLinkState(self.robot.hopper_id, 2, computeForwardKinematics=1)[0][2]
        not_done = (height > 0.0) and (height < 2.0)
        reward = 0      # use gail to set reward

        return self.obs, reward, not not_done, {}

    def apply_scale_clip_conf_from_pi(self, con_f):
        # each dim of input is roughly in [-1, 1]

        approx_mass = 18.0
        max_fz = approx_mass * 9.81 * 5    # 5mg
        # first dim represents fz
        fz = np.interp(con_f[0], [-1, 1], [-5, max_fz])
        # second dim represents fx
        fx = np.interp(con_f[1], [-1, 1], [-max_fz, max_fz])    # mu<=1.0
        # third dim represents f location in foot coordinate x
        f_loc_x = np.interp(con_f[2], [-1, 1], [-0.3, 0.3])     # foot length 0.5
        # fourth dim represents f location in foot coordinate z
        f_loc_z = np.interp(con_f[2], [-1, 1], [-0.1, 0.1])  # foot height 0.12

        # fx = fz = f_loc_x = f_loc_z = 0
        # fz = 15.0 * 9.81
        utils.apply_external_world_force_on_local_point(self.robot.hopper_id, 5,
                                                        [fx, 0, fz],
                                                        [f_loc_x, 0, f_loc_z],
                                                        self._p)

    def get_dist(self):
        return self.robot.x

    def get_ave_dx(self):
        if self.robot.last_x:
            return (self.robot.x - self.robot.last_x) / (self.control_skip * self._ts)
        else:
            return 0.0

    def update_extended_observation(self):
        # in out dyn policy setting, the obs is actually cat(st,at)

        self.obs = self.robot.get_robot_observation()

        if self.correct_obs_dx:
            dx = self.get_ave_dx() * self.robot.obs_scaling[5]
            self.obs[5] = dx

        if self.obs_noise:
            self.obs = utils.perturb(self.obs, 0.1, self.np_random)

        self.behavior_obs_len = len(self.obs)

        obs_nn = utils.wrap(self.obs, is_cuda=False)
        with torch.no_grad():
            _, action_nn, _, self.recurrent_hidden_states = self.hopper_actor_critic.act(
                obs_nn, self.recurrent_hidden_states, self.masks, deterministic=False     # TODO, det pi
            )
        action = utils.unwrap(action_nn, is_cuda=False)

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
        torso_x = self._p.getLinkState(self.robot.hopper_id, 2, computeForwardKinematics=1)[0]
        self._p.resetDebugVisualizerCamera(distance, yaw, -20, torso_x)
