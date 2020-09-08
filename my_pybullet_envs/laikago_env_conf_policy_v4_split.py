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

from .laikago_v2 import LaikagoBulletV2

from pybullet_utils import bullet_client
import pybullet
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math
import torch
from gan import utils
from collections import deque

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


# this is called V4 because it shares same obs space with Laikago env V4
# the robot is still Laikago V2 though, same as env V4
class LaikagoConFEnvV4(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 render=True,
                 init_noise=True,
                 act_noise=False,
                 obs_noise=False,
                 control_skip=10,

                 max_tar_vel=2.5,
                 energy_weight=0.1,
                 jl_weight=0.5,
                 ab=5.0,
                 q_pen_weight=0.4,
                 dq_pen_weight=0.001,
                 vel_r_weight=4.0,

                 train_dyn=True,  # if false, fix dyn and train motor policy
                 pretrain_dyn=False,        # pre-train with deviation to sim
                 enlarge_act_range=0.0,    # make behavior pi more diverse to match collection, only train_dyn
                 behavior_dir="trained_models_laika_bullet_61/ppo",
                 behavior_env_name="LaikagoBulletEnv-v4",
                 behavior_iter=None,
                 dyn_dir="",
                 dyn_env_name="LaikagoConFEnv-v4",
                 dyn_iter=None,

                 cuda_env=True,
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

        self.train_dyn = train_dyn
        self.enlarge_act_range = enlarge_act_range
        self.pretrain_dyn = pretrain_dyn
        self.cuda_env = cuda_env

        self.ratio = None

        if self.render:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self.np_random = None
        self.robot = LaikagoBulletV2(init_noise=self.init_noise,
                                     time_step=self._ts,
                                     np_random=self.np_random)
        self.seed(0)  # used once temporarily, will be overwritten outside though superclass api
        self.viewer = None
        self.timer = 0

        # self.behavior_past_obs_t_idx = [0, 4, 8]
        self.behavior_past_obs_t_idx = [0]  # t-3. t-6. t-9  (B does not take past a)
        # self.generator_past_obs_t_idx = [0, 2]
        # self.generator_past_act_t_idx = [0]

        self.past_obs_array = deque(maxlen=10)
        self.past_bact_array = deque(maxlen=10)     # only need to store past behavior action

        if self.train_dyn:
            if behavior_iter:
                behavior_iter = int(behavior_iter)
            self.dyn_actor_critic = None
            # load fixed behavior policy
            self.go_actor_critic, _, \
                self.recurrent_hidden_states, \
                self.masks = utils.load(
                    behavior_dir, behavior_env_name, self.cuda_env, behavior_iter
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
            #
            # self.discri = utils.load_gail_discriminator(dyn_dir,
            #                                             dyn_env_name,
            #                                             self.cuda_env,
            #                                             dyn_iter)
            #
            # self.feat_select_func = self.robot.feature_selection_all_laika

        self.reset_const = 100
        self.reset_counter = self.reset_const  # do a hard reset first

        # self.action_dim = 12

        self.init_state = None
        obs = self.reset()
        #
        # self.d_scores = []

        # # set up imaginary session for pre-train
        # self.set_up_imaginary_session()

        if self.train_dyn:
            self.action_dim = 12  # 12D action scales, see beginning of step() for comment
        else:
            self.action_dim = len(self.robot.ctrl_dofs)

        self.act = [0.0] * self.action_dim
        self.action_space = gym.spaces.Box(low=np.array([-1.] * self.action_dim),
                                           high=np.array([+1.] * self.action_dim))
        obs_dummy = np.array([1.12234567] * len(obs))
        self.observation_space = gym.spaces.Box(low=-np.inf * obs_dummy, high=np.inf * obs_dummy)

    def reset(self):

        if self.reset_counter < self.reset_const:
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
            # conf policy does not use bullet collision
            self._p.setCollisionFilterGroupMask(self.floor_id, -1, 0, 0)

            self.robot.reset(self._p)
            self.init_state = self._p.saveState()

        self._p.stepSimulation()

        self.timer = 0
        self.past_obs_array.clear()
        self.past_bact_array.clear()
        # self.d_scores = []
        obs = self.get_extended_observation()

        # self.ratios = np.array([[]]).reshape(0, self.action_dim)

        return obs

    # def set_up_imaginary_session(self):
    #     # create another bullet session to run reset & rollout
    #     self._imaginary_p = bullet_client.BulletClient()
    #     self._imaginary_robot = LaikagoBulletV2(init_noise=self.init_noise,
    #                                             time_step=self._ts,
    #                                             np_random=self.np_random)
    #
    #     self._imaginary_p.resetSimulation()
    #     self._imaginary_p.setTimeStep(self._ts)
    #     self._imaginary_p.setGravity(0, 0, -10)
    #     self._imaginary_p.setPhysicsEngineParameter(numSolverIterations=100)
    #     # there is a floor in this session
    #     floor_i = self._imaginary_p.loadURDF(os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0.0], useFixedBase=1)
    #     self._imaginary_robot.reset(self._imaginary_p)
    #
    #     self._imaginary_robot.soft_reset(self._imaginary_p)
    #
    #     # TODO: change torque limit for this session
    #
    #     self._imaginary_p.stepSimulation()

    # def rollout_one_step_imaginary(self):
    #     # and get the obs vector [no tar vel] in sim
    #     assert self.train_dyn
    #     assert self.pretrain_dyn
    #
    #     # robo_obs = self.obs[:-self.behavior_act_len]  # TODO: deprecate behavior_act_len
    #     robo_action = self.obs[-self.behavior_act_len:]
    #     # print(robo_obs, "in img obs")
    #     # print(robo_action, "in img act")
    #
    #     # robo_state_vec = self._imaginary_robot.transform_obs_to_state(robo_obs)
    #     robo_state_vec = self.robot.get_robot_raw_state_vec()
    #
    #     self._imaginary_robot.soft_reset_to_state(self._imaginary_p, robo_state_vec)
    #     robo_state_i = self._imaginary_robot.get_robot_raw_state_vec()
    #
    #     robo_action = np.clip(robo_action, -1.0, 1.0)       # should also clip
    #     for _ in range(self.control_skip):
    #         self._imaginary_robot.apply_action(robo_action)
    #         self._imaginary_p.stepSimulation()
    #         # if self.render:
    #         #     time.sleep(self._ts * 0.5)
    #
    #     return self._imaginary_robot.get_robot_observation(), robo_state_i       # pre-state_i

    # def rollout_one_step_imaginary_same_session(self):
    #     # and get the obs vector [no tar vel] in sim
    #     assert self.train_dyn
    #     assert self.pretrain_dyn
    #
    #     robo_action = self.obs[-self.behavior_act_len:]
    #
    #     robo_action = np.clip(robo_action, -1.0, 1.0)       # should also clip
    #     for _ in range(self.control_skip):
    #         self.robot.apply_action(robo_action)
    #         self._p.stepSimulation()
    #
    #     return self.robot.get_robot_observation()

    # def calc_obs_dist_pretrain(self, obs1, obs2):
    #     # TODO quat dist
    #     # print(np.array(obs1))
    #     # print("2", np.array(obs2))
    #     # print(np.linalg.norm(np.array(obs1) - np.array(obs2)))
    #     # print(1.5-np.linalg.norm(np.array(obs1[36:]) - np.array(obs2[36:])))
    #     # return -np.mean(np.abs((np.array(obs1[:36]) - np.array(obs2[:36])) / np.array(obs2[:36]))) * 100
    #     return 0.4-np.sum(np.abs(np.array(obs1[:36]) - np.array(obs2[:36])))      # obs len 48
    #     # return 6.0 -np.sum(np.abs(np.array(obs1[3:]) - np.array(obs2[3:]))) \
    #     #        -np.sum(np.abs(np.array(obs1[:6]) - np.array(obs2[:6]))) * 20.0    # obs len 48

    def step(self, a):
        # TODO: currently for laika, env_action is 12D, 4 feet 3D without wrench
        if self.train_dyn:
            env_action = a
            robo_action = self.past_bact_array[0]       # after tanh
        else:
            robo_action = a
            robo_action = np.tanh(robo_action)
            # update past_bact after tanh
            utils.push_recent_value(self.past_bact_array, robo_action)

            # env_pi_obs = utils.select_and_merge_from_s_a(
            #                 s_mt=list(self.past_obs_array),
            #                 a_mt=list(self.past_bact_array),
            #                 s_idx=self.generator_past_obs_t_idx,
            #                 a_idx=self.generator_past_act_t_idx
            #             )

            # TODO: calculate 4x split obs here
            obs_w_dq = self.robot.get_robot_observation(with_vel=True)
            env_pi_obs = np.concatenate((obs_w_dq, robo_action))
            env_pi_obs_nn = utils.wrap(env_pi_obs, is_cuda=self.cuda_env)
            with torch.no_grad():
                _, env_action_nn, _, self.recurrent_hidden_states = self.dyn_actor_critic.act(
                    env_pi_obs_nn, self.recurrent_hidden_states, self.masks, deterministic=False
                )
            env_action = utils.unwrap(env_action_nn, is_cuda=self.cuda_env)
        #
        # env_action = np.tanh(env_action)

        # if self.ratio is None:
        #     self.ratio = np.array([env_action / robo_action])
        # else:
        #     self.ratio = np.append(self.ratio, [env_action / robo_action], axis=0)
            # self.ratios = np.append(self.ratios, [env_action / robo_action], axis=0)
            #
            # env_pi_obs_feat = self.feat_select_func(self.obs)
            # dis_state = np.concatenate((env_pi_obs_feat, robo_action))
            # dis_state = utils.wrap(dis_state, is_cuda=self.cuda_env)

        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        x_0 = root_pos[0]

        # this is post noise (unseen), different from seen diversify of self.enlarge_act_scale
        if self.act_noise:
            robo_action = utils.perturb(robo_action, 0.05, self.np_random)

        # when call info, should call before sim_step() as in v4 (append s_t+1 later)
        # info will be used to construct D input outside.
        past_info = self.construct_past_traj_window()

        # # TODO
        # if self.pretrain_dyn:
        #     # self.state_id = self._p.saveState()
        #     self.img_obs, pre_s_i = self.rollout_one_step_imaginary()     # takes the old self.obs
        #     # img_obs = self.rollout_one_step_imaginary_same_session()
        #     # self._p.restoreState(self.state_id)
        #     pre_s = self.robot.get_robot_raw_state_vec()
        #     # print(pre_s_i)
        #     # print(pre_s)
        #     assert np.allclose(pre_s, pre_s_i, atol=1e-5)

        for _ in range(self.control_skip):
            self.robot.apply_action(robo_action)
            self.apply_scale_clip_conf_from_pi_new(env_action)
            self._p.stepSimulation()
            if self.render:
                time.sleep(self._ts * 1.0)
            self.timer += 1

        obs_new = self.get_extended_observation()       # and update past_obs_array
        past_info += [self.past_obs_array[0]]       # s_t+1

        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        x_1 = root_pos[0]
        self.velx = (x_1 - x_0) / (self.control_skip * self._ts)

        y_1 = root_pos[1]
        height = root_pos[2]
        q, dq = self.robot.get_q_dq(self.robot.ctrl_dofs)
        # print(np.max(np.abs(dq)))
        # in_support = self.robot.is_root_com_in_support()

        if not self.pretrain_dyn:
            reward = self.ab  # alive bonus
            tar = np.minimum(self.timer / 500, self.max_tar_vel)
            reward += np.minimum(self.velx, tar) * self.vel_r_weight
            # print("v", self.velx, "tar", tar)
            reward += -self.energy_weight * np.square(robo_action).sum()
            # print("act norm", -self.energy_weight * np.square(a).sum())

            pos_mid = 0.5 * (self.robot.ll + self.robot.ul)
            q_scaled = 2 * (q - pos_mid) / (self.robot.ul - self.robot.ll)
            joints_at_limit = np.count_nonzero(np.abs(q_scaled) > 0.97)
            reward += -self.jl_weight * joints_at_limit
            # print("jl", -self.jl_weight * joints_at_limit)

            reward += -np.minimum(np.sum(np.square(dq)) * self.dq_pen_weight, 5.0)
            weight = np.array([2.0, 1.0, 1.0] * 4)
            reward += -np.minimum(np.sum(np.square(q - self.robot.init_q) * weight) * self.q_pen_weight, 5.0)
            # print("vel pen", -np.minimum(np.sum(np.abs(dq)) * self.dq_pen_weight, 5.0))
            # print("pos pen", -np.minimum(np.sum(np.square(q - self.robot.init_q)) * self.q_pen_weight, 5.0))

            y_1 = root_pos[1]
            reward += -y_1 * 0.5
            # print("dev pen", -y_1*0.5)
        else:
            # reward = self.calc_obs_dist_pretrain(self.img_obs[:-4], self.obs[:len(self.img_obs[:-4])])
            reward = 0  # TODO

        # print("______")
        # print(in_support)

        # print("h", height)
        # print("dq.", np.abs(dq))
        # print((np.abs(dq) < 50).all())

        # print("------")
        # conf policy will not have body-in-contact flag
        not_done = (np.abs(dq) < 90).all() and (height > 0.2) and (height < 1.0)
        # not_done = (abs(y_1) < 5.0) and (height > 0.1) and (height < 1.0) and (rpy[2] > 0.1)
        # not_done = True
        #
        # if not not_done:
        #     print(self.ratio.shape)
        #     labels = list("123456789ABC")
        #     data = self.ratio
        #     from matplotlib import pyplot as plt
        #     width = 0.4
        #     fig, ax = plt.subplots()
        #     for i, l in enumerate(labels):
        #         x = np.ones(data.shape[0]) * i + (np.random.rand(data.shape[0]) * width - width / 2.)
        #         ax.scatter(x, data[:, i], s=25)
        #         median = np.median(data[:, i])
        #         ax.plot([i - width / 2., i + width / 2.], [median, median], color="k")
        #
        #     plt.ylim(-5, 5)
        #     ax.set_xticks(range(len(labels)))
        #     ax.set_xticklabels(labels)
        #     plt.show()
        #     self.ratio = None

        # if not self.train_dyn:
        #     dis_action = self.feat_select_func(self.obs)
        #     dis_action = utils.wrap(dis_action, is_cuda=self.cuda_env)
        #     d_score = self.discri.predict_prob_single_step(dis_state, dis_action)
        #     self.d_scores.append(utils.unwrap(d_score, is_cuda=self.cuda_env))
        #     # if len(self.d_scores) > 20 and np.mean(self.d_scores[-20:]) < 0.4:
        #     #     not_done = False
        #     # if not not_done or self.timer==1000:
        #     #     print(np.mean(self.d_scores))

        return obs_new, reward, not not_done, {"sas_window": past_info}

    # def return_imaginary_obs(self):
    #     # mods self.obs
    #     obs_i = np.copy(self.obs)
    #     # obs_i[:len(self.img_obs[:-4])] = self.img_obs[:-4]
    #     obs_i[:len(self.img_obs)] = self.img_obs
    #     return obs_i

    def apply_scale_clip_conf_from_pi_new(self, con_f):

        approx_mass = 26.0
        max_fz = approx_mass * 9.81 * 2  # 2mg        # TODO

        for foot_ind, link in enumerate(self.robot.feet):

            this_con_f = np.tanh(con_f[foot_ind * 3: (foot_ind + 1) * 3])   # [-1 ,1]

            pos, _ = self.robot.get_link_com_xyz_orn(link, fk=1)
            if pos[2] < 0.01:
                # first dim represents fz
                # fz = np.abs(this_con_f[0]) * max_fz
                fz = (this_con_f[0] + 1) / 2.0 * max_fz
            else:
                fz = 0.0

            fx = this_con_f[1] * 1.5 * fz
            fy = this_con_f[2] * 1.5 * fz

            utils.apply_external_world_force_on_local_point(self.robot.go_id, link,
                                                            [fx, fy, fz],
                                                            [0, 0, 0],
                                                            self._p)

    def construct_past_traj_window(self):
        # st, ... st-9, at, ..., at-9
        # call this before s_t+1 enters deque
        # order does not matter as long as it is the same in policy & expert batch
        # print(list(self.past_obs_array) + list(self.past_act_array))
        return list(self.past_obs_array) + list(self.past_bact_array)

    def get_ave_dx(self):
        return self.velx

    def get_dist(self):
        return self.robot.get_link_com_xyz_orn(-1)[0][0]

    def get_extended_observation(self):

        # with vel false
        cur_state = self.robot.get_robot_observation(with_vel=False)

        if self.obs_noise:
            cur_state = utils.perturb(cur_state, 0.1, self.np_random)

        # then update past obs
        utils.push_recent_value(self.past_obs_array, cur_state)

        # then construct behavior obs
        b_obs_all = utils.select_and_merge_from_s_a(
            s_mt=list(self.past_obs_array),
            a_mt=list(self.past_bact_array),
            s_idx=self.behavior_past_obs_t_idx,
            a_idx=np.array([])
        )
        # if train motor, return behavior obs and we are done
        if not self.train_dyn:
            return b_obs_all

        # else, train dyn
        # rollout b_pi
        obs_nn = utils.wrap(b_obs_all, is_cuda=self.cuda_env)
        with torch.no_grad():
            _, action_nn, _, self.recurrent_hidden_states = self.go_actor_critic.act(
                obs_nn, self.recurrent_hidden_states, self.masks, deterministic=False
            )
        b_cur_act = list(utils.unwrap(action_nn, is_cuda=self.cuda_env))
        b_cur_act = utils.perturb(b_cur_act, self.enlarge_act_range, self.np_random)
        b_cur_act = np.tanh(b_cur_act)

        # Store action after tanh (-1,1)
        utils.push_recent_value(self.past_bact_array, b_cur_act)

        #  construct G obs from updated past obs&b_act
        # g_obs_all = utils.select_and_merge_from_s_a(
        #     s_mt=list(self.past_obs_array),
        #     a_mt=list(self.past_bact_array),
        #     s_idx=self.generator_past_obs_t_idx,
        #     a_idx=self.generator_past_act_t_idx
        # )

        # TODO: calculate 4x split obs here
        obs_w_dq = self.robot.get_robot_observation(with_vel=True)
        g_obs_all = np.concatenate((obs_w_dq, b_cur_act))

        return g_obs_all

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
