#  MIT License
#
#  Copyright (c) 2017 Ilya Kostrikov and (c) 2020 Google LLC
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from third_party.a2c_ppo_acktr import algo, utils
from third_party.a2c_ppo_acktr.algo import gail
from third_party.a2c_ppo_acktr.arguments import get_args
from third_party.a2c_ppo_acktr.envs import make_vec_envs
from third_party.a2c_ppo_acktr.model import Policy
from third_party.a2c_ppo_acktr.storage import RolloutStorage

from gan import utils as gan_utils

import inspect
import pybullet_envs
import argparse
import pybullet as p

import my_pybullet_envs

import logging
import sys
sys.path.append("third_party")


def main():
    args, extra_dict = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, render=False, **extra_dict)

    if args.warm_start == '':
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy, 'hidden_size': args.hidden_size})
        actor_critic.to(device)
    else:
        # TODO: assume no state normalize ob_rms
        if args.cuda:
            actor_critic, _ = torch.load(args.warm_start)
        else:
            actor_critic, _ = torch.load(args.warm_start, map_location='cpu')
        # actor_critic.reset_variance(envs.action_space)
        # actor_critic.to(device)

    # actor_critic, _ = torch.load("/home/yifengj/tmp/bullet-dart/trained_models_Gdyn_laika_bullet_soft23gt_1_few2/ppo/LaikagoConFEnv-v1.pt")

    dummy = gym.make(args.env_name, render=False, **extra_dict)
    save_path = os.path.join(args.save_dir, args.algo)
    print("SAVE PATH:")
    print(save_path)
    try:
        os.makedirs(save_path)
    except FileExistsError:
        print("warning: path existed")
        # input("warning: path existed")
    except OSError:
        exit()
    pathname = os.path.join(save_path, "source_test.py")
    text_file = open(pathname, "w+")
    text_file.write(dummy.getSourceCode())
    text_file.close()
    print("source file stored")
    # input("source file stored press enter")

    dummy.reset()
    feat_select_func = None
    if args.gail_dyn:
        try:
            # TODO: hopper
            # feat_select_func = dummy.robot.feature_selection_laika
            # feat_select_func = dummy.robot.feature_selection_all_laika
            # feat_select_func = dummy.robot.feature_selection_withq_laika
            feat_select_func = dummy.robot.feature_selection_G2BD_laika_v2
        except:
            print("feat select not found")
            pass

    # dummy.close()

    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler("{0}/{1}.log".format(save_path, "console_output"))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1

        # TODO: s1 is just s0 shifted by 1 in time.
        expert_s0, expert_a0, expert_s1 = gan_utils.load_feat_sas_from_pickle(
            args.gail_traj_path,
            downsample_freq=int(args.gail_downsample_frequency),
            load_num_trajs=args.gail_traj_num
        )

        s_dim = expert_s0.shape[1]
        a_dim = expert_a0.shape[1]

        if not args.gail_dyn:
            assert s_dim == envs.observation_space.shape[0]
            assert a_dim == envs.action_space.shape[0]
        # else:
        # s_dim is s_feat_dim, the following not ture anymore
        # assert s_dim + a_dim == envs.observation_space.shape[0]

        if not args.gail_dyn:
            discr = gail.Discriminator(
                s_dim + a_dim, args.gail_dis_hdim,
                device)

            # discr = torch.load("/home/yifengj/tmp/bullet-dart/trained_models_Gdyn_laika_bullet_soft23gt_1_few2/ppo/LaikagoConFEnv-v1_D.pt")
        else:
            # learning dyn gail
            discr = gail.Discriminator(
                s_dim + a_dim + s_dim, args.gail_dis_hdim,
                device)

        if not args.gail_dyn:
            expert_dataset = TensorDataset(Tensor(expert_s0), Tensor(expert_a0))
        else:
            # learning dyn gail
            expert_s0a0 = np.concatenate((expert_s0, expert_a0), axis=1)
            expert_dataset = TensorDataset(Tensor(expert_s0a0), Tensor(expert_s1))

        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = DataLoader(
            expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)
    else:
        discr = None
        gail_train_loader = None

    obs = envs.reset()
    obs_feat = gan_utils.replace_obs_with_feat(obs, args.cuda, feat_select_func, return_tensor=True)
    feat_len = obs_feat.size(1)     # TODO: multi-dim obs broken

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size,
                              feat_len)


    rollouts.obs[0].copy_(obs)
    rollouts.obs_feat[0].copy_(obs_feat)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10000)
    gail_rewards = deque(maxlen=10)  # this is just a moving average filter

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    from third_party.a2c_ppo_acktr.baselines.common.running_mean_std import RunningMeanStd
    ret_rms = RunningMeanStd(shape=())

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # print(args.num_steps) 300*8
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            obs_feat = gan_utils.replace_obs_with_feat(obs, args.cuda, feat_select_func, return_tensor=True)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = Tensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = Tensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks, obs_feat)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        gail_loss, gail_loss_e, gail_loss_p = None, None, None
        if args.gail:
            # TODO: odd. turn this off for now since no state normalize
            # if j >= 10:
            #     envs.venv.eval()

            gail_epoch = args.gail_epoch
            # if j < 10:
            #     gail_epoch = 100  # TODO: Warm up

            for _ in range(gail_epoch):
                gail_loss, gail_loss_e, gail_loss_p = discr.update(gail_train_loader, rollouts,
                                                                   utils.get_vec_normalize(envs)._obfilt,
                                                                   args.gail_dyn, a_dim)    # TODO

            num_of_dones = (1.0 - rollouts.masks).sum().cpu().numpy() \
                           + args.num_processes / 2
            # print(num_of_dones)
            # TODO: hardcoded tar episode length 100
            num_of_expert_dones = (args.num_steps * args.num_processes) / 100.0
            # print(num_of_expert_dones)

            # d_sa < 0.5 if pi too short (too many pi dones),
            # d_sa > 0.5 if pi too long
            d_sa = 1 - num_of_dones / (num_of_dones + num_of_expert_dones)
            # print(d_sa)
            r_sa = np.log(d_sa) - np.log(1 - d_sa)        # d->1, r->inf

            # overwriting rewards by gail
            if not args.gail_dyn:
                for step in range(args.num_steps):
                    rollouts.rewards[step], returns = discr.predict_reward(
                        rollouts.obs[step], rollouts.actions[step], args.gamma,
                        rollouts.masks[step])
            else:
                for step in range(args.num_steps):
                    cur_obs_a = rollouts.obs[step, :, -a_dim:]  # second dim is num_proc
                    cur_obs_feat_a = torch.cat((rollouts.obs_feat[step], cur_obs_a), 1)
                    rollouts.rewards[step], returns = discr.predict_reward(
                        cur_obs_feat_a, rollouts.obs_feat[step + 1], args.gamma,
                        rollouts.masks[step], offset=-r_sa)

                    # redo reward normalize after overwriting
                    # print(rollouts.rewards[step], returns)
                    ret_rms.update(returns.view(-1).cpu().numpy())
                    rews = rollouts.rewards[step].view(-1).cpu().numpy()
                    rews = np.clip(rews / np.sqrt(ret_rms.var + 1e-7),
                                   -10.0, 10.0)
                    # print(ret_rms.var)    # just one number
                    rollouts.rewards[step] = Tensor(rews).view(-1,1)
                    # print("after", rollouts.rewards[step], returns)

            # final returns
            # print(returns)
            gail_rewards.append(torch.mean(returns).cpu().data)

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, not args.no_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
            or j == num_updates - 1) and args.save_dir != "":
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + "_" + str(j) + ".pt"))

            if args.gail:
                torch.save(discr, os.path.join(save_path, args.env_name + "_D.pt"))
                torch.save(discr, os.path.join(save_path, args.env_name + "_" + str(j) + "_D.pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            rootLogger.info(
                ("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes:" +
                 " mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, " +
                 "dist en {}, l_pi {}, l_vf {}, recent_gail_r {}," +
                 "loss_gail {}, loss_gail_e {}, loss_gail_p {}\n")
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss, np.mean(gail_rewards),
                            gail_loss, gail_loss_e, gail_loss_p))
            # actor_critic.dist.logstd._bias,

        episode_rewards.clear()


if __name__ == "__main__":
    main()
