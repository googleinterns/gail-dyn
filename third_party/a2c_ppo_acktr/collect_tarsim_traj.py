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

import argparse
import os
from typing import *

import sys
import time

import numpy as np
import torch

import gym
import my_pybullet_envs
import random

from matplotlib import pyplot as plt

import pickle
import joblib

from third_party.a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from third_party.a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
from third_party.a2c_ppo_acktr.arguments import parse_args_with_unknown
from gan.utils import *


def plot_avg_dis_prob(args, avg_reward_list, dxs):
    env_name = args.env_name
    _, axs = plt.subplots(2, 1)
    axs[0].plot(avg_reward_list)
    # plt.title('Average Dis Reward, Env: {}'.format(env_name))
    plt.xlabel('steps')
    # plt.ylabel('average reward')
    axs[1].plot(dxs)
    plt.show()
    np.save(os.path.join('./imgs', env_name + '_avg_dreward.npy'), np.array(avg_reward_list))
    plt.savefig(os.path.join('./imgs', env_name + '_avg_dreward.png'))
    input("press enter plt")

def plot_avg_dis_prob_2(args, avg_reward_list, avg_reward_list_2, dxs):
    env_name = args.env_name
    _, axs = plt.subplots(2, 1)
    axs[0].plot(avg_reward_list)
    axs[0].plot(avg_reward_list_2)
    # plt.title('Average Dis Reward, Env: {}'.format(env_name))
    plt.xlabel('steps')
    # plt.ylabel('average reward')
    axs[1].plot(dxs)
    plt.show()
    np.save(os.path.join('./imgs', env_name + '_avg_dreward.npy'), np.array(avg_reward_list))
    plt.savefig(os.path.join('./imgs', env_name + '_avg_dreward.png'))
    input("press enter plt")


sys.path.append("third_party")

parser = argparse.ArgumentParser(description="RL")
parser.add_argument(
    "--seed", type=int, default=1, help="random seed (default: 1)"
)
parser.add_argument(
    "--env-name",
    default="HumanoidSwimmerEnv-v1",
    help="environment to load and test on",
)
parser.add_argument(
    "--src-env-name",
    default="",
    help="environment to transfer policy from ("" if same as test env)",
)
parser.add_argument(
    "--load-dir",
    default="./trained_models/",
    help="directory to save agent logs (default: ./trained_models/)",
)
parser.add_argument(
    "--save-traj",
    type=int,
    default=0,
    help="whether to save traj tuples",
)
parser.add_argument(
    "--num-trajs",
    type=int,
    default=200,
    help="how many trajs to rollout/store",
)
parser.add_argument(
    "--save-path",
    default="./tmp.pkl",
    help="where the traj tuples are stored",
)
# parser.add_argument(
#     "--load-dis",
#     type=int,
#     default=0,
#     help="whether to load gail discriminator for debugging",
# )
# parser.add_argument(
#     "--enlarge-act-range",
#     type=float,
#     default=0.15,
#     help="add white noise to action during rollout",
# )
parser.add_argument(
    "--non-det",
    type=int,
    default=0,
    help="whether to use a non-deterministic policy, 1 true 0 false",
)
parser.add_argument(
    "--iter",
    type=int,
    default=None,
    help="which iter pi to test"
)
parser.add_argument(
    "--r-thres",
    type=int,
    default=4000,
    help="The threshold reward value above which it is considered a success.",
)

args, extra_dict = parse_args_with_unknown(parser)

np.set_printoptions(precision=2, suppress=None, threshold=sys.maxsize)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

is_cuda = False
device = "cuda" if is_cuda else "cpu"

args.det = not args.non_det

# If render is provided, use that. Otherwise, turn it on.
if "render" not in extra_dict:
    extra_dict["render"] = True

env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device=device,
    allow_early_resets=False,
    **extra_dict,
)
# dont know why there are so many wrappers in make_vec_envs...
env_core = env.venv.venv.envs[0].env.env

if args.src_env_name == "":
    env_name_transfer = args.env_name
else:
    env_name_transfer = args.src_env_name
actor_critic, ob_rms, recurrent_hidden_states, masks \
    = load(args.load_dir, env_name_transfer, is_cuda, args.iter)
# discri = None
# if args.load_dis:
#     discri = load_gail_discriminator(args.load_dir, env_name_transfer, is_cuda, args.iter)

if ob_rms:
    print(ob_rms.mean)
    print(ob_rms.var)
    print(ob_rms.count)
    input("ob_rms")

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

all_trajs = {}
cur_traj = []
cur_traj_idx = 0

obs = env.reset()
# print("obs", obs)
# input("reset, press enter")
done = False

reward_total = 0

list_rewards = []
list_traj_lengths = []
list_r_per_step = []
dist = 0
last_dist = 0

dis_probs_imaginary = None
dis_probs_real = None
dxs = []
# if args.load_dis:
#     dis_probs_imaginary = []
#     dis_probs_real = []

while True:
    # try:
    #     env_core.reset_counter = 5000
    # except:
    #     pass

    with torch.no_grad():
        # value, action, _, recurrent_hidden_states = actor_critic.act(
        #     obs, recurrent_hidden_states, masks, deterministic=True
        # )
        # action += torch.normal(torch.zeros(action.size()), 0.1).to(device)
        # print(action)
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det
        )

        # # TODO, name duplicate
        # # TODO parameter space noise
        # # xx% noise before tanh
        # action += (torch.rand(action.size()).to(device) - 0.5) * (args.enlarge_act_range * 2)
        # # print(action)

    # if args.save_traj:
    #     tuple_sas = []
    #     obs_feat = replace_obs_with_feat(obs, is_cuda, feat_select_func, return_tensor=False)
    #     tuple_sas.append(obs_feat[0])   # only one process env
    #
    #     # save clamped action (note: dyn envs might have action larger than 1)
    #     action = action.clamp(-1., 1)
    # print("obs", obs)
    # print("act", torch.tanh(action))



    # if args.load_dis:
    #     obs_feat = replace_obs_with_feat(obs, is_cuda, feat_select_func, return_tensor=True)
    #     dis_state = torch.cat((obs_feat, obs[:, env_core.behavior_obs_len:]), 1)

    # Obser reward and next obs
    obs, reward, done, info = env.step(action)
    list_r_per_step.append(reward)

    if args.save_traj:
        sas_window = info[0]["sas_window"]      # info[0] because 1-core dummy vec env.

        # tuple_sas.append(list(unwrap(action, is_cuda=is_cuda)))
        #
        # obs_feat = replace_obs_with_feat(obs, is_cuda, feat_select_func, return_tensor=False)
        # tuple_sas.append(obs_feat[0])

        next_obs = list(unwrap(obs, is_cuda=is_cuda))

        # print(sas_window)
        cur_traj.append(sas_window)

    # if args.load_dis:
    #     dis_action = replace_obs_with_feat(obs, is_cuda, feat_select_func, return_tensor=True)
    #     dis_r = discri.predict_prob_single_step(dis_state, dis_action)
    #     dis_probs_real.append(unwrap(dis_r, is_cuda=is_cuda))

        # if len(dis_probs_real)>20 and np.mean(dis_probs_real[-20:]) < 0.4:
        #     done = True
        #     env.reset()

        # try:
        #     obs_i = env_core.return_imaginary_obs()
        #     dis_action = obs_i[:env_core.behavior_obs_len]      # dis action is next state
        #     dis_action = wrap(dis_action, is_cuda=is_cuda)
        #     dis_action = replace_obs_with_feat(dis_action, is_cuda, feat_select_func, return_tensor=True)
        #     dis_r = discri.predict_prob_single_step(dis_state, dis_action)
        #     dis_probs_imaginary.append(unwrap(dis_r, is_cuda=is_cuda))
        # except:
        #     pass

    # dxs.append(env_core.get_ave_dx())

    try:
        env_core.cam_track_torso_link()
        last_dist = dist
        dist = env_core.get_dist()
    except:
        print("not bullet locomotion env")

    reward_total += reward.cpu().numpy()[0][0]

    if done:
        list_rewards.append(reward_total)
        list_traj_lengths.append(len(list_r_per_step))
        print(
            f"{args.load_dir}\t"
            f"tr: {reward_total:.1f}\t"
            f"x: {last_dist:.2f}\t"
            f"tr_ave: {reward_total/len(list_r_per_step):.2f}\t"
            f"total_per_step_r_ave: {np.sum(list_rewards)/np.sum(list_traj_lengths):.2f}\t"
        )
        reward_total = 0.0
        # env_core.reset_counter = 0

        cur_traj_idx += 1
        if cur_traj_idx >= args.num_trajs:
            break

        if args.save_traj:
            print(np.array(cur_traj).shape)
            all_trajs[cur_traj_idx] = cur_traj
            cur_traj = []

        # if args.load_dis:
        #     print(
        #         f"{np.array(dis_probs_real).mean()}\t"
        #     )
        #     # plot_avg_dis_prob_2(args, dis_probs_imaginary, dis_probs_real, list_r_per_step)
        #     dis_probs_imaginary = []
        #     dis_probs_real = []
        # else:
        #     # plot_avg_dis_prob(args, list_r_per_step, dxs)
        #     pass
        list_r_per_step = []
        dxs = []

    masks.fill_(0.0 if done else 1.0)

with open(args.save_path, "wb") as handle:
    # print(all_trajs)
    pickle.dump(all_trajs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # joblib.dump(all_trajs, handle)

bins_list = np.arange(40) * 50.0
print(bins_list)
plt.hist(list_rewards, alpha=0.5, label='r hist', bins=bins_list)
plt.legend(loc='upper right')
plt.show()
