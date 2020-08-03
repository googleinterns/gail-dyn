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

from matplotlib import pyplot as plt

import pickle

from third_party.a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from third_party.a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

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
parser.add_argument(
    "--load-dis",
    type=int,
    default=0,
    help="whether to load gail discriminator for debugging",
)
parser.add_argument(
    "--enlarge-act-range",
    type=int,
    default=1,
    help="add white noise to action during rollout",
)
parser.add_argument(
    "--non-det",
    type=int,
    default=0,
    help="whether to use a non-deterministic policy, 1 true 0 false",
)
parser.add_argument(
    "--iter", type=int, default=-1, help="which iter pi to test"
)
parser.add_argument(
    "--r-thres",
    type=int,
    default=4000,
    help="The threshold reward value above which it is considered a success.",
)

args, unknown = parser.parse_known_args()  # this is an 'internal' method
# which returns 'parsed', the same as what parse_args() would return
# and 'unknown', the remainder of that
# the difference to parse_args() is that it does not exit when it finds redundant arguments
np.set_printoptions(precision=2, suppress=None, threshold=sys.maxsize)


def try_numerical(string):
    # convert all extra arguments to numerical type (float) if possible
    # assume always float (pass bool as 0 or 1)
    # else, keep the argument as string type
    try:
        num = float(string)
        return num
    except ValueError:
        return string


def pairwise(iterable):
    # s -> (s0, s1), (s2, s3), (s4, s5), ...
    a = iter(iterable)
    return zip(a, a)


def wrap(obs: np.ndarray, is_cuda: bool) -> torch.Tensor:
    obs = torch.Tensor([obs])
    if is_cuda:
        obs = obs.cuda()
    return obs


def unwrap(action: torch.Tensor, is_cuda: bool, clip=False) -> np.ndarray:
    action = action.squeeze()
    action = action.cpu() if is_cuda else action
    if clip:
        action = np.clip(action.numpy(), -1.0, 1.0)
    else:
        action = action.numpy()
    return action


def plot_avg_dis_reward(args, avg_reward_list, dxs):
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


for arg, value in pairwise(
        unknown
):  # note: assume always --arg value (no --arg)
    assert arg.startswith(("-", "--"))
    parser.add_argument(
        arg, type=try_numerical
    )  # assume always float (pass bool as 0 or 1)

args_w_extra = parser.parse_args()
args_dict = vars(args)
args_w_extra_dict = vars(args_w_extra)
extra_dict = {
    k: args_w_extra_dict[k] for k in set(args_w_extra_dict) - set(args_dict)
}

is_cuda = True
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

# TODO: not working yet
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.src_env_name == "":
    env_name_transfer = args.env_name
else:
    env_name_transfer = args.src_env_name
if args.iter >= 0:
    path = os.path.join(
        args.load_dir, env_name_transfer + "_" + str(args.iter) + ".pt"
    )
    path_D = os.path.join(
        args.load_dir, env_name_transfer + "_" + str(args.iter) + "_D.pt"
    )
else:
    path = os.path.join(args.load_dir, env_name_transfer + ".pt")
    path_D = os.path.join(args.load_dir, env_name_transfer + "_D.pt")

if is_cuda:
    actor_critic, ob_rms = torch.load(path)
    if args.load_dis:
        discri = torch.load(path_D)
else:
    actor_critic, ob_rms = torch.load(path, map_location="cpu")
    if args.load_dis:
        discri = torch.load(path_D, map_location="cpu")

if ob_rms:
    print(ob_rms.mean)
    print(ob_rms.var)
    print(ob_rms.count)
    input("ob_rms")

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(
    1, actor_critic.recurrent_hidden_state_size
)
masks = torch.zeros(1, 1)

all_trajs = {}
cur_traj = []
cur_traj_idx = 0

obs = env.reset()
# print("obs", obs)
# input("reset, press enter")
done = False

reward_total = 0

list_length = 0
dist = 0
last_dist = 0

if args.load_dis:
    dis_rewards = []
    dxs = []

while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det
        )
        if args.enlarge_act_range:
            # 15% noise if a clipped to -1, 1
            action += (torch.rand(action.size()).to(device) - 0.5) * 0.3

    if args.save_traj:
        tuple = list(unwrap(obs, is_cuda=is_cuda))

    if args.load_dis:
        dis_state = obs
    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)

    if args.save_traj:
        # print("action", action)
        # print("obs", obs)
        tuple += list(unwrap(action, is_cuda=is_cuda))
        tuple += list(unwrap(obs, is_cuda=is_cuda))
        # print(tuple)
        assert len(tuple) == env_core.action_dim + env_core.obs_dim * 2
        cur_traj.append(tuple)

    if args.load_dis:
        # print("aaa")
        dis_action = obs[:, :env_core.behavior_obs_len]
        dis_r = discri.predict_prob_single_step(dis_state, dis_action)
        dis_rewards.append(unwrap(dis_r, is_cuda=is_cuda))
        dxs.append(env_core.get_ave_dx())

    try:
        env_core.cam_track_torso_link()
        last_dist = dist
        dist = env_core.get_dist()
    except:
        print("not bullet locomotion env")

    reward_total += reward.cpu().numpy()[0][0]

    if done:
        print(
            f"{args.load_dir}\t"
            f"tr: {reward_total:.1f}\t"
            f"x: {last_dist:.2f}\t"
        )
        reward_total = 0.0

        if args.save_traj:
            print(np.array(cur_traj).shape)
            all_trajs[cur_traj_idx] = cur_traj
            cur_traj_idx += 1
            cur_traj = []
            if cur_traj_idx >= args.num_trajs:
                break

        if args.load_dis:
            plot_avg_dis_reward(args, dis_rewards, dxs)
            dis_rewards = []
            dxs = []

    masks.fill_(0.0 if done else 1.0)

with open(args.save_path, "wb") as handle:
    # print(all_trajs)
    pickle.dump(all_trajs, handle, protocol=pickle.HIGHEST_PROTOCOL)
