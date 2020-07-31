import argparse
import os

# workaround to unpickle olf model files
import sys
import time

import numpy as np
import torch

import gym
import my_pydart_envs

import pickle

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

sys.path.append("a2c_ppo_acktr")

parser = argparse.ArgumentParser(description="RL")
parser.add_argument(
    "--seed", type=int, default=1, help="random seed (default: 1)"
)
parser.add_argument(
    "--env-name",
    default="HumanoidSwimmerEnv-v1",
    help="environment to train on (default: PongNoFrameskip-v4)",
)
parser.add_argument(
    "--load-dir",
    default="./trained_models/",
    help="directory to save agent logs (default: ./trained_models/)",
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
    "--r_thres",
    type=int,
    default=4000,
    help="The threshold reward value above which it is considered a success.",
)

args, unknown = parser.parse_known_args()  # this is an 'internal' method
# which returns 'parsed', the same as what parse_args() would return
# and 'unknown', the remainder of that
# the difference to parse_args() is that it does not exit when it finds redundant arguments


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
)
# dont know why there are so many wrappers in make_vec_envs...
env_core = env.venv.venv.envs[0].env.env

env_name_transfer = "HopperURDFEnv-v1"
if args.iter >= 0:
    path = os.path.join(
        args.load_dir, args.env_name + "_" + str(args.iter) + ".pt"
    )
else:
    path = os.path.join(args.load_dir, env_name_transfer + ".pt")

if is_cuda:
    actor_critic, ob_rms = torch.load(path)
else:
    actor_critic, ob_rms = torch.load(path, map_location="cpu")

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

obs = env.reset()
# print("obs", obs)
# input("reset, press enter")
done = False

if args.env_name.find("Bullet") > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if p.getBodyInfo(i)[0].decode() == "torso":
            torsoId = i

reward_total = 0

list_length = 0
dist = 0

while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det
        )

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)

    if extra_dict["render"]:
        env.render()

    reward_total += reward
    last_dist = dist
    dist = env_core.get_dist()

    if done:
        print(
            f"{args.load_dir}\t"
            f"tr: {reward_total.numpy()[0][0]:.1f}\t"
            f"x: {last_dist:.2f}\t"
        )
        reward_total = 0.0
        input("press enter")

    masks.fill_(0.0 if done else 1.0)
