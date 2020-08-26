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

import torch
import numpy as np
import gzip, pickle, pickletools
import joblib
import os
import pybullet


def load(policy_dir: str, env_name: str, is_cuda: bool, iter_num=None):
    """Loads parameters for a specified policy.

    Args:
        policy_dir: The directory to load the policy from.
        env_name: The environment name of the policy.
        is_cuda: Whether to use gpu.
        iter_num: The iteration of the policy model to load.

    Returns:
        actor_critic: The actor critic model.
        ob_rms: ?
        recurrent_hidden_states: The recurrent hidden states of the model.
        masks: ?
    """
    if iter_num is not None and iter_num >= 0:
        path = os.path.join(policy_dir, env_name + "_" + str(int(iter_num)) + ".pt")
    else:
        path = os.path.join(policy_dir, env_name + ".pt")
    print(f"| loading policy from {path}")
    if is_cuda:
        actor_critic, ob_rms = torch.load(path)
    else:
        actor_critic, ob_rms = torch.load(path, map_location="cpu")
    d = "cuda" if is_cuda else "cpu"
    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size, device=torch.device(d))
    masks = torch.zeros(1, 1, device=torch.device(d))
    return (
        actor_critic,
        ob_rms,
        recurrent_hidden_states,
        masks,
    )


def load_gail_discriminator(policy_dir: str, env_name: str, is_cuda: bool, iter_num=None):
    """Loads parameters for a specified gail discriminator.

    Args:
        policy_dir: The directory to load the policy from.
        env_name: The environment name of the policy.
        is_cuda: Whether to use gpu.
        iter_num: The iteration of the policy model to load.

    Returns:
        discri: the gail D
        recurrent_hidden_states: The recurrent hidden states of the model.
        masks: ?
    """
    if iter_num is not None and iter_num >= 0:
        path = os.path.join(policy_dir, env_name + "_" + str(int(iter_num)) + "_D.pt")
    else:
        path = os.path.join(policy_dir, env_name + "_D.pt")
    print(f"| loading gail discriminator from {path}")
    if is_cuda:
        discri = torch.load(path)
    else:
        discri = torch.load(path, map_location="cpu")
    return discri


def wrap(obs, is_cuda: bool) -> torch.Tensor:
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
        action = action.detach().numpy()
    return action


def perturb(arr, r=0.02, np_rand_gen=np.random):
    r = np.abs(r)
    return np.copy(
        np.array(arr) + np_rand_gen.uniform(low=-r, high=r, size=len(arr))
    )


def perturb_scalar(num, r=0.02, np_rand_gen=np.random):
    r = np.abs(r)
    return num + np_rand_gen.uniform(low=-r, high=r)


def load_combined_sas_from_pickle(pathname, downsample_freq=1, load_num_trajs=None):
    # if load_num_trajs None, load all trajs
    with open(pathname, "rb") as handle:
        saved_file = pickle.load(handle)
    # with open(pathname, "rb") as handle:
    #     saved_file = joblib.load(handle)
    # with gzip.open(pathname, 'rb') as f:
    #     p = pickle.Unpickler(f)
    #     saved_file = p.load()

    n_trajs = len(saved_file)
    # See https://github.com/pytorch/pytorch/issues/14886
    # .long() for fixing bug in torch v0.4.1
    start_idx = torch.randint(
        0, downsample_freq, size=(n_trajs,)).long()

    XY = []
    for traj_idx, traj_tuples in saved_file.items():
        XY.extend(traj_tuples[start_idx[traj_idx]::downsample_freq])  # downsample the rows
        if load_num_trajs and traj_idx >= load_num_trajs - 1:
            break
    return np.array(XY)

def load_feat_sas_from_pickle(pathname, downsample_freq=1, load_num_trajs=None):
    # if load_num_trajs None, load all trajs
    with open(pathname, "rb") as handle:
        saved_file = pickle.load(handle)
    # with open(pathname, "rb") as handle:
    #     saved_file = joblib.load(handle)
    # with gzip.open(pathname, 'rb') as f:
    #     p = pickle.Unpickler(f)
    #     saved_file = p.load()

    n_trajs = len(saved_file)
    # See https://github.com/pytorch/pytorch/issues/14886
    # .long() for fixing bug in torch v0.4.1
    start_idx = torch.randint(
        0, downsample_freq, size=(n_trajs,)).long()

    sas = []
    for traj_idx, traj_tuples in saved_file.items():
        sas.extend(traj_tuples[start_idx[traj_idx]::downsample_freq])  # downsample the rows
        if load_num_trajs and traj_idx >= load_num_trajs - 1:
            break

    s = list(np.array(sas)[:, 0])
    a = list(np.array(sas)[:, 1])
    s_next = list(np.array(sas)[:, 2])

    print(np.array(s).shape)
    print(np.array(a).shape)
    print(np.array(s_next).shape)

    return np.array(s), np.array(a), np.array(s_next)


def get_link_com_xyz_orn(body_id, link_id, bullet_session):
    # get the world transform (xyz and quaternion) of the Center of Mass of the link
    # We *assume* link CoM transform == link shape transform (the one you use to calculate fluid force on each shape)
    assert link_id >= -1
    if link_id == -1:
        link_com, link_quat = bullet_session.getBasePositionAndOrientation(body_id)
    else:
        link_com, link_quat, *_ = bullet_session.getLinkState(body_id, link_id, computeForwardKinematics=1)
    return list(link_com), list(link_quat)


def apply_external_world_force_on_local_point(body_id, link_id, world_force, local_com_offset, bullet_session):
    link_com, link_quat = get_link_com_xyz_orn(body_id, link_id, bullet_session)
    _, inv_link_quat = bullet_session.invertTransform([0., 0, 0], link_quat)  # obj->world
    local_force, _ = bullet_session.multiplyTransforms([0., 0, 0], inv_link_quat, world_force, [0, 0, 0, 1])
    bullet_session.applyExternalForce(body_id, link_id, local_force,
                                      local_com_offset, flags=bullet_session.LINK_FRAME)


def replace_obs_with_feat(obs, is_cuda : bool, feat_select_func=None, return_tensor=False):
    # obs is a tensor of (num_processes, obs_dim)
    # if feat_select none, selection is identity function where obs is unchanged

    def identity(x):
        return x

    if feat_select_func is None:
        feat_select_func = identity

    obs = obs.cpu() if is_cuda else obs
    obs_un = obs.detach().numpy()
    feat_chunk = []
    for obs_each in obs_un:
        feat_chunk.append(feat_select_func(obs_each))

    if return_tensor:
        feat_chunk = torch.Tensor(feat_chunk)
        if is_cuda:
            feat_chunk = feat_chunk.cuda()

    return feat_chunk

