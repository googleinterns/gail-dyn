import torch
import numpy as np
import pickle

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

    n_trajs = len(saved_file)
    # See https://github.com/pytorch/pytorch/issues/14886
    # .long() for fixing bug in torch v0.4.1
    start_idx = torch.randint(
        0, downsample_freq, size=(n_trajs,)).long()

    XY = []
    for traj_idx, traj_tuples in saved_file.items():
        XY.extend(traj_tuples[start_idx[traj_idx]::downsample_freq])    # downsample the rows
        if load_num_trajs and traj_idx >= load_num_trajs - 1:
            break
    return np.array(XY)