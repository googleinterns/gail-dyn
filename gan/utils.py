import torch
import numpy as np


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