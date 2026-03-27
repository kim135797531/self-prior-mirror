#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""utils.py
Created by Dongmin Kim at 24. 8. 9.

This module does stuff.
"""
from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
from torch import distributions as D
from torch.distributions import constraints
from torch.nn import Module, functional as F
from einops import repeat, reduce, rearrange
from torch.distributions import constraints


# rewards = [rew1,     rew2,     ..., rew16] (16)
# gam_ret = [rew1val0, rew2val1, ..., rew16val15] (16)
# values  = [val0,     val1,     ..., val15,     val16] (17)
def storm_calc_lambda_return(rewards, values, gamma, gae_lambda):
    batch_size, batch_length = rewards.shape[:2]
    gamma_return = torch.zeros(
        (batch_size, batch_length + 1), dtype=rewards.dtype, device=rewards.device
    )
    gamma_return[:, -1] = values[:, -1]

    for t in reversed(range(batch_length)):
        gamma_return[:, t] = (
            rewards[:, t]
            + gamma * (1 - gae_lambda) * values[:, t]
            + gamma * gae_lambda * gamma_return[:, t + 1]
        )
    return gamma_return[:, :-1]


# rewards = [rew1,     rew2,     ..., rew16] (16)
# gam_ret = [rew1val1, rew2val2, ..., rew16val16] (16)
# values  = [val1,     val2,     ..., val16] (16)
def sheeprl_compute_lambda_values(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    gae_lambda: float,
):
    batch_len = values.shape[1]
    vals = [values[:, -1]]  # Last value
    interm = rewards + continues * values * (1 - gae_lambda)
    for t in reversed(range(batch_len)):
        vals.append(interm[:, t] + continues[:, t] * gae_lambda * vals[-1])

    ret = torch.stack(list(reversed(vals)), dim=1)[:, :-1]
    return ret


# def lambda_dreamer_values(rewards, value_preds, gamma, gae_lambda):
#     batch_length = rewards[:-1].size(0)
#     # gamma = 0.99
#     # gae_lambda = 0.95
#     lambda_returns = torch.zeros_like(rewards)
#     if type(gamma) in [int, float]:
#         gamma = torch.ones_like(rewards) * gamma
#     lambda_returns[-1] = rewards[-1] + gamma[-1] * value_preds[-1]
#     for step in reversed(range(batch_length)):
#         lambda_returns[step] = rewards[step] + gamma[step] * (
#             (1 - gae_lambda) * value_preds[step + 1]
#             + gae_lambda * lambda_returns[step + 1]
#         )
#     return lambda_returns


class EMAScalar:
    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    @torch.no_grad()
    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar

    def state_dict(self):
        return {"scalar": self.scalar, "decay": self.decay}

    def load_state_dict(self, state: dict):
        self.scalar = state["scalar"]
        self.decay = state["decay"]


def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage * len(flat_x))
    if torch.are_deterministic_algorithms_enabled():
        sorted_x, indices = flat_x.sort()
        per = sorted_x[kth]
    else:
        per = torch.kthvalue(flat_x, kth).values
    return per


def get_parameters(modules: Iterable[Module]):
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class FreezeParameters:
    def __init__(self, modules: Iterable[Module]):
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def state_dict(self):
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict):
        self.mean = state["mean"].copy()
        self.var = state["var"].copy()
        self.count = state["count"]


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def stack_states(rssm_states: list, dim=0):
    return dict(
        mean=torch.stack([state["mean"] for state in rssm_states], dim=dim),
        std=torch.stack([state["std"] for state in rssm_states], dim=dim),
        stoch=torch.stack([state["stoch"] for state in rssm_states], dim=dim),
        deter=torch.stack([state["deter"] for state in rssm_states], dim=dim),
    )


def flatten_state(rssm_state: dict):
    return dict(
        mean=torch.reshape(rssm_state["mean"], [-1, rssm_state["mean"].shape[-1]]),
        std=torch.reshape(rssm_state["std"], [-1, rssm_state["std"].shape[-1]]),
        stoch=torch.reshape(rssm_state["stoch"], [-1, rssm_state["stoch"].shape[-1]]),
        deter=torch.reshape(rssm_state["deter"], [-1, rssm_state["deter"].shape[-1]]),
    )


def detach_state(rssm_state: dict):
    return dict(
        mean=rssm_state["mean"].detach(),
        std=rssm_state["std"].detach(),
        stoch=rssm_state["stoch"].detach(),
        deter=rssm_state["deter"].detach(),
    )


def expand_state(rssm_state: dict, n: int):
    return dict(
        mean=rssm_state["mean"].expand(n, *rssm_state["mean"].shape),
        std=rssm_state["std"].expand(n, *rssm_state["std"].shape),
        stoch=rssm_state["stoch"].expand(n, *rssm_state["stoch"].shape),
        deter=rssm_state["deter"].expand(n, *rssm_state["deter"].shape),
    )


def get_dist(rssm_state: dict):
    return D.independent.Independent(D.Normal(rssm_state["mean"], rssm_state["std"]), 1)


def get_random_path(args, path_obs_vision, path_obs_proprio, path_act):
    ret = []
    targets = [path_obs_vision, path_obs_proprio, path_act]
    shapes = [3, 1, 1]

    n_paths = args.n_paths
    n_steps = args.n_steps
    for target, shape in zip(targets, shapes):
        if target is None:
            continue

        target = target[0, 0, :]

        if shape == 1:
            shape_str = "c -> b t c"
        else:
            shape_str = "c h w -> b t c h w"
        target = repeat(target, shape_str, t=n_steps, b=n_paths)
        ret.append(target)

    return ret


def get_scaled_rgb_array_inplace(obs: torch.Tensor):
    # Input: Torch array (prefer for uint8, 0~255)
    # Output: Torch array (float32, -0.5~0.5)

    obs = obs.type(torch.float32)

    # 0~255 -> 0~1
    obs /= 255

    # 0~1 -> -0.5~0.5
    obs -= 0.5

    return obs


def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class SymLogTwoHotLoss:
    def __init__(self, conf, num_classes, lower_bound, upper_bound):
        super().__init__()
        self.conf = conf
        self.num_classes = num_classes
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bin_length = (upper_bound - lower_bound) / (num_classes - 1)
        self.bins = torch.linspace(
            lower_bound, upper_bound, num_classes, device=conf.device
        )

    def forward(self, output, target):
        target = symlog(target)
        assert target.min() >= self.lower_bound and target.max() <= self.upper_bound

        index = torch.bucketize(target, self.bins)
        diff = target - self.bins[index - 1]  # -1 to get the lower bound
        weight = diff / self.bin_length
        weight = torch.clamp(weight, 0, 1)
        weight = weight.unsqueeze(-1)

        target_prob = (1 - weight) * F.one_hot(
            index - 1, self.num_classes
        ) + weight * F.one_hot(index, self.num_classes)

        loss = -target_prob * F.log_softmax(output, dim=-1)
        loss = loss.sum(dim=-1)
        return loss

    def decode(self, output):
        return symexp(F.softmax(output, dim=-1) @ self.bins)


def mse_loss_func(x, y):
    loss = (y - x) ** 2
    loss = reduce(loss, "B L ... -> B L", "sum")
    loss = loss.mean()
    return loss


class TanhBijector(D.Transform):
    def __init__(self):
        super().__init__()
        self.domain = constraints.real
        self.codomain = constraints.interval(-1.0, 1.0)
        self.bijective = True

    @property
    def sign(self):
        return 1.0

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )

        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (np.log(2) - x - F.softplus(-2.0 * x))


def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


class SampleDist:
    # From contrastive-aif
    def __init__(self, dist: D.Distribution, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = (
            torch.argmax(logprob, dim=0)
            .reshape(1, batch_size, 1)
            .expand(1, batch_size, feature_size)
        )
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()


def log_metrics(
    writer, is_eval, tot_episodes, current_step, cur_return, cur_free_energy
):
    """Log return and free_energy to tensorboard and CSV-related step."""
    eval_str = "11_metric_eval" if is_eval else "91_metric_train"
    s = f"{eval_str}/return"
    writer.add_scalar(s, cur_return, global_step=current_step)
    s = f"{eval_str}/return_over_episodes"
    writer.add_scalar(s, cur_return, global_step=tot_episodes)
    s = f"{eval_str}/free_energy"
    writer.add_scalar(s, cur_free_energy, global_step=current_step)


def log_video(writer, np_rng, tag, step_info, img, max_render_steps=100):
    """Log video tensor to writer. img: (batch, timesteps, channel, height, width)."""
    img = img[:, :max_render_steps, :, :, :]
    if type(img) != np.ndarray:
        img = img.numpy()
    img = img + np_rng.random(img.shape) * 0.01
    img = np.clip(img, 0, 1.0)
    draw_time = True
    if draw_time:
        episode_len = img.shape[1]
        img_width = img.shape[4]
        img_height = img.shape[3]
        draw_len = min(episode_len, img_width * img_height)
        for draw_time_i in range(draw_len):
            row = draw_time_i // img_width
            col = draw_time_i % img_width
            if row < img_height:
                img[:, draw_time_i, :, row, col] = 1.0
    if img.shape[2] == 1:
        img = rearrange(img, "b t c h w -> b (t c) h w")
        img = repeat(img, "b t h w -> b t c h w", c=3)
    fps = 5 if "self_prior" in tag else 15
    writer.add_video(tag, img, global_step=step_info, fps=fps)
