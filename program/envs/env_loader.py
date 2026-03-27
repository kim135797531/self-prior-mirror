#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""env_util.py
Created by Dongmin Kim at 24. 8. 9.

This module does stuff.
"""
from copy import deepcopy
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from program.models.configs.model_config import ModelConfig


def make_env(conf: ModelConfig, vector_id: int) -> gym.Env:
    suite = conf.suite
    action_repeat = conf.action_repeat
    timelimit = conf.timelimit
    detach_step_threshold = conf.detach_step_threshold
    sticker_detach_distance_threshold = conf.sticker_detach_distance_threshold

    if suite == "robot_mirror":
        env_id = "ArmMirrorEnv-v0"
        entry_point = "program.envs.env_arm_mirror:ArmMirrorEnv"
        model_path = "./program/envs/env_arm_mirror.xml"
    else:
        raise NotImplementedError(suite)

    register(id=env_id, entry_point=entry_point)
    env = gym.make(
        id=env_id,
        model_path=model_path,
        detach_step_threshold=detach_step_threshold,
        sticker_detach_distance_threshold=sticker_detach_distance_threshold,
        render_mode="rgb_array",
    )

    if action_repeat is not None and action_repeat > 1:
        env = ActionRepeat(env, action_repeat)

        if detach_step_threshold is not None:
            if hasattr(env.unwrapped, "model"):
                timestep = env.unwrapped.model.opt.timestep
            else:
                timestep = 1
            print(f"Env timestep is {timestep}, and action repeat is {action_repeat}.")
            print(
                f"So each step for model is {timestep}*{action_repeat} = {timestep * action_repeat} seconds."
            )

    if timelimit is not None:
        env = gym.wrappers.TimeLimit(env, timelimit)

    env = GiveNoiseToVision(env)
    env = NormalizeProprioception(env)
    env = NormalizeActions(env)
    env = MergeDoneTruncated(env)
    env = PerEnvOptions(env, vector_id)

    return env


class GiveNoiseToVision(gym.ObservationWrapper):
    def __init__(self, env, obs_key="vision", noise=0.01):
        # noise: 0.0 ~ 1.0
        super().__init__(env)
        self._obs_key = obs_key
        self._noise = int(noise * 256)

    def observation(self, observation):
        obs = observation[self._obs_key]
        shape = obs.shape

        # Prevent overflow
        obs = obs.astype(np.int32)
        obs += self.np_random.integers(
            low=-self._noise, high=self._noise, size=shape, dtype=np.int32
        )
        obs.clip(0, 255, out=obs)

        # Return to unsigned int8
        obs = obs.astype(np.uint8)
        observation[self._obs_key] = obs
        return observation


class NormalizeProprioception(gym.ObservationWrapper):
    def __init__(self, env):
        # Normalize proprioception outputs to the -1 to 1 range.
        super().__init__(env)
        self.observation_space = deepcopy(super().observation_space)
        proprio_space: gym.spaces.Box = self.observation_space["proprioception"]
        self._low = proprio_space.low
        self._high = proprio_space.high
        self.observation_space["proprioception"] = gym.spaces.Box(
            low=-1, high=1, shape=self._low.shape, dtype=np.float32
        )

    def observation(self, obs):
        proprio = obs["proprioception"]
        proprio = (proprio - self._low) / (self._high - self._low)
        proprio = 2 * proprio - 1  # Scale to -1 ~ 1
        obs["proprioception"] = proprio
        return obs

    def reset(self, seed=None, options=None):
        self.observation_space.seed(seed)
        observation, info = super().reset(seed=seed, options=options)
        return observation, info


class NormalizeActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        print(
            "d-kim:"
            "NormalizeActions: Assumes policy outputs are normalized to -1..1.\n"
            "Use tanh or a normal distribution based policy output.\n"
            "Out-of-range values are clipped before sending to the environment.\n"
            "Do not clip actions again in training code.\n"
            ""
        )
        # Normalize only finite action bounds.
        # Assumes neural network actions are normalized to -1..1 (e.g., tanh output).
        action_space: gym.spaces.Box = self.action_space

        # Mask for finite-bounded dimensions.
        self._finite_mask = np.logical_and(
            np.isfinite(action_space.low), np.isfinite(action_space.high)
        )

        # Replace infinite bounds with -1 and 1 (keep finite bounds unchanged).
        self._low = np.where(self._finite_mask, action_space.low, -1)
        self._high = np.where(self._finite_mask, action_space.high, 1)

        # Expose action space as fully normalized -1..1.
        # Finite dimensions are restored later during denormalization.
        low = np.where(self._finite_mask, -np.ones_like(self._low), self._low)
        high = np.where(self._finite_mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def action(self, action):
        # Restore original low/high from normalized -1..1 actions.
        original = (action + 1) / 2
        original = (original * (self._high - self._low)) + self._low
        # Keep infinite dimensions as-is; denormalize only finite ones.
        original = np.where(self._finite_mask, original, action)
        return original

    def reset(self, seed=None, options=None):
        self.action_space.seed(seed)
        observation, info = super().reset(seed=seed, options=options)
        return observation, info


class ActionRepeat(gym.Wrapper):
    def __init__(self, env, repeat):
        super().__init__(env)
        self._repeat = repeat

        # repeat = 0 -> not execute...?? nonsense
        # repeat = 1 -> execute once, is same with original
        # repeat = 2 -> execute twice, so it's meaningful
        assert self._repeat > 1

    def step(self, action):
        self.unwrapped.skip_render = True

        for _ in range(self._repeat - 1):
            observation, reward, terminated, truncated, info = super().step(action)
            if terminated:
                raise ValueError(
                    "d-kim: ActionRepeat cannot be used with environments that have terminated when skip_render is True."
                )

        self.unwrapped.skip_render = False
        observation, reward, terminated, truncated, info = super().step(action)

        return observation, reward, terminated, truncated, info


class MergeDoneTruncated(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        merged_done = terminated or truncated
        return observation, reward, merged_done, merged_done, info


class PerEnvOptions(gym.Wrapper):
    def __init__(self, env, vector_id: int):
        super().__init__(env)
        self.vector_id = vector_id

    def reset(self, seed=None, options=None):
        if isinstance(options, (list, tuple)):
            my_options = options[self.vector_id]
        else:
            my_options = options
        return self.env.reset(seed=seed, options=my_options)
