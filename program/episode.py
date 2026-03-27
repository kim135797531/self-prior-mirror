#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""episode.py
Created by Dongmin Kim at 24. 8. 9.

This module does stuff.
"""
from collections import deque
import numpy as np
import torch
import itertools

from program.models.configs.model_config import ModelConfig


class EpisodeStore:
    def __init__(self, conf: ModelConfig, np_rng: np.random.Generator):
        self.conf = conf
        self.np_rng = np_rng

        self.max_episode_stores = conf.max_episode_stores
        self.max_steps = conf.max_episode_stores * conf.timelimit

        # policy_used, caregiver_used
        self.n_categories = 2

        self.episodes_by_group = dict()

        self.group_keys = list(
            itertools.product([False, True], repeat=self.n_categories)
        )
        n_groups = len(self.group_keys)

        for policy_used, caregiver_used in self.group_keys:
            key = (policy_used, caregiver_used)
            self.episodes_by_group[key] = deque(
                maxlen=self.max_episode_stores // n_groups
            )
            # self.obs_visions_by_group[key] = deque(maxlen=self.max_steps // n_groups)
            # self.obs_proprios_by_group[key] = deque(maxlen=self.max_steps // n_groups)
            # self.acts_by_group[key] = deque(maxlen=self.max_steps // n_groups)
            # self.dones_by_group[key] = deque(maxlen=self.max_steps // n_groups)

        self.n_episodes = 0
        # self.n_steps = 0

    def get_key(self, policy_used, sticker_used):
        return policy_used, sticker_used

    def add_episode_dict(self, episode_dict: dict):
        policy_used = episode_dict["policy_used"]
        sticker_used = episode_dict["sticker_used"]
        key = (policy_used, sticker_used)

        # Add episodes
        obs_vision = np.concatenate(episode_dict["obs_vision"], axis=1)
        obs_proprio = np.concatenate(episode_dict["obs_proprio"], axis=1)
        act = np.concatenate(episode_dict["act"], axis=1)
        done = np.array(episode_dict["done"])

        episode = Episode(
            obs_vision,
            obs_proprio,
            act,
            done,
            policy_used,
            sticker_used,
        )
        self.episodes_by_group[key].append(episode)
        self.n_episodes = sum(len(deque) for deque in self.episodes_by_group.values())

        # Final check
        # assert self.n_steps == self.n_episodes * self.conf.timelimit

    def state_dict(self):
        """Serialize for checkpoint. Keys are tuple (policy_used, sticker_used)."""
        return {
            "episodes_by_group": {
                key: [ep.state_dict() for ep in list(deq)]
                for key, deq in self.episodes_by_group.items()
            },
        }

    def load_state_dict(self, state: dict, np_rng: np.random.Generator):
        """Restore from checkpoint. np_rng is the runner's generator (not saved)."""
        self.np_rng = np_rng
        for key, ep_list in state["episodes_by_group"].items():
            deq = self.episodes_by_group[key]
            deq.clear()
            for ep_state in ep_list:
                deq.append(Episode.from_state_dict(ep_state))
        self.n_episodes = sum(len(deq) for deq in self.episodes_by_group.values())

    @torch.no_grad()
    def sample_paths(
        self, n_paths, path_length, device, group_keys=None, group_probs=None
    ):
        # group_keys example:
        # All: [(False, False), (False, True), (True, False), (True, True)]
        # Random only (policy off): [(False, False), (False, True)]
        # Hand only (caregiver off): [(False, False), (True, False)]
        #
        # group_probs example:
        # None: all probs are equal
        # [0.1, 0.2, 0.3, 0.4]: probs for each group
        if group_keys is None:
            group_keys = self.group_keys
        n_groups = len(group_keys)

        if group_probs is None:
            n_paths_per_group = n_paths // n_groups
            remainder = n_paths % n_groups
            n_paths_per_group = [n_paths_per_group] * n_groups
        else:
            assert len(group_probs) == n_groups
            n_paths_per_group = [int(n_paths * prob) for prob in group_probs]
            remainder = n_paths - sum(n_paths_per_group)

        for i in range(remainder):
            n_paths_per_group[i] += 1

        for i, n_paths_to_sample in enumerate(n_paths_per_group):
            if n_paths_to_sample == 0:
                print(
                    f"Warning: No paths available for group {group_keys[i]}. "
                    f"Check int(n_paths * 0.05) > 0"
                )

        path_obs_vision_list = []
        path_obs_proprio_list = []
        path_act_list = []
        path_policy_used_list = []
        path_sticker_used_list = []

        remain_from_prev_group = 0
        for n_paths_to_sample, key in zip(n_paths_per_group, group_keys):
            # If there are remaining paths from the previous group, add them to this group
            n_paths_to_sample = n_paths_to_sample + remain_from_prev_group
            remain_from_prev_group = 0

            if n_paths_to_sample == 0:
                continue

            # Get all episodes in this group
            episodes = self.episodes_by_group[key]
            if len(episodes) == 0:
                remain_from_prev_group = n_paths_to_sample
                print(
                    f"Warning: No episodes available in group {key}. "
                    f"Skipping sampling for this group."
                )
                continue

            # Collect all valid sample positions across all episodes in this group
            all_valid_positions = []  # List of (episode_index, start_index) tuples

            for ei, episode in enumerate(episodes):
                episode_length = episode.act.shape[1]
                max_start_index = episode_length - path_length

                if max_start_index > 0:
                    # Add all valid starting positions in this episode
                    for start_idx in range(max_start_index + 1):
                        all_valid_positions.append((ei, start_idx))

            # If no valid positions found in this group, continue to next group
            if len(all_valid_positions) == 0:
                remain_from_prev_group = n_paths_to_sample
                print(
                    f"Warning: No valid positions available in group {key}. "
                    f"Skipping sampling for this group."
                )
                continue

            # Randomly select n_paths_to_sample positions from all valid positions
            # If we have fewer valid positions than requested, use all available
            n_positions_to_sample = min(n_paths_to_sample, len(all_valid_positions))
            if n_positions_to_sample != n_paths_to_sample:
                print(
                    f"Warning: Requested {n_paths_to_sample} paths, "
                    f"but only {len(all_valid_positions)} valid positions available in group {key}. "
                    f"Sampling {n_positions_to_sample} paths."
                )
                diff = n_paths_to_sample - n_positions_to_sample
                remain_from_prev_group = diff if diff > 0 else 0

            # Randomly select positions without replacement
            selected_position_indices = self.np_rng.choice(
                len(all_valid_positions), size=n_positions_to_sample, replace=False
            )

            # Extract data for each selected position
            for pos_idx in selected_position_indices:
                ei, start_idx = all_valid_positions[pos_idx]
                episode = episodes[ei]

                path_obs_vision_list.append(
                    episode.obs_vision[:, start_idx : start_idx + path_length].to(
                        device, non_blocking=True
                    )
                )
                path_obs_proprio_list.append(
                    episode.obs_proprio[:, start_idx : start_idx + path_length].to(
                        device, non_blocking=True
                    )
                )
                path_act_list.append(
                    episode.act[:, start_idx : start_idx + path_length].to(
                        device, non_blocking=True
                    )
                )
                path_policy_used_list.append(episode.policy_used)
                path_sticker_used_list.append(episode.sticker_used)

        path_obs_vision = torch.cat(path_obs_vision_list, dim=0)
        path_obs_proprio = torch.cat(path_obs_proprio_list, dim=0)
        path_act = torch.cat(path_act_list, dim=0)

        if remain_from_prev_group != 0:
            print(
                f"Warning: {remain_from_prev_group} paths could not be sampled due to insufficient valid positions."
            )

        # print(
        #     f"{sum(path_policy_used_list)}/{len(path_policy_used_list)}, {sum(path_sticker_used_list)}/{len(path_sticker_used_list)}"
        # )

        return (path_obs_vision, path_obs_proprio, path_act)


class Episode:
    def __init__(
        self,
        obs_vision,
        obs_proprio,
        act,
        done,
        policy_used,
        sticker_used,
    ):
        self._obs_visions = torch.Tensor(obs_vision).pin_memory()
        self._obs_proprios = torch.Tensor(obs_proprio).pin_memory()
        self._actions = torch.Tensor(act).pin_memory()
        self._done = torch.Tensor(done).pin_memory()
        self._policy_used = policy_used
        self._sticker_used = sticker_used

    def state_dict(self):
        return {
            "obs_vision": self._obs_visions.cpu().numpy(),
            "obs_proprio": self._obs_proprios.cpu().numpy(),
            "act": self._actions.cpu().numpy(),
            "done": self._done.cpu().numpy(),
            "policy_used": self._policy_used,
            "sticker_used": self._sticker_used,
        }

    @classmethod
    def from_state_dict(cls, d: dict):
        return cls(
            d["obs_vision"],
            d["obs_proprio"],
            d["act"],
            d["done"],
            d["policy_used"],
            d["sticker_used"],
        )

    def __len__(self):
        return len(self._actions)

    @property
    def obs_vision(self):
        return self._obs_visions

    @property
    def obs_proprio(self):
        return self._obs_proprios

    @property
    def act(self):
        return self._actions

    @property
    def done(self):
        return self._done

    @property
    def policy_used(self):
        return self._policy_used

    @property
    def sticker_used(self):
        return self._sticker_used

    @property
    def n_steps(self):
        return len(self._actions)
