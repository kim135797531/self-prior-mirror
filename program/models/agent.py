#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""agent.py
Created by Dongmin Kim at 24. 8. 9.

This module does stuff.
"""

import copy
import einops
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce
from torch.distributions import (
    kl_divergence,
    OneHotCategorical,
)

from program.models.efe_policy.efe_policy import EFEPolicy
from program.models.efe_policy.efe_value import EFEValue
from program.models.configs.model_config import ModelConfig
from program.models.fe_world.fe_world import FEWorld
from program.models.fe_world.obs_provider import ObservationProvider
from program.models.self_prior import CategoricalTransformer


class Agent(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self, conf: ModelConfig, np_rng: np.random.Generator):
        super().__init__()
        self.conf = conf
        self.np_rng = np_rng

        # Observation preprocessing
        self.obs_provider = ObservationProvider(conf)

        # Free energy
        self.fe_world = FEWorld(conf)

        # Expected free energy
        self.efe_policy = EFEPolicy(conf)
        self.efe_value = EFEValue(conf)
        self.n_mini_imagine_paths = conf.n_imagine_paths
        self.n_mini_imagine_paths_checked = False

        # Self-prior
        self.self_prior = CategoricalTransformer(conf)

        # Learning parameters (efep)
        # self.ambiguity_rms = RunningMeanStd()
        # self.ambiguity_beta = 1e-3

        # Apply small value for policy heads
        self.efe_policy.policy[-1].weight.data *= 0.01
        self.efe_policy.policy[-1].bias.data.fill_(0.0)

        # Apply 0.0 for value heads
        nn.init.zeros_(self.efe_value._model[-1].weight.data)
        self.efe_value._model[-1].bias.data.fill_(0.0)

        # The initialization below gives equal or worse performance.
        # self.obs_provider._obs_vision_decoder.decoder[-1].apply(
        #     self.uniform_init_weights(1.0)
        # )
        # self.obs_provider._obs_proprio_decoder.model[-1].apply(
        #     self.uniform_init_weights(1.0)
        # )
        # self.fe_world.prior.apply(self.uniform_init_weights(1.0))
        # self.fe_world.posterior.apply(self.uniform_init_weights(1.0))
        # self.self_prior.fc_out.apply(self.uniform_init_weights(0.0))

        self.efe_value_target = copy.deepcopy(self.efe_value)

        if self.conf.use_slow_self_prior:
            self.self_prior_target = copy.deepcopy(self.self_prior)

        if self.conf.compile:
            print("Start compile")
            fullgraph = True
            self.obs_provider: ObservationProvider = torch.compile(
                self.obs_provider, fullgraph=fullgraph
            )
            self.fe_world: FEWorld = torch.compile(self.fe_world, fullgraph=fullgraph)
            self.efe_policy: EFEPolicy = torch.compile(
                self.efe_policy, fullgraph=fullgraph
            )
            self.efe_value: EFEValue = torch.compile(
                self.efe_value, fullgraph=fullgraph
            )
            self.efe_value_target: EFEValue = torch.compile(
                self.efe_value_target, fullgraph=fullgraph
            )
            self.self_prior: CategoricalTransformer = torch.compile(
                self.self_prior, fullgraph=fullgraph
            )
            if self.conf.use_slow_self_prior:
                self.self_prior_target: CategoricalTransformer = torch.compile(
                    self.self_prior_target, fullgraph=fullgraph
                )
            print("Finish compile")

    def imagine(
        self,
        obs_visions: torch.Tensor,
        obs_proprios: torch.Tensor,
        acts: torch.Tensor,
        get_reconstruction: bool = False,
    ):
        self.train()
        device = self.conf.device

        post_buffer = torch.empty(
            (
                self.conf.n_imagine_paths,
                self.conf.n_imagine_steps + 1,
                self.conf.world_stoch_size * self.conf.world_class_size,
            ),
            dtype=torch.float32,
            device=device,
        )
        post_logit_buffer = torch.empty(
            (
                self.conf.n_imagine_paths,
                self.conf.n_imagine_steps + 1,
                self.conf.world_stoch_size,
                self.conf.world_class_size,
            ),
            dtype=torch.float32,
            device=device,
        )
        hidden_buffer = torch.empty(
            (
                self.conf.n_imagine_paths,
                self.conf.n_imagine_steps + 1,
                self.conf.world_hidden_size,
            ),
            dtype=torch.float32,
            device=device,
        )
        action_buffer = torch.empty(
            (
                self.conf.n_imagine_paths,
                self.conf.n_imagine_steps,
                self.conf.action_size,
            ),
            dtype=torch.float32,
            device=device,
        )

        with torch.autocast(
            device_type=self.conf.device,
            dtype=torch.bfloat16,
            enabled=self.conf.use_amp,
        ):
            # recon index
            # ri = int(self.np_rng.integers(n_imagine_paths))

            # recon episode index (now checking sticker)
            ri = (len(obs_visions) // 4) + 1

            imagine_loss_dict = dict()
            recon_dict = dict()
            prior_samples = []
            imagine_prior_samples = []

            n_imagine_paths = self.conf.n_imagine_paths
            n_imagine_context_steps = self.conf.n_imagine_context_steps
            n_imagine_steps = self.conf.n_imagine_steps

            # Initialize
            self.fe_world.transformer.reset_kv_cache_list(n_imagine_paths)

            with torch.no_grad():
                # Run transformer across full sequence to build prior at the final timestep.
                obs_embeds = self.obs_provider(obs_visions, obs_proprios)
                post_logits, post_samples = self.fe_world.posterior_resample(obs_embeds)

                # d-kim: Questions from original STORM implementation:
                # 1) Why feed one token at a time instead of the whole sequence for final prior?
                # -> During training, full-sequence attention captures global correlations.
                # -> During inference, per-token decoding with key/value cache is faster.
                # 2) Why feed from the first step instead of only the last token?
                # -> Cached key/value already contains previous context, so current token still sees full history.
                for i in range(n_imagine_context_steps):
                    post_sample, action = (
                        post_samples[:, i : i + 1],
                        acts[:, i : i + 1],
                    )
                    hidden = self.fe_world.transformer.forward_with_kv_cache(
                        post_sample, action
                    )
                    prior_logit, prior_sample = self.fe_world.prior_resample(hidden)

                    # For reconstruction
                    prior_samples.append(prior_sample[ri, :].detach())

                # Use the final prior as the first posterior of imagined trajectory.
                post_buffer[:, 0:1] = prior_sample
                post_logit_buffer[:, 0:1] = prior_logit
                hidden_buffer[:, 0:1] = hidden

            # Start imagination rollout.
            for i in range(n_imagine_steps):
                post_sample = post_buffer[:, i : i + 1]
                hidden = hidden_buffer[:, i : i + 1]

                # Get action
                policy_feature = torch.cat([post_sample, hidden], dim=-1)
                action_dist = self.efe_policy(policy_feature.detach())
                action = action_dist.rsample()
                action_buffer[:, i : i + 1] = action

                # Get prior
                hidden = self.fe_world.transformer.forward_with_kv_cache(
                    post_sample, action
                )
                prior_logit, prior_sample = self.fe_world.prior_resample(hidden)

                post_buffer[:, i + 1 : i + 2] = prior_sample
                post_logit_buffer[:, i + 1 : i + 2] = prior_logit
                hidden_buffer[:, i + 1 : i + 2] = hidden

                # For reconstruction
                imagine_prior_samples.append(prior_sample[ri, :].detach())

            if self.conf.use_reinforce:
                post = post_buffer[:, 1:]
            else:
                post = post_logit_buffer[:, 1:]

            use_gradient = not self.conf.use_reinforce
            (free_energy_buffer, _) = self.get_expected_free_energy(
                post, use_gradient=use_gradient
            )

            imagine_loss_dict["31_imagine_expected_free_energy"] = (
                free_energy_buffer.detach().mean().item()
            )

            if get_reconstruction:
                with torch.no_grad():
                    prior_samples = torch.cat(prior_samples, dim=0)
                    imagine_prior_samples = torch.cat(imagine_prior_samples, dim=0)

                    recon_next_obs_vision = obs_visions[ri, 1:]
                    recon_next_post_sample = post_samples[ri, 1:]
                    recon_prior_sample = prior_samples[:-1]
                    recon_imagine_prior_sample = imagine_prior_samples

                    recon_next_post_vision, recon_next_post_proprio = (
                        self.obs_provider.decode_from_feature(recon_next_post_sample)
                    )
                    recon_prior_vision, recon_prior_proprio = (
                        self.obs_provider.decode_from_feature(recon_prior_sample)
                    )
                    recon_imagine_prior_vision, recon_imagine_prior_proprio = (
                        self.obs_provider.decode_from_feature(
                            recon_imagine_prior_sample
                        )
                    )

                    recon_next_obs_vision = recon_next_obs_vision.float()
                    recon_next_post_vision = recon_next_post_vision.float()
                    recon_prior_vision = recon_prior_vision.float()
                    recon_imagine_prior_vision = recon_imagine_prior_vision.float()

                    # Change the range
                    recon_next_obs_vision = recon_next_obs_vision + 0.5
                    recon_next_post_vision = recon_next_post_vision + 0.5
                    recon_prior_vision = recon_prior_vision + 0.5
                    recon_imagine_prior_vision = recon_imagine_prior_vision + 0.5

                    truth_post_diff = (
                        recon_next_obs_vision - recon_next_post_vision + 1
                    ) / 2
                    truth_prior_diff = (
                        recon_next_obs_vision - recon_prior_vision + 1
                    ) / 2
                    post_prior_diff = (
                        recon_next_post_vision - recon_prior_vision + 1
                    ) / 2

                    recon_dict["01_truth"] = (
                        recon_next_obs_vision.detach().cpu().numpy()
                    )
                    recon_dict["02_post"] = (
                        recon_next_post_vision.detach().cpu().numpy()
                    )
                    recon_dict["03_prior"] = recon_prior_vision.detach().cpu().numpy()

                    # Simple recon test
                    recon_dict["04_truth_post_diff"] = (
                        truth_post_diff.detach().cpu().numpy()
                    )
                    # World model test
                    recon_dict["05_truth_prior_diff"] = (
                        truth_prior_diff.detach().cpu().numpy()
                    )
                    # World model test 2
                    recon_dict["06_post_prior_diff"] = (
                        post_prior_diff.detach().cpu().numpy()
                    )
                    # Imagine test
                    recon_dict["07_imagine_prior"] = (
                        recon_imagine_prior_vision.detach().cpu().numpy()
                    )

        return (
            torch.cat([post_buffer, hidden_buffer], dim=-1),
            action_buffer,
            free_energy_buffer,
            imagine_loss_dict,
            recon_dict,
        )

    def get_expected_free_energy(self, prior_sample, use_gradient=False):
        # G optimization.
        # In other words, calculate the expected free energy for the future (dataset).
        # Create loss to optimize expected free energy.
        expected_free_energy_dict = dict()

        # If the prior_states is a single data, add the time dimension.
        if use_gradient:
            prior_logit = prior_sample
            assert len(prior_logit.shape) == 4  # B, T(1), 32, 32
        else:
            assert len(prior_sample.shape) == 3  # B, T(1), (32*32)
        # if len(prior_sample.shape) == 2:
        #     prior_sample = rearrange(prior_sample, "... -> 1 ...")

        preferred_obs_n_dim = self.obs_provider.obs_embed_size

        # Use the predicted future prior to calculate the expected free energy.
        batch_size, batch_length, state_dim = prior_sample.shape[:3]
        train_shape = prior_sample.shape[:2]
        train_data_n = np.prod(train_shape)

        # 1. Pragmatic
        # 1.a Pragmatic value E_qo[log p(o)]
        #   If the preferred_obs is assumed to be a distribution, how likely is the predicted_obs?
        bos_x = self.self_prior.prepare(prior_sample, use_gradient=use_gradient)
        logits, targets = self.self_prior.get_logits(bos_x, use_gradient=use_gradient)
        logprob_preferences = self.self_prior.get_logprob(
            logits, targets, use_gradient=use_gradient
        )
        logprob_preferences = rearrange(
            logprob_preferences, "(b t) -> b t", b=batch_size, t=batch_length
        )

        expected_free_energy = -logprob_preferences

        expected_free_energy_dict["31_imagine_expected_free_energy"] = (
            expected_free_energy.detach().mean().item()
        )

        return expected_free_energy, expected_free_energy_dict

    def categorical_kl_div_loss_func(self, p_logits, q_logits, free_bits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = kl_divergence(p_dist, q_dist)
        kl_div = reduce(kl_div, "B L D -> B L", "sum")
        real_kl_div = kl_div
        kl_div = torch.max(kl_div, kl_div.new_full(kl_div.size(), free_bits))
        return kl_div, real_kl_div
