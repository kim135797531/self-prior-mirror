#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""world_model.py
Created by Dongmin Kim at 24. 8. 9.

This module does stuff.
"""
import torch
import torch.nn as nn
from torch.distributions import OneHotCategoricalStraightThrough, OneHotCategorical

from einops import rearrange
from program.models.configs.model_config import ModelConfig
from program.models.fe_world.fe_world_transformer import FEWorldTransformer


class FEWorld(nn.Module):
    def __init__(self, conf: ModelConfig):
        super().__init__()
        self.conf = conf
        self.world_stoch_size = conf.world_stoch_size
        self.world_class_size = conf.world_class_size
        self.obs_embed_size = conf.obs_embed_size

        self.prior = nn.Linear(
            conf.world_hidden_size, self.world_stoch_size * self.world_class_size
        )
        self.posterior = nn.Linear(
            self.obs_embed_size, self.world_stoch_size * self.world_class_size
        )
        self.transformer = FEWorldTransformer(conf)

    def unimix(self, logits):
        mixing_ratio = 0.01
        probs = torch.softmax(logits, dim=-1)
        mixing_probs = (
            mixing_ratio * torch.ones_like(probs) / self.world_class_size
            + (1 - mixing_ratio) * probs
        )
        logits = torch.log(mixing_probs)
        return logits

    def _make_logit_sample(self, logits):
        logits = rearrange(logits, "... (K C) -> ... K C", K=self.world_stoch_size)
        logits = self.unimix(logits)
        dist = OneHotCategoricalStraightThrough(logits=logits)
        samples = dist.rsample()
        samples = rearrange(samples, "... K C -> ... (K C)")
        return logits, samples

    def prior_resample(self, state):
        return self._make_logit_sample(self.prior(state))

    def posterior_resample(self, obs_embed):
        return self._make_logit_sample(self.posterior(obs_embed))
