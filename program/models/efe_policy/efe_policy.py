import torch
from torch import nn as nn, distributions as D
from torch.nn import functional as F
import numpy as np

from program.models.configs.model_config import ModelConfig
from program.utils import TanhBijector, SampleDist


class EFEPolicy(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.config = config
        self.action_size = config.action_size
        self.world_stoch_size = config.world_stoch_size
        self.world_class_size = config.world_class_size
        self.prior_size = config.world_stoch_size * config.world_class_size
        self.feature_size = self.prior_size + config.world_hidden_size
        self.hidden_size = config.policy_hidden_size
        self.layers = config.policy_layers
        self.activation = nn.ELU  # From CAIF (STORM = ReLU, DreamerV3 = SiLU)
        self.min_std = 1e-4  # CAIF = 1e-4, DreamerV3 = 0.1
        self.max_std = 1.0  # From DreamerV3
        self.init_std = 0.5  # CAIF = 5.0, DreamerV3 = 2.0
        self.raw_init_std = np.log(np.exp(self.init_std) - 1)
        self.layer_norm_eps = 1e-3  # From DreamerV3
        self.mean_scale = 1.0  # CAIF = 5.0, DreamerV3 = 1.0

        self.policy = nn.Sequential(
            nn.Linear(self.feature_size, self.hidden_size, bias=False),
            nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps),
            self.activation(inplace=True),
            *[
                nn.Linear(self.hidden_size, self.hidden_size, bias=False),
                nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps),
                self.activation(inplace=True),
            ]
            * (self.layers - 1),
            nn.Linear(self.hidden_size, self.action_size * 2)
        )

    def forward(self, state_features):
        raw_mean, raw_std = torch.chunk(self.policy(state_features), 2, -1)
        mean = self.mean_scale * torch.tanh(raw_mean / self.mean_scale)
        std = F.softplus(raw_std + self.raw_init_std) + self.min_std

        dist = D.Normal(loc=mean, scale=std)
        dist = D.TransformedDistribution(dist, TanhBijector())
        dist = D.Independent(dist, 1)
        dist = SampleDist(dist)

        return dist
