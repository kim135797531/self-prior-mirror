from einops.layers.torch import Rearrange
from torch import nn as nn, distributions as D

from program.models.configs.model_config import ModelConfig


class ObservationEncoderLinear(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.obs_proprio_size = config.proprio_size
        self.embed_size = config.obs_embed_size

        self.model = nn.Sequential(
            nn.Linear(self.obs_proprio_size, 32),
            nn.ELU(inplace=True),
            nn.Linear(32, self.embed_size),
            nn.LayerNorm(self.embed_size),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class ObservationDecoderLinear(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.obs_proprio_size = config.proprio_size
        self.shape = (self.obs_proprio_size,)
        self.world_stoch_size = config.world_stoch_size
        self.world_class_size = config.world_class_size
        self.obs_embed_size = config.obs_embed_size
        # d-kim: 250703: Since downstream assumes tanh-bounded embeddings, activation was changed from ELU to tanh.
        self.feat2embed = nn.Sequential(
            nn.Linear(
                self.world_stoch_size * self.world_class_size, self.obs_embed_size
            ),
            nn.LayerNorm(self.obs_embed_size),
            nn.ELU(),
        )

        self.model = nn.Sequential(
            nn.Linear(self.obs_embed_size, 32),
            nn.ELU(inplace=True),
            nn.Linear(32, self.obs_proprio_size),
        )

    def forward(self, x, is_feature):
        if is_feature:
            x = self.feat2embed(x)
        mean = self.model(x)
        return mean
