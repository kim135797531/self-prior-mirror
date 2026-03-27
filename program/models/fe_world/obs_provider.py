import numpy as np
import torch
import torch.nn as nn

from program.models.configs.model_config import ModelConfig
from program.models.fe_world.obs_cnn import ObservationEncoderCNN, ObservationDecoderCNN
from program.models.fe_world.obs_linear import (
    ObservationEncoderLinear,
    ObservationDecoderLinear,
)


class ObservationProvider(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.world_stoch_size = config.world_stoch_size
        self.world_class_size = config.world_class_size

        self._obs_vision_encoder = ObservationEncoderCNN(
            config=config,
            depth=config.cnn_channel_depth,
            stride=2,
            padding=0,
            shape=(3, 64, 64),
            activation=nn.ELU,
        )
        self._obs_vision_decoder = ObservationDecoderCNN(
            depth=config.cnn_channel_depth,
            stride=2,
            padding=0,
            shape=(3, 64, 64),
            activation=nn.ELU,
            feature_size=self.world_stoch_size * self.world_class_size,
        )

        self.obs_cnn_embed_size = self._obs_vision_encoder.cnn_embed_size
        self.obs_proprio_embed_size = self.obs_cnn_embed_size

        self.obs_embed_size = self.obs_cnn_embed_size
        config.obs_cnn_embed_size = self.obs_cnn_embed_size
        config.obs_proprio_embed_size = self.obs_proprio_embed_size
        config.obs_embed_size = self.obs_embed_size

        self._obs_proprio_encoder = ObservationEncoderLinear(config)
        self._obs_proprio_decoder = ObservationDecoderLinear(config)

        self._obs_mixing = nn.Sequential(
            nn.Linear(
                config.obs_cnn_embed_size + config.obs_proprio_embed_size,
                config.obs_embed_size,
            ),
            nn.LayerNorm(config.obs_embed_size),
            nn.ELU(),
        )

    def forward(self, obs_vision, obs_proprio):
        obs_embed_proprio = self._obs_proprio_encoder(obs_proprio)
        obs_embed_vision = self._obs_vision_encoder(obs_vision)

        obs_embed = torch.cat([obs_embed_proprio, obs_embed_vision], dim=-1)
        obs_embed = self._obs_mixing(obs_embed)

        return obs_embed

    def decode_from_feature(self, feature):
        obs_vision = self._obs_vision_decoder(feature, is_feature=True)
        obs_proprio = self._obs_proprio_decoder(feature, is_feature=True)

        return obs_vision, obs_proprio

    def decode_from_embed(self, embed):
        obs_vision = self._obs_vision_decoder(embed, is_feature=False)
        obs_proprio = self._obs_proprio_decoder(embed, is_feature=False)

        return obs_vision, obs_proprio
