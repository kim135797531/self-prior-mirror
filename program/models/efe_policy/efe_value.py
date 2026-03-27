from torch import nn as nn
from program.models.configs.model_config import ModelConfig


class EFEValue(nn.Module):
    def __init__(
        self,
        conf: ModelConfig,
    ):
        super().__init__()

        self.layers = conf.policy_layers
        self.feature_size = (
            conf.world_stoch_size * conf.world_class_size + conf.world_hidden_size
        )
        self.hidden_size = conf.value_hidden_size
        self.activation = nn.ELU

        # Output for discrete symLog scaled reward estimation
        self._model = nn.Sequential(
            nn.Linear(self.feature_size, self.hidden_size, bias=False),
            nn.LayerNorm(self.hidden_size),
            self.activation(inplace=True),
            *[
                nn.Linear(self.hidden_size, self.hidden_size, bias=False),
                nn.LayerNorm(self.hidden_size),
                self.activation(inplace=True),
            ]
            * (self.layers - 1),
            nn.Linear(self.hidden_size, conf.reward_symlog_classes),
        )

    def forward(self, features):
        # CAIF built a distribution and computed log_prob.
        # STORM uses symlog-based discrete loss, so it returns logits directly.
        return self._model(features)
