import numpy as np
import torch.distributions as D
import torch.nn as nn
from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange


class ObservationEncoderCNN(nn.Module):
    """
    <encoder>
    depth = 32, stride = 2
    3, 64, 64 = 12288
    32, 31, 31 = 30752 (kernel=4, stride=2)
    64, 14, 14 = 12544 (kernel=4, stride=2)
    128, 6, 6 = 4608 (kernel=4, stride=2)
    256, 2, 2 = 1024 (kernel=4, stride=2)
    cnn_embed_size = 256*2*2 = 1024
    """

    def __init__(self, config, depth, stride, padding, shape, activation):
        super().__init__()
        self.config = config
        self.depth = depth
        self.stride = stride
        self.padding = padding
        self.shape = shape

        self.cnn_embed_size = self._calc_cnn_embed_size()
        self.world_stoch_size = config.world_stoch_size
        self.world_class_size = config.world_class_size
        assert self.cnn_embed_size % self.world_stoch_size == 0
        assert self.cnn_embed_size % self.world_class_size == 0

        # d-kim: 250703: Originally cnn/proprio embeddings were added directly.
        # ELU outputs then ranged from -1 to infinity.
        # Later this embedding is used by a flow model; zuko NSF assumes roughly -5..5 inputs.
        # Therefore tanh normalization to -1..1 is used.

        # d-kim: 250708: Tested both full 1024-d linear and last-32-only linear variants.
        # Surprisingly, layernorm over full 1024 from CNN performed worse.
        # Normalizing only the last dimension improved performance much faster.
        # Possible reason: 1024 -> 32x32 maps to logits for 32 categorical variables,
        # so normalizing only the final logit dimension may be more appropriate.
        self.convolutions = nn.Sequential(
            nn.Conv2d(shape[0], 1 * depth, 4, stride, padding=padding),
            activation(inplace=True),
            nn.Conv2d(1 * depth, 2 * depth, 4, stride, padding=padding),
            activation(inplace=True),
            nn.Conv2d(2 * depth, 4 * depth, 4, stride, padding=padding),
            activation(inplace=True),
            nn.Conv2d(4 * depth, 8 * depth, 4, stride, padding=padding),
            activation(inplace=True),
            # 64x4x4 -> 1024
            Rearrange("... c h w -> ... (c h w)"),
            # d-kim: Normalize start
            # 1024 -> 32x32
            Rearrange(
                "... (x y) -> ... x y", x=self.world_stoch_size, y=self.world_class_size
            ),
            nn.Linear(self.world_class_size, self.world_class_size),
            nn.LayerNorm(self.world_class_size),
            nn.ELU(),
            # 32x32 -> 1024
            Rearrange(
                "... x y -> ... (x y)", x=self.world_stoch_size, y=self.world_class_size
            ),
            # d-kim: Normalize end
        )

    def forward(self, obs):
        # d-kim: Results differ between einops.pack and per-sample execution.
        # This happens because torch.backends.cudnn.allow_tf32 = True.
        # TF32 trades precision for speed.
        # Reference: https://docs.pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
        x, packed_shape = pack([obs], "* c h w")
        x = self.convolutions(x)
        [embed] = unpack(x, packed_shape, "* dim")
        return embed

    def _calc_cnn_embed_size(self):
        conv1_shape = conv_out_shape(
            self.shape[1:], padding=self.padding, kernel_size=4, stride=self.stride
        )
        conv2_shape = conv_out_shape(
            conv1_shape, padding=self.padding, kernel_size=4, stride=self.stride
        )
        conv3_shape = conv_out_shape(
            conv2_shape, padding=self.padding, kernel_size=4, stride=self.stride
        )
        conv4_shape = conv_out_shape(
            conv3_shape, padding=self.padding, kernel_size=4, stride=self.stride
        )
        embed_size = 8 * self.depth * np.prod(conv4_shape).item()
        return embed_size


class ObservationDecoderCNN(nn.Module):
    """
    <original decoder>
    depth = 32, stride = 2
    1024, 1, 1 (from linear 230 -> 1024)
    128, 6, 6
    64, 16, 16
    32, 35, 35
    3, 64, 64

    <arm_env decoder>
    depth = 32, stride = 2
    1024, 1, 4 (from linear 230 -> 4096=1024*1*4), conv4_shape
    128, 2, 8, conv3_shape
    64, 5, 17, conv2_shape
    32, 13, 38, conv1_shape
    1, 30, 80

    <arm_env small decoder>
    depth = 4, stride = 2
    16, 1, 4 (from linear 230 -> 64=16*1*4), conv4_shape
    16, 2, 8, conv3_shape
    8, 6, 18, conv2_shape
    4, 14, 39, conv1_shape
    1, 30, 80
    """

    def __init__(self, depth, stride, padding, shape, activation, feature_size):
        super().__init__()
        self.depth = depth
        self.shape = shape

        c, h, w = shape
        conv1_kernel_size = 6
        conv2_kernel_size = 6
        conv3_kernel_size = 5
        conv4_kernel_size = 5
        conv1_shape = conv_out_shape((h, w), padding, conv1_kernel_size, stride)
        conv1_pad = output_padding_shape(
            (h, w), conv1_shape, padding, conv1_kernel_size, stride
        )
        conv2_shape = conv_out_shape(conv1_shape, padding, conv2_kernel_size, stride)
        conv2_pad = output_padding_shape(
            conv1_shape, conv2_shape, padding, conv2_kernel_size, stride
        )
        conv3_shape = conv_out_shape(conv2_shape, padding, conv3_kernel_size, stride)
        conv3_pad = output_padding_shape(
            conv2_shape, conv3_shape, padding, conv3_kernel_size, stride
        )
        conv4_shape = conv_out_shape(conv3_shape, padding, conv4_kernel_size, stride)
        conv4_pad = output_padding_shape(
            conv3_shape, conv4_shape, padding, conv4_kernel_size, stride
        )
        self.conv_shape = (32 * depth, *conv4_shape)
        # d-kim: 250703: Since downstream assumes tanh-bounded embeddings, activation was changed from ELU to tanh.
        self.feat2embed = nn.Sequential(
            nn.Linear(feature_size, 32 * depth * np.prod(conv4_shape).item()),
            nn.LayerNorm(32 * depth * np.prod(conv4_shape).item()),
            nn.ELU(),
            Rearrange("... (c h w) -> ... c h w", c=32 * depth, h=conv4_shape[0]),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                32 * depth,
                4 * depth,
                conv4_kernel_size,
                stride,
                padding=padding,
                output_padding=conv4_pad,
            ),
            activation(inplace=True),
            nn.ConvTranspose2d(
                4 * depth,
                2 * depth,
                conv3_kernel_size,
                stride,
                padding=padding,
                output_padding=conv3_pad,
            ),
            activation(inplace=True),
            nn.ConvTranspose2d(
                2 * depth,
                1 * depth,
                conv2_kernel_size,
                stride,
                padding=padding,
                output_padding=conv2_pad,
            ),
            activation(inplace=True),
            nn.ConvTranspose2d(
                1 * depth,
                shape[0],
                conv1_kernel_size,
                stride,
                padding=padding,
                output_padding=conv1_pad,
            ),
        )

    def forward(self, x, is_feature):
        x, packed_shape = pack([x], "* dim")

        if is_feature:
            x = self.feat2embed(x)
        else:
            c, h, w = self.conv_shape
            x = rearrange(x, "... (c h w) -> ... c h w", c=c, h=h, w=w)

        x = self.decoder(x)
        [mean] = unpack(x, packed_shape, "* c h w")
        return mean


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2.0 * padding - (kernel_size - 1.0) - 1.0) / stride + 1.0)


def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1


def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)


def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(
        output_padding(h_in[i], conv_out[i], padding, kernel_size, stride)
        for i in range(len(h_in))
    )
