import torch
from torch import nn as nn, distributions as D
from torch.nn import functional as F
from einops import rearrange, repeat

from program.models.configs.model_config import ModelConfig


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, feat_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        assert self.head_dim * num_heads == feat_dim

        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.out_proj = nn.Linear(feat_dim, feat_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.attention = ScaledDotProductAttention(temperature=self.head_dim**0.5)

        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6)

    def forward(self, q_in, k_in, v_in, mask=None):
        B, Lq, C = q_in.shape
        _, Lk, _ = k_in.shape
        H = self.num_heads
        D = self.head_dim

        residual = q_in

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.q_proj(q_in).view(B, Lq, H, D)
        k = self.k_proj(k_in).view(B, Lk, H, D)
        v = self.v_proj(v_in).view(B, Lk, H, D)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(B, Lq, -1)
        q = self.dropout(self.out_proj(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionWiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class AttentionBlockKVCache(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            feat_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.pos_ffn = PositionWiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

    def forward(self, q, k, v, slf_attn_mask=None):
        output, attn = self.slf_attn(q, k, v, mask=slf_attn_mask)
        output = self.pos_ffn(output)
        return output, attn


class PositionalEncoding1D(nn.Module):
    def __init__(self, max_length: int, embed_dim: int):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim

        self.pos_emb = nn.Embedding(self.max_length, embed_dim)

    def forward(self, feat):
        batch_size, batch_length = feat.shape[:2]

        pos_emb = self.pos_emb(torch.arange(self.max_length, device=feat.device))
        pos_emb = repeat(pos_emb, "L D -> B L D", B=batch_size)

        feat = feat + pos_emb[:, :batch_length, :]
        return feat

    def forward_with_position(self, feat, position):
        batch_size, batch_length = feat.shape[:2]
        assert batch_length == 1

        pos_emb = self.pos_emb(torch.arange(self.max_length, device=feat.device))
        pos_emb = repeat(pos_emb, "L D -> B L D", B=batch_size)

        feat = feat + pos_emb[:, position : position + 1, :]
        return feat


class FEWorldTransformer(nn.Module):
    def __init__(self, conf: ModelConfig):
        super().__init__()
        self.conf = conf

        self.world_stoch_size = conf.world_stoch_size
        self.world_class_size = conf.world_class_size
        self.world_hidden_size = conf.world_hidden_size
        self.action_size = conf.action_size
        self.max_steps = conf.transformer_max_steps
        self._activation = nn.ELU

        self.mix = nn.Sequential(
            nn.Linear(
                self.world_stoch_size * self.world_class_size + self.action_size,
                self.world_hidden_size,
                bias=False,
            ),
            nn.LayerNorm(self.world_hidden_size),
            self._activation(inplace=True),
            nn.Linear(self.world_hidden_size, self.world_hidden_size, bias=False),
            nn.LayerNorm(self.world_hidden_size),
        )
        self.pos_enc = PositionalEncoding1D(
            max_length=self.max_steps, embed_dim=self.world_hidden_size
        )
        self.pre_norm = nn.LayerNorm(self.world_hidden_size, eps=1e-6)

        self._attn_layers = nn.ModuleList(
            [
                AttentionBlockKVCache(
                    feat_dim=self.world_hidden_size,
                    hidden_dim=self.world_hidden_size * 2,
                    num_heads=conf.transformer_n_heads,
                    dropout=conf.transformer_dropout,
                )
                for _ in range(conf.transformer_n_layers)
            ]
        )
        self.kv_cache = []

    def forward(self, samples, action, mask):
        x = self.mix(torch.cat([samples, action], dim=-1))
        x = self.pos_enc(x)
        x = self.pre_norm(x)
        for layer in self._attn_layers:
            x, _ = layer(x, x, x, mask)
        return x

    def reset_kv_cache_list(self, batch_size):
        device = self.mix[0].weight.device
        self.kv_cache.clear()
        for _ in range(len(self._attn_layers)):
            self.kv_cache.append(
                torch.zeros(
                    size=(batch_size, 0, self.world_hidden_size),
                    dtype=torch.float32,
                    device=device,
                )
            )

    def forward_with_kv_cache(self, samples, action):
        # d-kim: (from STORM)
        # Cache key/value tensors for faster transformer decoding.
        # Feed one query token at a time instead of the full sequence and store generated key/value states.
        # This enables autoregressive prediction while reusing previous computations.
        B, L = samples.shape[:2]
        assert L == 1

        # As cache grows, the mask region filled with ones must also grow.
        pos = self.kv_cache[0].shape[1]
        mask = torch.ones((1, 1, pos + 1), device=samples.device).bool()

        x = self.mix(torch.cat([samples, action], dim=-1))
        # Use only position-aware token embeddings.
        x = self.pos_enc.forward_with_position(x, position=pos)
        x = self.pre_norm(x)

        # Update features while caching key/value for each attention layer.
        for i, layer in enumerate(self._attn_layers):
            self.kv_cache[i] = torch.cat([self.kv_cache[i], x], dim=1)
            x, _ = layer(x, self.kv_cache[i], self.kv_cache[i], mask)

        return x
