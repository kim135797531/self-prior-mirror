import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from program.models.configs.model_config import ModelConfig


class CategoricalTransformer(nn.Module):
    def __init__(self, conf: ModelConfig):
        super().__init__()
        self.device = conf.device

        # Number of discrete variables (K = 32)
        self.num_vars = conf.world_stoch_size
        # Number of classes per discrete variable (C = 32, values 0..31)
        self.num_classes = conf.world_class_size

        # BOS + each variable = 33 tokens
        self.seq_len = self.num_vars + 1
        # BOS + each variable = 33 class types
        self.vocab_size = self.num_classes + 1

        self.d_model = conf.self_prior_hidden_size

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Parameter(torch.randn(self.seq_len, self.d_model))

        # d-kim: The name encoder is confusing, but this behaves like a decoder.
        # Torch's built-in decoder does not allow removing cross-attention.
        # Removing cross-attention from the encoder yields a GPT-style decoder.
        #  https://discuss.pytorch.org/t/nn-transformerdecoderlayer-without-encoder-input/183990
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=conf.self_prior_n_heads,
                dim_feedforward=self.d_model * 2,
                batch_first=True,
            ),
            num_layers=conf.self_prior_n_layers,
        )
        self.fc_out = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, bos_x, use_gradient=False):
        if use_gradient:
            BT, K, C_plus_1 = bos_x.shape
            tok = bos_x @ self.token_emb.weight  # (B*T, K, D) (16384, 32, 128)
        else:
            BT, K = bos_x.shape
            tok = self.token_emb(bos_x)  # (B*T, K, D)

        pos = self.pos_emb[:K, :]  # (K, D)
        bos_x = tok + pos

        # causal mask for autoregression
        mask = torch.triu(
            torch.ones(K, K, device=bos_x.device) * float("-inf"),
            diagonal=1,
        )

        out = self.decoder(src=bos_x, mask=mask)
        logits = self.fc_out(out)  # (B*T, K, D)
        return logits

    def prepare(self, latent, use_gradient=False):
        if use_gradient:
            # logits = post or prior logits
            # (B, T, K, C) logits without bos token -> (B*T, K, C) with bos token
            #
            # x is discrete logits of shape (B, T, K, C).
            x = rearrange(latent, "B T K C -> (B T) K C", C=self.num_classes)
            x = torch.softmax(x.float(), dim=-1)
            # (B, T, K, C) -> (B, T, K, C+1)
            # dim C: pad_left=0, pad_right=1 (append BOS probability 0 at the end).
            pad = (0, 1)
            x = F.pad(x, pad=pad)
        else:
            # latent = post or prior
            # (B, T, K*C) one-hot without bos token -> (B*T, K) with bos token
            #
            # x is discrete latent variables of shape (B, T, K).
            x = rearrange(latent, "B T (K C) -> (B T) K C", C=self.num_classes)
            x = x.argmax(dim=-1)

        BT = x.shape[0]

        # bos_x is x with a BOS token prepended.
        bos_idx = self.num_classes  # Append BOS at index 64 after classes 0..63.
        bos = torch.full((BT, 1), bos_idx, dtype=torch.long, device=x.device)

        if use_gradient:
            bos = F.one_hot(bos, num_classes=self.vocab_size)

        bos_x = torch.cat([bos, x], dim=1)

        return bos_x

    def get_logits(self, bos_x, use_gradient=False):
        # Compute logits x_0..x_L from BOS..x_{L-1}.
        logits = self.forward(bos_x[:, :-1], use_gradient)
        # x_{1}, ..., x_{L}
        # TODO: Maybe need to detach?
        targets = bos_x[:, 1:]

        return logits, targets

    def get_logprob(self, logits, targets, use_gradient=False):
        if use_gradient:
            targets = targets
        else:
            # [B*T, D] -> [B*T, D, 1]
            targets = rearrange(targets, "... (d 1) -> ... d 1")

        # Find the log probabilities
        x = F.log_softmax(logits.float(), dim=-1)

        if use_gradient:
            x = x * targets
            logprob = reduce(x, "... k d -> ...", "sum")
        else:
            x = x.gather(-1, targets)
            logprob = reduce(x, "... d 1 -> ...", "sum")

        return logprob

    def get_sample(self, n_samples):
        bos_idx = self.num_classes  # Append BOS at index 64 after classes 0..63.
        x = torch.full((n_samples, 1), bos_idx, dtype=torch.long, device=self.device)
        for _ in range(self.num_vars):
            # Logits from the last step.
            logits = self.forward(x)[:, -1]
            logits[:, bos_idx] = -1e9  # Prevent BOS from being sampled.
            next_tok = torch.distributions.Categorical(logits=logits).sample()
            x = torch.cat([x, next_tok.unsqueeze(1)], dim=1)

        # Remove BOS token.
        x = x[:, 1:]

        # long tensor (B, K) -> one hot tensor (B, K, C)
        self_prior_samples = F.one_hot(x, num_classes=self.num_classes).float()
        self_prior_samples = rearrange(self_prior_samples, "B K C -> B (K C)")

        return self_prior_samples
