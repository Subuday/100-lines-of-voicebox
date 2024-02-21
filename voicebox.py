import math
from typing import Optional
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange
from attend import Attend

from helpers import default, exists


class SinusoidalPositionEmbedder(Module):

    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        half_dim = dim // 2
        self.w = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = 2 * math.pi * x * rearrange(self.w, 'd -> 1 d')
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        return fouriered

class ConvolutionPositionEmbedder(Module):

    def __init__(
        self,
        dim: int,
        *,
        kernel_size: int,
        groups: Optional[int] = None
    ):
        super().__init__()
        assert kernel_size % 2 == 1, 'kernel size must be even'
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups = default(groups, dim), padding = kernel_size // 2),
            nn.GELU()
        )

    def forward(self, x, mask = None):
        # TODO: Add mask support
        assert not mask, 'masking is not yet implemented'
        x = rearrange(x, 'b c t -> b t c')
        x = self.block(x)
        x = rearrange(x, 'b t c -> b c t')
        return x


class GEGLU(Module):
    
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma


class FeedForward(Module):
    def __init__(
        self,
        dim: int,
        dropout: float = 0.,
        mult: int = 4
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(), #TODO: Read about GEGLU
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.block(x)


class Attention(Module):
    def __init__(
        self,
        dim: int,
        *,
        dim_head: int = 64,
        num_heads: int = 8,
        dropout: float = 0.,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.to_qkv = nn.Linear(dim, dim_head * num_heads * 3, bias = False)
        self.attend = Attend(dropout = dropout)
        self.to_out = nn.Linear(dim_head * num_heads, dim)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h = self.num_heads), (q, k, v))
        # TODO: Add normalization component
        # TODO: Add rotary component
        output = self.attend(q, k, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        return output
        

class Transformer(Module):
    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        dim_head: int,
        num_heads: int,
        use_skip_connection: bool = False,
        skip_connection_scale: Optional[float] = None,
        ff_mult: int = 4,
        ff_dropout: float = 0.,
    ):
        super().__init__()
        self.skip_connection_scale = None
        if use_skip_connection:
            self.skip_connection_scale = default(skip_connection_scale, 2 ** -0.5)

        self.layers = nn.ModuleList([])
        assert depth % 2 == 0
        for layer in range(1, depth + 1):
            has_skip_connection = use_skip_connection and layer > (depth // 2)
            self.layers.append(
                nn.ModuleList([
                    nn.Linear(dim * 2, dim) if has_skip_connection else None,
                    # TODO: Check GateLoop
                    # TODO: Check AdaptiveRMSNorm
                    RMSNorm(dim = dim),
                    Attention(dim = dim, dim_head = dim_head, num_heads = num_heads),
                    RMSNorm(dim = dim),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
                ])
            )

        self.out_norm = RMSNorm(dim)

    def forward(self, x):
        hidden_states = []
        for connector, attn_norm, attn, ff_norm, ff in self.layers:
            # In the paper, they use a u-net like skip connection.
            # It's unclear how much this helps, as no ablations or further numbers given besides a brief one-two sentence mention.
            if not exists(connector):
                hidden_states.append(x)
            else:
                hidden_state = hidden_states.pop()
                hidden_state *= self.skip_connection_scale
                x = torch.cat((x, hidden_state), dim = -1)
                x = connector(x)

            attn_input = attn_norm(x)
            x = x + attn(attn_input)

            ff_input = ff_norm(x)
            x = x + ff(ff_input)

        return self.out_norm(x)


class Voicebox(Module):

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(128, 512)
        self.z_embed = nn.Embedding(2001 + 1, 512)
        self.lin_proj = nn.Linear(2 * 512 + 512, 512)
        self.conv_embed = ConvolutionPositionEmbedder(512, kernel_size = 31, groups = None) # TODO: Read about ConvolutionPositionEmbedder
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedder(512),
            nn.Linear(512, 2048),
            nn.SiLU()
        )
        self.transformer = Transformer(
            dim = 512,
            depth = 6,
            dim_head = 64,
            num_heads = 8,
            use_skip_connection = True
        )
        self.out_proj = nn.Linear(512, 128)

    def forward(
        self,
        z,
        x,
        t,
        x_ctx = None,
        x_ctx_mask = None,
        cond_dropout_prob: float = 0.2,
    ):
        x = self.in_proj(x)

        if x_ctx is not None:
            x_ctx = self.in_proj(x_ctx)

            x_ctx_mask = rearrange(x_ctx_mask, 'b c -> b c 1')
            x_ctx *= ~x_ctx_mask
        
        if cond_dropout_prob > 0.:
            # TODO: Add CGF
            pass

        z_emb  = self.z_embed(z)
        
        x = torch.cat([*filter(exists, (z_emb, x, x_ctx))], dim = -1)
        
        x = self.lin_proj(x)
        x = x + self.conv_embed(x)

        t = self.time_embed(t)

        x = self.transformer(x)

        x = self.out_proj(x)

        return x