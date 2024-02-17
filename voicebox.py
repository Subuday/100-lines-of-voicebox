import math
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange
from attend import Attend

from helpers import exists


class PositionEmbedder(Module):

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

class GEGLU(Module):
    
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)


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
        ff_mult: int = 4,
        ff_dropout: float = 0.,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        assert depth % 2 == 0
        for layer in range(1, depth + 1):
            has_scip_connection = use_skip_connection and layer > (depth // 2)
            self.layers.append(
                nn.ModuleList([
                    nn.Linear(dim * 2, dim) if has_scip_connection else None,
                    # GateLoop(dim = dim, use_jax_associative_scan = gateloop_use_jax, post_ln = True) if use_gateloop_layers else None,
                    # rmsnorm_klass(dim = dim),
                    Attention(dim = dim, dim_head = dim_head, num_heads = num_heads),
                    # rmsnorm_klass(dim = dim),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
                ])
            )


    def forward(self, x):
        for _,  attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class Voicebox(Module):

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(128, 512)
        self.z_embed = nn.Embedding(2001 + 1, 512)
        self.lin_proj = nn.Linear(2 * 512 + 512, 512)
        self.time_embed = nn.Sequential(
            PositionEmbedder(512),
            nn.Linear(512, 2048),
            nn.SiLU()
        )
        self.transformer = Transformer(
            dim = 512,
            depth = 6,
            dim_head = 64,
            num_heads = 8,
        )
        self.out_proj = nn.Linear(512, 128)

    @torch.inference_mode()
    def inference(self, z, x, t, x_ctx, x_ctx_mask, cond_dropout=0.2):
        cond_output = self._inference(z, x, t, x_ctx, x_ctx_mask, cond_dropout_prob = 0.)
        uncond_output = self._inference(z, x, t, x_ctx, x_ctx_mask, cond_dropout_prob = 1.)
        return (1 - cond_dropout) * cond_output + cond_dropout * uncond_output

    def _inference(
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
        # TODO: Add convolution

        t = self.time_embed(t)

        x = self.transformer(x)

        x = self.out_proj(x)

        return x