import torch
from torch import nn
from torch.nn import Module

from einops import rearrange

from helpers import exists


class Voicebox(Module):

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(128, 512)
        self.z_embed = nn.Embedding(2001 + 1, 512)
        self.lin_proj = nn.Linear(2 * 512 + 512, 512)
        self.time_embed = nn.Linear(1, 2048)
        self.transformer = nn.Linear(512, 512)
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