from __future__ import annotations
from einops import repeat
import torch
from torch import nn
from torch.nn import Module
from torchdiffeq import odeint
from voicebox import Voicebox


class FlowMachingPipeline(Module):

    @staticmethod
    def from_pretrained(model_path, device = "cuda") -> FlowMachingPipeline:
        model = Voicebox()
        model.load_state_dict(torch.load(model_path, map_location = 'cpu'))
        model.to(device)
        return FlowMachingPipeline(model)

    def __init__(
        self, 
        model: Voicebox,
        sigma_min: float = 1e-5
    ):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min

    @torch.no_grad()
    def inference(
        self,
        steps: int = 10,
        cond_dropout: float = 0.2,
        device = 'cuda'
    ):
        def fn(t, x):
            t = repeat(t, '-> b', b = x.shape[0])
            z = torch.randint(1, 1000, (1, 767)).to(device)
            x_ctx = torch.rand_like(x).to(device)
            cond_mask_copy = torch.zeros((1, 384), dtype = torch.bool)
            cond_mask_infill = torch.zeros((1, 383), dtype = torch.bool)
            x_ctx_mask = torch.cat([cond_mask_copy, cond_mask_infill], dim = -1).to(device)

            cond_output = self.model.forward(z, x, t, x_ctx, x_ctx_mask, cond_dropout_prob = 0.)
            uncond_output = self.model.forward(z, x, t, x_ctx, x_ctx_mask, cond_dropout_prob = 1.)
            output = (1 - cond_dropout) * cond_output + cond_dropout * uncond_output

            assert output.shape == x.shape
            return x
        
        x_0 = torch.rand(1, 767, 128).to(device)
        t = torch.linspace(0, 1, steps).to(device)
        x_1 = odeint(fn, x_0, t)[-1]

    def forward(self, x_1):
        pass