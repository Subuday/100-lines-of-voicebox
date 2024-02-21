from __future__ import annotations
from einops import repeat
import torch
from torchdiffeq import odeint

from voicebox import Voicebox


class FlowMachingPipeline():

    @staticmethod
    def from_pretrained(model_path, device = "cuda") -> FlowMachingPipeline:
        model = Voicebox()
        model.load_state_dict(torch.load(model_path, map_location = 'cpu'))
        model.to(device)
        return FlowMachingPipeline(model)

    def __init__(self, model: Voicebox):
        self.model = model

    def __call__(
        self,
        steps: int = 10,
        device = 'cuda'
    ):
        self.model.eval()

        def fn(t, x):
            t = repeat(t, '-> b', b = x.shape[0])
            z = torch.randint(1, 1000, (1, 767)).to(device)
            x_ctx = torch.rand_like(x).to(device)
            cond_mask_copy = torch.zeros((1, 384), dtype = torch.bool)
            cond_mask_infill = torch.zeros((1, 383), dtype = torch.bool)
            x_ctx_mask = torch.cat([cond_mask_copy, cond_mask_infill], dim = -1).to(device)
            output = self.model.inference(z, x, t, x_ctx, x_ctx_mask)
            assert output.shape == x.shape
            return x
        
        x_0 = torch.rand(1, 767, 128).to(device)
        t = torch.linspace(0, 1, steps).to(device)
        x_1 = odeint(fn, x_0, t)[-1]