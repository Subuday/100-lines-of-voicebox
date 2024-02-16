import torch
from torchdiffeq import odeint


class VoiceboxPipeline():

    def __init__(self):
        pass

    def __call__(
        self,
        steps: int = 10,
        device = 'cuda'
    ):
        def fn(t, x):
            return x

        x_0 = torch.rand(4, 80, 120).to(device)
        t = torch.linspace(0, 1, steps).to(device)
        x_1 = odeint(fn, x_0, t)[-1]