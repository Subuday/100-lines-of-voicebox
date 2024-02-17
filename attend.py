from typing import Optional
import torch
from torch import nn, einsum
from torch.nn import Module
from packaging import version

from helpers import default

class Attend(Module):
    def __init__(
        self,
        scale: Optional[float] = None,
        dropout = 0.,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)

        self.use_flash_attention = use_flash_attention
        assert not (use_flash_attention and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'
        # TODO: Implement flash attention
        assert not use_flash_attention, 'flash attention is not yet implemented'

    def forward(self, q, k, v, mask = None):
        # TODO: Add mask support
        assert not mask, 'masking is not yet implemented'
        scale = default(self.scale, q.shape[-1] ** -0.5)

        qk = einsum(f"b h i d, b h j d -> b h i j", q, k)
        qk *= scale
        qk = qk.softmax(dim=-1)
        qk = self.dropout(qk)
        return einsum(f"b h i j, b h j d -> b h i d", qk, v)