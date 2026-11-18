"""Mix-FFN from SegFormer, Eq. 10 of the paper.

    Y_out = MLP(GELU(Conv3x3(MLP(y_in)))) + y_in

A feed-forward block with a depth-wise 3x3 convolution between its two
linear layers, so each position mixes in a little local spatial context
without positional encodings. It follows the attention in each relational
stage, with the residual connection written in the equation.

The two "MLP" layers are token-wise linear maps, which on a (B, C, H, W) tensor
are exactly 1x1 convolutions. Working in spatial form throughout means there is
no flatten / unflatten step to get the dimension order wrong.

The expansion ratio (hidden width = expansion * channels) is not given in the
paper; 4 is SegFormer's default and is a constructor argument here.
"""

from __future__ import annotations

import torch
from torch import nn


class MixFFN(nn.Module):
    """Eq. 10. Input and output are (B, C, H, W)."""

    def __init__(self, channels: int, expansion: int = 4):
        super().__init__()
        if expansion < 1:
            raise ValueError(f"expansion must be >= 1, got {expansion}")
        hidden = channels * expansion
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        # Depth-wise: one 3x3 filter per channel (groups == channels).
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.dwconv(self.fc1(x))))
