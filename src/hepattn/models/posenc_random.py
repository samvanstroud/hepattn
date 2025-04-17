import numpy as np
import torch
from torch import Tensor, nn


class PositionEncoderRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    Adapted from https://github.com/pytorch-labs/segment-anything-fast/blob/main/segment_anything_fast/modeling/prompt_encoder.py
    """

    def __init__(self, dim: int, variables: list[str], scale: float = 1) -> None:
        super().__init__()
        assert scale > 0
        assert dim % 2 == 0, "Dimension must be even"
        self.variables = variables
        self.gaussian_matrix = torch.nn.parameter.Buffer(scale * torch.randn((len(variables), dim // 2)))

    def forward(self, xs: dict[str, Tensor]) -> Tensor:
        xs = torch.cat([xs[f].unsqueeze(-1) for f in self.variables], dim=-1)
        xs = 2 * np.pi * xs
        xs @= self.gaussian_matrix
        return torch.cat([torch.sin(xs), torch.cos(xs)], dim=-1)
