import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return x1 * F.silu(x2)

class SiLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x)