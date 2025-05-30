from pathlib import Path

import torch

from hepattn.flex import relative_position, relative_position_wrapped
from hepattn.flex.relative_position import identity
from hepattn.flex.utils import visualize_attention_scores


def test_relative_position(device: str = "cpu"):
    """Visualize the attention scores of sliding window mask mod.

    Args:
        device (str): Device to use for computation. Defaults
    """

    def make_tensor(q_len):
        return torch.rand(1, 1, q_len, 8, device=device)

    q_len = torch.tensor([24])
    query = make_tensor(q_len[0])
    out_dir = Path(__file__).parent.parent / Path("outputs/flex")
    out_dir.mkdir(exist_ok=True)

    path = out_dir / "baseline.png"
    visualize_attention_scores(query, query, score_mod=identity, device=device, name="baseline", path=path)

    path = out_dir / "relative_position.png"
    visualize_attention_scores(query, query, score_mod=relative_position, device=device, name="relative_position", path=path)

    score_mod = relative_position_wrapped(q_len)
    path = out_dir / "wrapped.png"
    visualize_attention_scores(query, query, score_mod=score_mod, device=device, name="wrapped", path=path)
