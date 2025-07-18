from pathlib import Path

import pytest
import torch

from hepattn.flex import sliding_window_mask, sliding_window_mask_wrapped
from hepattn.flex.utils import visualize_attention_scores


@pytest.mark.parametrize("device", ["cpu"])
def test_sliding_window(device: str):
    """Visualize the attention scores of sliding window mask mod.

    Args:
        device (str): Device to use for computation. Defaults
    """

    def make_tensor(q_len):
        return torch.ones(1, 1, q_len, 8, device=device)

    q_len = torch.tensor([24])
    query, key = make_tensor(q_len[0]), make_tensor(q_len[0])

    out_dir = Path(__file__).parent.parent / Path("outputs/flex")
    out_dir.mkdir(exist_ok=True)

    mask = sliding_window_mask(4)
    path = out_dir / "sliding_window_mask.png"
    visualize_attention_scores(query, key, mask_mod=mask, device=device, name="sliding_window_mask", path=path)

    mask = sliding_window_mask_wrapped(6, q_len=q_len)
    path = out_dir / "sliding_window_mask_wrap.png"
    visualize_attention_scores(query, key, mask_mod=mask, device=device, name="sliding_window_mask_wrap", path=path)

    query, key = make_tensor(48), make_tensor(48)
    q_len[0] = 48
    path = out_dir / "sliding_window_mask_wrap2.png"
    visualize_attention_scores(query, key, mask_mod=mask, device=device, name="sliding_window_mask_wrap2", path=path)
