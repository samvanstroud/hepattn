from pathlib import Path

import torch

from hepattn.flex import relative_position_wrapped, sliding_window_mask
from hepattn.flex.utils import visualize_attention_scores


def test_combined_mods(device="cpu"):
    def make_tensor(q_len):
        return torch.ones(1, 1, q_len, 8, device=device)

    q_len = torch.tensor([24])
    query, key = make_tensor(q_len[0]), make_tensor(q_len[0])

    mask = sliding_window_mask(4)
    score_mod = relative_position_wrapped(q_len)
    q_len = torch.tensor([24])
    query = make_tensor(q_len[0])

    out_dir = Path(__file__).parent.parent / Path("outputs/flex")
    out_dir.mkdir(exist_ok=True, parents=True)
    path = out_dir / "combined.png"

    visualize_attention_scores(query, key, score_mod=score_mod, mask_mod=mask, device=device, name="combined", path=path)
