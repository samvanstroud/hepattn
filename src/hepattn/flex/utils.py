import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    _ModificationType,  # noqa: PLC2701
    _score_mod_signature,
    _vmap_for_bhqkv,  # noqa: PLC2701
)

try:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
except ImportError:
    from torch._higher_order_ops.flex_attention import TransformGetItemToIndex  # noqa: PLC2701
from contextlib import nullcontext

Tensor = torch.Tensor


def create_score_mod(
    query: torch.Tensor,
    key: torch.Tensor,
    score_mod: _score_mod_signature | None,
    mask_mod: _mask_mod_signature | None,
    device: str = "cuda",
    _compile: bool = False,
    scale: float | None = None,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> torch.Tensor:
    b = torch.arange(0, 1, device=device) + batch_idx
    h = torch.arange(0, 1, device=device) + head_idx
    m = torch.arange(0, query.shape[0], device=device)
    n = torch.arange(0, key.shape[0], device=device)

    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    type_ = _ModificationType.SCORE_MOD if score_mod is not None else _ModificationType.MASK_MOD
    ctx = nullcontext() if _compile else TransformGetItemToIndex()

    with ctx:
        mod_fn = score_mod if type_ == _ModificationType.SCORE_MOD else mask_mod
        prefix = (0,) if type_ == _ModificationType.SCORE_MOD else ()
        mod = _vmap_for_bhqkv(mod_fn, prefix=prefix)
        scores = query @ key.transpose(-2, -1)
        scores *= scale_factor
        scores = scores.view(1, 1, query.shape[0], key.shape[0])
        return mod(scores, b, h, m, n) if type_ == _ModificationType.SCORE_MOD else mod(b, h, m, n)


def _name_to_title(name: str) -> str:
    title = name.replace("_", " ")
    return " ".join(word.capitalize() for word in title.split())


def visualize_attention_scores(
    query: Tensor,
    key: Tensor,
    score_mod: _score_mod_signature | None = None,
    mask_mod: _mask_mod_signature | None = None,
    device: str = "cuda",
    name: str = "attention_scores",
    path: Path | None = None,
    batch_idx: int = 0,
    head_idx: int = 0,
    scale: float | None = None,
):
    """Generate and save a visualization of attention scores.

    Args:
        query (Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim).
        key (Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim).
        score_mod (Optional[Callable]): If this is set this will take precedence over the mask_mod.
        mask_mod (Optional[Callable]): The mask_mod function used to create block_mask
        device (str): Device to run computations on (default: "cuda").
        name (str): Base name for the file and title (default: 'attention_scores').
        path (Path): Path to save the visualization. If None, will be saved to the current working directory.
        batch_idx (int): Index of the batch to visualize (default: 0).
        head_idx (int): Index of the head to visualize (default: 0).
        scale (float): Scale factor to apply to the attention scores. If None, will be set to 1 / sqrt(head_dim).

    """
    assert score_mod is not None or mask_mod is not None, "Must provide either score_mod or mask_mod"
    query = query[batch_idx, head_idx, :, :]
    key = key[batch_idx, head_idx, :, :]
    scores_viz = create_score_mod(
        query,
        key,
        score_mod=score_mod,
        mask_mod=mask_mod,
        scale=scale,
        device=device,
        batch_idx=batch_idx,
        head_idx=head_idx,
    )

    suffix_title = f"Batch {batch_idx}, Head {head_idx}" if batch_idx != 0 or head_idx != 0 else ""

    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    color = "viridis" if score_mod is not None else "cividis"
    im = ax.imshow(scores_viz.cpu().detach()[0, 0, :, :], aspect="auto", cmap=color)
    fig.colorbar(im)

    title = _name_to_title(name)
    file_path = Path(name).with_suffix(".png") if path is None else path.with_suffix(".png")
    ax.set_title(f"{title}\n{suffix_title}", fontsize=20)

    ax.set_xlabel("Key Tokens", fontsize=18)
    ax.set_ylabel("Query Tokens", fontsize=18)

    # Move y-axis ticks and labels to the top
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

    # Add tick labels if the number of tokens is manageable
    num_query_tokens, num_kv_tokens = scores_viz.shape[-2:]
    if num_query_tokens <= 32 and num_kv_tokens <= 32:
        ax.set_xticks(range(num_kv_tokens))
        rotation = 45 if num_kv_tokens > 12 else 0
        ax.set_xticklabels([f"KV{i}" for i in range(num_kv_tokens)], fontsize=16, rotation=rotation)
        ax.set_yticks(range(num_query_tokens))
        ax.set_yticklabels([f"Q{i}" for i in range(num_query_tokens)], fontsize=16)
        # Align grid with pixel boundaries
        ax.set_xticks(np.arange(-0.5, num_kv_tokens, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_query_tokens, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free up memory

    print(f"Visualization saved as {file_path}")
