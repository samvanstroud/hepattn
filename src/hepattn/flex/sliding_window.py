from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature


def sliding_window_mask(window_size: int) -> _mask_mod_signature:
    def mask_mod(b, h, q_idx, kv_idx) -> Tensor:  # noqa: ARG001
        return (q_idx - kv_idx <= window_size // 2) & (kv_idx - q_idx <= window_size // 2)

    return mask_mod


def sliding_window_mask_wrapped(window_size: int, q_len: Tensor) -> _mask_mod_signature:
    def mask_mod(b, h, q_idx, kv_idx) -> Tensor:  # noqa: ARG001
        diagonal = (q_idx - kv_idx <= window_size // 2) & (kv_idx - q_idx <= window_size // 2)
        wrap = ((q_idx - kv_idx + q_len[0]) <= window_size // 2) | ((kv_idx - q_idx + q_len[0]) <= window_size // 2)
        return diagonal | wrap

    return mask_mod


def sliding_window_mask_strided(
    window_size: int, stride: float = 1.0
) -> _mask_mod_signature:
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    def mask_mod(b, h, q_idx, kv_idx) -> Tensor:  # noqa: ARG001
        # Calculate the center position for each query based on stride
        q_center = round(q_idx * stride)
        # Check if kv_idx is within the window around the strided center
        within_window = (kv_idx - q_center <= window_size // 2) & (
            q_center - kv_idx <= window_size // 2
        )
        return within_window

    return mask_mod


def sliding_window_mask_strided_wrapped(
    window_size: int, stride: float = 1.0, q_len: Tensor = None
) -> _mask_mod_signature:
    """
    Creates a sliding window mask with striding and wrapping support.

    Args:
        window_size: Size of the attention window (must be even)
        stride: Stride factor for sparse attention patterns
        q_len: Sequence length tensor for wrapping

    Returns:
        A function that computes attention masks with striding and wrapping
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even for strided sliding window")

    def mask_mod(b, h, q_idx, kv_idx) -> Tensor:  # noqa: ARG001
        # Calculate the center position for each query based on stride
        q_center = round(q_idx * stride)

        # Standard diagonal window
        diagonal = (kv_idx - q_center <= window_size // 2) & (
            q_center - kv_idx <= window_size // 2
        )

        wrap_left = ((kv_idx - q_center + q_len[0]) <= window_size // 2) & (
            q_center - kv_idx + q_len[0] <= window_size // 2
        )
        wrap_right = ((kv_idx - q_center - q_len[0]) <= window_size // 2) & (
            q_center - kv_idx - q_len[0] <= window_size // 2
        )
        return diagonal | wrap_left | wrap_right

    return mask_mod


# def auto_sliding_window_mask(q, kv, window_size, wrap=True):
#     n_objects = q.shape[1]
#     n_inputs = kv.shape[1]
#     device = q.device
#     stride = n_inputs / n_objects
#     if wrap:
#         mask = sliding_window_mask_strided_wrapped(window_size, stride, q_len)
#     else:
#         mask = sliding_window_mask_strided(window_size, stride)
#     assert not (
#         ~mask.any(dim=0)
#     ).any(), "Some columns are all False, increase window size"
#     return mask.unsqueeze(0)
