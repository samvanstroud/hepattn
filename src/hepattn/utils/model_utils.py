from torch import Tensor


def unmerge_inputs(x: dict[str, Tensor], input_names: list[str]) -> dict[str, Tensor]:
    """Unmerge the input features from a merged representation into separate input types.

    Args:
        x: Dictionary containing the merged input features.
        input_names: List of input names to unmerge.

    Returns:
        dict: Updated dictionary with separate input features for each type.
    """
    batch_size = x["key_embed"].shape[0]
    dim = x["key_embed"].shape[-1]
    for input_name in input_names:
        x[f"{input_name}_embed"] = x["key_embed"][x[f"key_is_{input_name}"]].view(batch_size, -1, dim)

    return x
