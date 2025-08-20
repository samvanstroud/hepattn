import torch
from torch import Tensor, nn


class Sorter(nn.Module):
    def __init__(
        self,
        input_sort_field: str,
        raw_variables: list[str] | None = None,
        input_sort_keys: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        self.input_sort_field = input_sort_field
        self.raw_variables = raw_variables or []
        self.input_names = None

    def sort_inputs(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        input_names = [*self.input_names, "key"]

        for input_name in input_names:
            num_hits = inputs[f"{input_name}_embed"].shape[1]
            sort_idx = self.get_sort_idx(inputs, input_name, num_hits)
            for key, x in inputs.items():
                if x is None or input_name not in key or "key_is" in key:
                    continue

                if key == f"{input_name}_embed":
                    sort_dim = 1
                    this_sort_idx = sort_idx.unsqueeze(-1).expand_as(x)
                elif key.startswith(input_name):
                    sort_dim = 1
                    this_sort_idx = sort_idx
                else:
                    raise ValueError(f"Unexpected key {key} for input hit {input_name}")

                shape_before = x.shape
                inputs[key] = self.batched_sort(x, this_sort_idx, sort_dim)
                assert inputs[key].shape == shape_before, f"Shape mismatch after sorting: {inputs[key].shape} != {shape_before} for key {key}"

        return inputs

    def sort_targets(self, targets: dict, sort_fields: dict[str, Tensor]) -> dict:
        sort_indices = {}
        for input_name in self.input_names:
            assert input_name != "key"
            if input_name == "key":
                continue
            key_sort_idx = self.get_sort_idx(sort_fields, input_name)
            num_hits = sort_fields[f"{input_name}_{self.input_sort_field}"].shape[1]
            sort_indices[input_name] = {"key_sort_idx": key_sort_idx, "num_hits": num_hits}

        for input_name in sort_indices:
            for key, x in targets.items():
                if x is None or input_name not in key:
                    continue

                # sort target mask
                if x.ndim == 3:
                    sort_dim = 2
                    sort_idx = sort_indices[input_name]["key_sort_idx"]
                    sort_idx = sort_idx.unsqueeze(1).expand_as(x)
                # sort target for input constituent
                elif x.ndim == 2:
                    sort_dim = 1
                    sort_idx = sort_indices[input_name]["key_sort_idx"]
                else:
                    raise ValueError(f"Unexpected key {key} for input hit {input_name}")

                shape_before = x.shape
                targets[key] = self.batched_sort(x, sort_idx, sort_dim=sort_dim)
                assert targets[key].shape == shape_before, f"Shape mismatch after sorting: {targets[key].shape} != {shape_before} for key {key}"
        return targets

    def get_sort_idx(self, x: dict[str, Tensor], input_name: str, num_hits=None) -> Tensor:
        sort_value = x[f"{input_name}_{self.input_sort_field}"]
        sort_idx = torch.argsort(sort_value, dim=-1).squeeze()
        if num_hits is not None:
            assert sort_idx.shape[-1] == num_hits, f"Key sort index shape {sort_idx.shape} does not match num_hits {num_hits}"
        return sort_idx

    def batched_sort(self, tensor: Tensor, sort_idx: Tensor, sort_dim: int) -> Tensor:
        """Sort a tensor along the dimension that has the same shape as key_embed[0].

        Args:
            tensor: Tensor to sort.
            sort_idx: Sort indices.
            sort_dim: Dimension to sort along.

        Returns:
            Sorted tensor.

        Raises:
            ValueError: If the tensor shape does not match the expected number of hits along the specified dimension.
        """
        if tensor.ndim < sort_idx.ndim:
            raise ValueError(f"Tensor {tensor.shape} has fewer dimensions than sort_idx {sort_idx.shape}")
        return torch.gather(tensor, sort_dim, sort_idx)
