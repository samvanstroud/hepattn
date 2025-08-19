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

    def sort_inputs(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Sort inputs before passing to encoder for better window attention performance."""
        input_names = [*self.input_names, "key"]
        return self._sort_tensors(x, input_names, x)

    def sort_targets(self, targets: dict, sort_fields: dict[str, Tensor]) -> dict:
        """Sort targets to align with sorted outputs."""
        return self._sort_tensors(targets, self.input_names, sort_fields)

    def _sort_tensors(self, tensors: dict, input_names: list[str], sort_source: dict[str, Tensor]) -> dict:
        print(tensors.keys(), sort_source.keys())
        """Unified sorting logic for both inputs and targets."""

        # sort combined inputs once
        combined_sort_vals = sort_source[f"key_{self.input_sort_field}"]
        combined_sort_idx = torch.argsort(combined_sort_vals, dim=-1)
        combined_num = combined_sort_idx.shape[0]

        for input_name in input_names:
            num_hits = sort_source[f"{input_name}_embed"].shape[1]
            sort_idx = torch.argsort(sort_source[f"{input_name}_{self.input_sort_field}"], dim=-1)

            for key, x in tensors.items():
                if x is None or input_name not in key:
                    continue

                # this is the [batch, combined_num] mask to extract a single input type
                if "key_is_" in key:
                    this_sort_idx = combined_sort_idx
                    print(f"Sorting {key} for input {input_name} with num_hits {num_hits}")
                    print(f"x shape: {x.shape}, this_sort_idx shape: {this_sort_idx.shape}")
                    tensors[key] = torch.gather(x, -1, combined_sort_idx)

                # these are the uncombined inputs
                else:
                    sort_dim = self._get_sort_dimension(key, input_name)
                    this_sort_idx = sort_idx
                    if x.ndim != sort_idx.ndim:
                        this_sort_idx = this_sort_idx.unsqueeze(-1).expand_as(x)
                    print(f"Sorting {key} for input {input_name} with num_hits {num_hits}")
                    print(f"x shape: {x.shape}, this_sort_idx shape: {this_sort_idx.shape}")
                    tensors[key] = torch.gather(x, sort_dim, this_sort_idx)

        return tensors

    def get_sort_idx(self, x: dict[str, Tensor], input_name: str, num_hits=None) -> Tensor:
        sort_value = x[f"{input_name}_{self.input_sort_field}"]
        sort_idx = torch.argsort(sort_value, dim=-1)
        if len(sort_idx.shape) == 2:
            sort_idx = sort_idx[0]
        assert len(sort_idx.shape) == 1, "Sort index must be 1D"
        if num_hits is not None:
            assert sort_idx.shape[0] == num_hits, f"Key sort index shape {sort_idx.shape} does not match num_hits {num_hits}"
        return sort_idx

    def _get_sort_dimension(self, tensor_key: str, input_name: str) -> int:
        """Determine sort dimension: embeddings use dim 1, all others use -1."""
        return 1 if tensor_key.endswith("_embed") else -1

    def _sort_tensor_by_index(self, tensor: Tensor, sort_idx: Tensor, num_hits: int, sort_dim: int) -> Tensor:
        """Sort tensor along specified dimension.

        Raises:
            ValueError: If tensor dimension doesn't match expected num_hits.
        """
        if tensor.shape[sort_dim] != num_hits:
            raise ValueError(f"Sort dimension {sort_dim} has size {tensor.shape[sort_dim]} but expected {num_hits}")
        return tensor.index_select(sort_dim, sort_idx)
