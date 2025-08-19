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
        input_names = [self.input_names, "key"]
        return self._sort_tensors(x, input_names, x)

    def sort_targets(self, targets: dict, sort_fields: dict[str, Tensor]) -> dict:
        """Sort targets to align with sorted outputs."""
        input_names = [name for name in self.input_names if name != "key"]
        return self._sort_tensors(targets, input_names, sort_fields)

    def _sort_tensors(self, tensors: dict, sort_source: dict[str, Tensor]) -> dict:
        """Unified sorting logic for both inputs and targets."""
        for input_hit in self.input_names:
            # Get sorting info
            num_hits = sort_source[f"{input_hit}_embed"].shape[1]
            sort_idx = self.get_sort_idx(sort_source, input_hit, num_hits)

            # Sort all relevant tensors in-place
            for key, val in tensors.items():
                if val is None or not self._should_sort_tensor(key, input_hit):
                    continue
                sort_dim = self._get_sort_dimension(key, input_hit)
                tensors[key] = self._sort_tensor_by_index(val, sort_idx, num_hits, sort_dim)

        return tensors

    def get_sort_idx(self, x: dict[str, Tensor], input_hit: str, num_hits=None) -> Tensor:
        sort_value = x[f"{input_hit}_{self.input_sort_field}"]
        sort_idx = torch.argsort(sort_value, dim=-1)
        if len(sort_idx.shape) == 2:
            sort_idx = sort_idx[0]
        assert len(sort_idx.shape) == 1, "Sort index must be 1D"
        if num_hits is not None:
            assert sort_idx.shape[0] == num_hits, f"Key sort index shape {sort_idx.shape} does not match num_hits {num_hits}"
        return sort_idx

    def _should_sort_tensor(self, tensor_key: str, input_hit: str) -> bool:
        """Check if tensor should be sorted - unified logic for inputs and targets."""
        return tensor_key.startswith(input_hit) or tensor_key.endswith(input_hit) or f"_{input_hit}_" in tensor_key

    def _get_sort_dimension(self, tensor_key: str, input_hit: str) -> int:
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
