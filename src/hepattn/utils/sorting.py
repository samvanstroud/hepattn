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

    def sort_inputs(self, x: dict[str, Tensor], input_names: list[str]) -> dict[str, Tensor]:
        """Sort inputs before passing to encoder for better window attention performance.

        Parameters
        ----------
        x : dict[str, Tensor]
            Dictionary containing embeddings and other data to be sorted.

        Returns:
        -------
        dict[str, Tensor]
            Sort indices for key and query dimensions.
        """
        self.input_names = [*input_names, "key"]
        for input_hit in input_names:
            # Get key_embed shape for reference in sorting
            num_hits = x[f"{input_hit}_embed"].shape[1]
            sort_idx = self.get_sort_idx(x, input_hit, num_hits)
            for key, val in x.items():
                if val is None:
                    continue
                if not (key.startswith(input_hit) or key.endswith(input_hit)):
                    continue
                x[key] = self._sort_tensor_by_index(val, sort_idx, num_hits)
        return x

    def sort_targets(self, targets: dict, sort_fields: dict[str, Tensor]) -> dict:
        """Sort targets to align with sorted outputs."""
        sort_indices = {}
        for input_hit in self.input_names:
            if input_hit == "key":
                continue
            key_sort_idx = self.get_sort_idx(sort_fields, input_hit)
            num_hits = key_sort_idx.shape[0]
            sort_indices[input_hit] = {"key_sort_idx": key_sort_idx, "num_hits": num_hits}

        targets_sorted = targets.copy()

        for input_hit in sort_indices:
            for key, value in targets.items():
                key_split = key.split("_")[1]
                sort_dim = 2 if key_split.startswith(input_hit) else None
                if key.startswith(input_hit) or key_split.startswith(input_hit):
                    targets_sorted[key] = self._sort_tensor_by_index(
                        value,
                        sort_indices[input_hit]["key_sort_idx"],
                        sort_indices[input_hit]["num_hits"],
                        sort_dim=sort_dim,
                    )
        return targets_sorted

    def get_sort_idx(self, x: dict[str, Tensor], input_hit: str, num_hits=None) -> Tensor:
        sort_value = x[f"{input_hit}_{self.input_sort_field}"]
        sort_idx = torch.argsort(sort_value, dim=-1)
        if len(sort_idx.shape) == 2:
            sort_idx = sort_idx[0]
        assert len(sort_idx.shape) == 1, "Sort index must be 1D"
        if num_hits is not None:
            assert sort_idx.shape[0] == num_hits, f"Key sort index shape {sort_idx.shape} does not match num_hits {num_hits}"
        return sort_idx

    def _sort_tensor_by_index(self, tensor: Tensor, sort_idx: Tensor, num_hits: int, sort_dim: int | None = None) -> Tensor:
        """Sort a tensor along the dimension that has the same shape as key_embed[0].

        Parameters
        ----------
        tensor : Tensor
            Tensor to sort.
        sort_idx : Tensor
            Sort indices.
        num_hits : int
            Number of hits.
        sort_dim : int | None
            Dimension to sort along.

        Returns:
        Tensor
            Sorted tensor.
        """
        if sort_dim is None:
            sort_dim = 0 if tensor.ndim == 1 else 1
        if tensor.shape[sort_dim] != num_hits:
            print(f"Sort dimension {sort_dim} has size {tensor.shape[sort_dim]} but num_hits is {num_hits}")
            return tensor
        return tensor.index_select(sort_dim, sort_idx)
