import torch
from torch import Tensor, nn


class Sorter(nn.Module):
    def __init__(
        self,
        input_sort_field: str,
        raw_variables: list[str] | None = None,
        input_sort_keys: dict[str, dict[str, int]] | None = None,
        target_sort_keys: dict[str, dict[str, str | int]] | None = None,
    ) -> None:
        super().__init__()
        self.input_sort_field = input_sort_field
        self.raw_variables = raw_variables or []
        self.input_sort_keys = input_sort_keys or {}
        self.target_sort_keys = target_sort_keys or {}

    def sort_inputs(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
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
        self.sort_indices = {}

        # Get key_embed shape for reference in sorting

        for input_hit in self.input_sort_keys:
            num_hits = x[f"{input_hit}_embed"].shape[1]
            sort_idx = self.get_sort_idx(x, input_hit, num_hits)
            for input_key, input_key_dim in self.input_sort_keys[input_hit].items():
                assert input_key in x, f"Input sort key {input_key} not found in x"
                if x[input_key] is not None:
                    x[input_key] = self._sort_tensor_by_index(x[input_key], sort_idx, num_hits, sort_dim=input_key_dim)

            # Sort raw variables if they have the right shape
            # TODO: what should the sort index for these be? Would they be associated with a particular input hit???
            # If I run these for both they will only sort if the num_hits matches? Risk is if multiple input_hit types have same no. hits?
            for raw_var in self.raw_variables:
                if raw_var in x:
                    x[raw_var] = self._sort_tensor_by_index(x[raw_var], sort_idx, num_hits)

        return x

    def sort_targets(self, targets: dict, sort_fields: dict[str, Tensor]) -> dict:
        """Sort targets to align with sorted outputs."""
        sort_indices = {}
        for input_hit in self.input_sort_keys:
            if input_hit == "key":
                continue
            key_sort_idx = self.get_sort_idx(sort_fields, input_hit)
            num_hits = key_sort_idx.shape[0]
            sort_indices[input_hit] = {"key_sort_idx": key_sort_idx, "num_hits": num_hits}

        targets_sorted = targets.copy()

        for input_hit in sort_indices:
            for key, value in targets.items():
                if input_hit not in key:
                    continue
                if key in self.target_sort_keys:
                    targets_sorted[key] = self._sort_tensor_by_index(
                        value,
                        sort_indices[input_hit]["key_sort_idx"],
                        sort_indices[input_hit]["num_hits"],
                        sort_dim=self.target_sort_keys[key]["input_hit_dim"],
                    )
                    assert not torch.allclose(targets_sorted[key], value), f"Target {key} is not sorted"
                else:
                    # sort targets if there is a dim that == num_hits and input_hit is in key
                    # this is a catch in case other class / element is added - shouldn't miss tensor that needs to be sorted
                    targets_sorted[key] = self._sort_tensor_by_index(
                        value, sort_indices[input_hit]["key_sort_idx"], sort_indices[input_hit]["num_hits"]
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
        if sort_dim is not None:
            assert tensor.shape[sort_dim] == num_hits, f"Sort dimension {sort_dim} has size {tensor.shape[sort_dim]} but num_hits is {num_hits}"
            return tensor.index_select(sort_dim, sort_idx)
        # If sort_dim is not provided, find the dimension with matching size
        for dim, size in enumerate(tensor.shape):
            if size == num_hits:
                sort_dim = dim
                break
        if sort_dim is not None:
            return tensor.index_select(sort_dim, sort_idx)
        return tensor
