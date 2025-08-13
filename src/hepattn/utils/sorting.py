import torch
from torch import Tensor, nn


class Sorter:
    def __init__(
        self,
        input_sort_field: str | None = None,
        raw_variables: list[str] | None = None,
        input_nets: nn.ModuleList | None = None,
    ) -> None:
        self.input_sort_field = input_sort_field
        self.raw_variables = raw_variables or []
        self.input_nets = input_nets or nn.ModuleList()

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
        self.num_hits = self._get_num_hits(x)
        assert self.num_hits > 0, "key embed not found in x"

        # Sort key embeddings and related data by the sort field
        if f"key_{self.input_sort_field}" not in x:
            print(f"Warning: Sort field {self.input_sort_field} not found in x")
            return x

        key_sort_value = x[f"key_{self.input_sort_field}"]
        key_sort_idx = torch.argsort(key_sort_value, dim=-1)
        if len(key_sort_idx.shape) == 2:
            key_sort_idx = key_sort_idx[0]
        assert len(key_sort_idx.shape) == 1, "Sort index must be 1D"
        assert key_sort_idx.shape[0] == self.num_hits, f"Key sort index shape {key_sort_idx.shape} does not match num_hits {self.num_hits}"
        self.sort_indices["key"] = key_sort_idx

        # TODO: sort key_phi (key_{input_sort_field})
        x[f"key_{self.input_sort_field}"] = self._sort_tensor_by_index(x[f"key_{self.input_sort_field}"], key_sort_idx, self.num_hits)

        for input_name in [input_net.input_name for input_net in self.input_nets]:
            if input_name + "_embed" in x:
                x[input_name + "_embed"] = self._sort_tensor_by_index(x[input_name + "_embed"], key_sort_idx, self.num_hits)
            if input_name + "_valid" in x:
                x[input_name + "_valid"] = self._sort_tensor_by_index(x[input_name + "_valid"], key_sort_idx, self.num_hits)
            if f"key_is_{input_name}" in x:
                x[f"key_is_{input_name}"] = self._sort_tensor_by_index(x[f"key_is_{input_name}"], key_sort_idx, self.num_hits)

        # Sort key embeddings
        x["key_embed"] = self._sort_tensor_by_index(x["key_embed"], key_sort_idx, self.num_hits)

        # Sort key validity mask
        if x["key_valid"] is not None:
            x["key_valid"] = self._sort_tensor_by_index(x["key_valid"], key_sort_idx, self.num_hits)

        # Sort raw variables if they have the right shape
        for raw_var in self.raw_variables:
            if raw_var in x and x[raw_var].shape[-1] == x["key_embed"].shape[-2]:
                x[raw_var] = self._sort_tensor_by_index(x[raw_var], key_sort_idx, self.num_hits)
            else:
                print(f"Warning: Raw variable {raw_var} has invalid shape: {x[raw_var].shape}")

        return x

    def sort_targets(self, targets: dict, sort_indices: dict[str, Tensor]) -> dict:
        """Sort targets to align with sorted outputs."""
        # TODO: check that this sorts all the targets correctly
        for key, value in targets.items():
            targets[key] = self._sort_tensor_by_index(value, sort_indices["key"], self.num_hits)

        return targets

    def _sort_tensor_by_index(self, tensor: Tensor, sort_idx: Tensor, num_hits: int) -> Tensor:
        """Sort a tensor along the dimension that has the same shape as key_embed[0].

        Parameters
        ----------
        tensor : Tensor
            Tensor to sort.
        sort_idx : Tensor
            Sort indices.
        num_hits : int
            Number of hits.

        Returns:
        Tensor
            Sorted tensor.
        """
        if tensor is None:
            return None
        sort_dim = None
        # If num hits is provided, find the dimension with matching size
        for dim, size in enumerate(tensor.shape):
            if size == num_hits:
                sort_dim = dim
                break

        if sort_dim is not None:
            return tensor.index_select(sort_dim, sort_idx.to(tensor.device))
        return tensor

    def _get_num_hits(self, x: dict[str, Tensor]) -> int:
        """Get the shape of key_embed tensor for reference in sorting.

        Parameters
        ----------
        x : dict[str, Tensor]
            Dictionary containing the key_embed tensor.

        Returns:
        int
            Number of hits from key_embed tensor.
        """
        if "key_embed" in x:
            return x["key_embed"].shape[-2]
        print(f"Key embed not found in x: {x.keys()}")
        return 0
