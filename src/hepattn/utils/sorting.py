import torch
from torch import Tensor, nn


class Sorter(nn.Module):
    def __init__(self, input_sort_field: str) -> None:
        super().__init__()
        self.input_sort_field = input_sort_field
        self.input_names = None  # set by MaskFormer

    def sort_inputs(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        input_names = [*self.input_names, "key"]
        sort_idxs = {}

        for input_name in input_names:
            sort_idx = torch.argsort(inputs[f"{input_name}_{self.input_sort_field}"], dim=-1)
            sort_idxs[input_name] = sort_idx

            for key, x in inputs.items():
                if x is None or input_name not in key or "key_is_" in key:  # TODO: implement key_is_ sort!
                    continue

                # embeddings
                if key == f"{input_name}_embed":
                    sort_dim = 1
                    this_sort_idx = sort_idx.unsqueeze(-1).expand_as(x)

                # normal inputs
                elif key.startswith(input_name):
                    sort_dim = 1
                    this_sort_idx = sort_idx

                # input type masks
                # elif key == f"key_is_{input_name}":
                #    if input_name != "key":
                #        continue
                #    continue

                else:
                    raise ValueError(f"Unexpected key {key} for input type {input_name}")

                shape_before = x.shape
                inputs[key] = torch.gather(x, sort_dim, this_sort_idx)
                assert inputs[key].shape == shape_before, f"Shape mismatch after sorting: {inputs[key].shape} != {shape_before} for key {key}"

        return inputs

    def sort_targets(self, targets: dict, sort_fields: dict[str, Tensor]) -> dict:
        for input_name in self.input_names:
            sort_idx = torch.argsort(sort_fields[f"{input_name}_{self.input_sort_field}"], dim=-1)

            for key, x in targets.items():
                if x is None or input_name not in key:
                    continue

                # sort target mask
                if x.ndim == 3:
                    sort_dim = 2
                    this_sort_idx = sort_idx
                    this_sort_idx = sort_idx.unsqueeze(1).expand_as(x)

                # sort target for input constituent
                elif x.ndim == 2:
                    sort_dim = 1
                    this_sort_idx = sort_idx
                else:
                    raise ValueError(f"Unexpected key {key} for input hit {input_name}")

                shape_before = x.shape
                targets[key] = torch.gather(x, sort_dim, this_sort_idx)
                assert targets[key].shape == shape_before, f"Shape mismatch after sorting: {targets[key].shape} != {shape_before} for key {key}"

        return targets
