import torch
from torch import Tensor, nn


class HitFilter(nn.Module):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: nn.Module,
        tasks: nn.ModuleList,
        input_sort_field: str | None = None,
    ):
        super().__init__()

        self.input_nets = input_nets
        self.encoder = encoder
        self.input_sort_field = input_sort_field
        self.tasks = tasks

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # Atomic input names
        input_names = [input_net.input_name for input_net in self.input_nets]

        x = {}

        # Embed the input objects
        for input_net in self.input_nets:
            input_name = input_net.input_name
            x[input_name + "_embed"] = input_net(inputs)
            x[input_name + "_valid"] = inputs[input_name + "_valid"]

            x[f"key_is_{input_name}"] = torch.cat([torch.full((inputs[i + "_valid"].shape[-1],), i == input_name) for i in input_names], dim=-1)

        # Merge the input objects and he padding mask into a single set
        x["key_embed"] = torch.concatenate([x[input_name + "_embed"] for input_name in input_names], dim=-2)
        x["key_valid"] = torch.concatenate([x[input_name + "_valid"] for input_name in input_names], dim=-1)

        # Also merge the field being used for sorting in window attention if requested
        if self.input_sort_field is not None:
            x[f"key_{self.input_sort_field}"] = torch.concatenate(
                [inputs[input_name + "_" + self.input_sort_field] for input_name in input_names], dim=-1
            )
        else:
            x[f"key_{self.input_sort_field}"] = None

        # Pass merged input hits through the encoder
        if self.encoder is not None:
            x["key_embed"] = self.encoder(x["key_embed"], x[f"key_{self.input_sort_field}"])

        # Unmerge the updated features back into the separate input types
        for input_name in input_names:
            x[input_name + "_embed"] = x["key_embed"][..., x[f"key_is_{input_name}"], :]

        outputs = {"final": {}}
        for task in self.tasks:
            outputs["final"][task.name] = task(x)

        return outputs

    def predict(self, outputs: dict) -> dict:
        preds = {"final": {}}
        for task in self.tasks:
            preds["final"][task.name] = task.predict(outputs["final"][task.name])
        return preds

    def loss(self, outputs: dict, targets: dict) -> dict:
        losses = {"final": {}}
        for task in self.tasks:
            losses["final"][task.name] = task.loss(outputs["final"][task.name], targets)
        return losses
