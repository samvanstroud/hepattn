import torch
from torch import Tensor, nn

from hepattn.models.decoder import MaskFormerDecoderLayer


class Fitter(nn.Module):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: nn.Module,
        decoder_layer_config: dict,
        num_decoder_layers: int,
        tasks: nn.ModuleList,
        dim: int,
        target_object: str = "particle",
        intermediate_losses: bool = False,
    ):
        super().__init__()

        self.input_nets = input_nets
        self.encoder = encoder
        self.decoder_layers = nn.ModuleList([MaskFormerDecoderLayer(**decoder_layer_config) for _ in range(num_decoder_layers)])
        self.tasks = tasks
        self.target_object = target_object
        self.intermediate_losses = intermediate_losses
        self.query_initial = nn.Parameter(torch.randn(dim))

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # Atomic input names
        input_names = [input_net.input_name for input_net in self.input_nets]

        assert "key" not in input_names, "'key' input name is reserved."
        assert "query" not in input_names, "'query' input name is reserved."

        x = {}

        x[f"{self.target_object}_valid"] = inputs[f"{self.target_object}_valid"]

        # Embed the input objects
        for input_net in self.input_nets:
            input_name = input_net.input_name
            x[input_name + "_embed"] = input_net(inputs)
            x[input_name + "_valid"] = inputs[input_name + "_valid"]
            x[f"{self.target_object}_{input_name}_valid"] = inputs[f"{self.target_object}_{input_name}_valid"]

            # These slices can be used to pick out specific
            # objects after we have merged them all together
            # TODO: Clean this up
            device = inputs[input_name + "_valid"].device
            x[f"key_is_{input_name}"] = torch.cat(
                [torch.full((inputs[i + "_valid"].shape[-1],), i == input_name, device=device, dtype=torch.bool) for i in input_names], dim=-1
            )

        # Merge the input objects and he padding mask into a single set
        x["key_embed"] = torch.concatenate([x[f"{input_name}_embed"] for input_name in input_names], dim=-2)
        x["key_valid"] = torch.concatenate([x[f"{input_name}_valid"] for input_name in input_names], dim=-1)
        x[f"{self.target_object}_key_valid"] = torch.concatenate(
            [x[f"{self.target_object}_{input_name}_valid"] for input_name in input_names], dim=-1
        )

        # (B, N, M, 1), (B, M, 1, N) -> (B, M, N, N) -> (B, N, N)
        attn_mask = torch.any(x[f"{self.target_object}_key_valid"].unsqueeze(-1) & x[f"{self.target_object}_key_valid"].unsqueeze(-2), dim=-3)
        batch_size = x[f"{self.target_object}_key_valid"].shape[0]
        query_size = x[f"{self.target_object}_key_valid"].shape[1]

        x["key_embed"] = self.encoder(x["key_embed"], kv_mask=x["key_valid"], attn_mask=attn_mask)

        # Generate the queries that represent objects
        x["query_embed"] = self.query_initial.expand(batch_size, query_size, -1)  # (D,) -> (B, N, D)
        x["query_valid"] = x[f"{self.target_object}_valid"]

        # Pass encoded inputs through decoder to produce outputs
        outputs = {}
        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            if self.intermediate_losses:
                outputs[f"layer_{layer_index}"] = {}

                for task in self.tasks:
                    # Get the outputs of the task given the current embeddings and record them
                    task_outputs = task(x)

                    outputs[f"layer_{layer_index}"][task.name] = task_outputs

            # Update the keys and queries
            x["query_embed"], x["key_embed"] = decoder_layer(
                x["query_embed"],
                x["key_embed"],
                attn_mask=x[f"{self.target_object}_key_valid"],
                kv_mask=x["key_valid"],
            )

            # Unmerge the updated features back into the separate input types
            for input_name in input_names:
                x[input_name + "_embed"] = x["key_embed"][..., x[f"key_is_{input_name}"], :]

        # Get the final outputs - we don't need to compute attention masks or update things here
        outputs["final"] = {}
        for task in self.tasks:
            outputs["final"][task.name] = task(x)

        return outputs

    def predict(self, outputs: dict) -> dict:
        """Takes the raw model outputs and produces a set of actual inferences / predictions.
        For example will take output probabilies and apply threshold cuts to prduce boolean predictions.

        Parameters
        ----------
        outputs:
            The outputs produces the forward pass of the model.
        """
        preds = {}

        # Compute predictions for each task in each block
        for layer_name, layer_outputs in outputs.items():
            preds[layer_name] = {}

            for task in self.tasks:
                preds[layer_name][task.name] = task.predict(layer_outputs[task.name])

        return preds

    def loss(self, outputs: dict, targets: dict) -> dict:
        """Computes the loss between the forward pass of the model and the data / targets.

        Parameters
        ----------
        outputs:
            The outputs produces the forward pass of the model.
        outputs:
            The data containing the targets.
        """
        # Compute the losses for each task in each block
        losses = {}
        for layer_name in outputs:
            if layer_name != "final" and not self.intermediate_losses:
                continue

            losses[layer_name] = {}
            for task in self.tasks:
                losses[layer_name][task.name] = task.loss(outputs[layer_name][task.name], targets)

        return losses
