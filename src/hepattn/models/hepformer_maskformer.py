"""
Based on
- https://github.com/facebookresearch/MaskFormer
- https://github.com/facebookresearch/Mask2Former
"""

from collections.abc import Mapping

from torch import nn
from torch.nn import ModuleList

from hepattn.models.hepformer_loss import HEPFormerLoss


class FilterTracker(nn.Module):
    def __init__(
        self,
        loss_config: Mapping,
        input_net: nn.Module,
        mask_decoder: nn.Module,
        encoder: nn.Module | None = None,
        tasks: ModuleList | None = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.init_net = input_net
        self.mask_decoder = mask_decoder
        if tasks is None:
            tasks = []
        self.tasks = tasks

        # setup loss
        self.loss = HEPFormerLoss(**loss_config, tasks=tasks)

    def forward(self, inputs: dict, labels: dict, do_loss_matching: bool = True):
        input_pad_mask = None

        # run init net and encoder
        if self.encoder:
            x = self.init_net(inputs)
            # Pass merged input constituents through the encoder
            x_sort_value = inputs["hit_phi"]
            x = self.encoder(x, x_sort_value=x_sort_value)
            embed_x = x

        # get mask and flavour predictions
        preds = self.mask_decoder(embed_x, input_pad_mask)

        # print("mask logits mean/std:", preds["masks"].mean().item(), preds["masks"].std().item())

        # get the loss, updating the preds and labels with the best matching
        if do_loss_matching:  # set to false for inference timings
            preds, labels, loss = self.loss(preds, labels)
        else:
            loss = {}  # disable the bipartite matching
            for task in self.tasks:
                if task.input_type == "queries":
                    preds.update(task(preds, labels))

        # configurable tasks here
        for task in self.tasks:
            if task.input_type == "queries":
                task_loss = task.get_loss(preds, labels)
            else:
                task_preds, task_loss = task(preds, labels)
                preds.update(task_preds)
            loss.update(task_loss)

        return preds, loss


# need to return outputs in correct format
# need hepformer_loss to return losses in correct format
