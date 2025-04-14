import pathlib

import comet_ml  # noqa: F401
import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.cli import ArgsType
from lion_pytorch import Lion
from torch import nn

from hepattn.experiments.trackml.old.trackml import TrackMLDataModule
from hepattn.utils.cli import CLI

config_dir = pathlib.Path(__file__).parent / "configs"


class HitFilter(LightningModule):
    def __init__(
        self,
        name: str,
        init: nn.Module,
        encoder: nn.Module,
        dense: nn.Module,
        pos_enc: nn.Module | None = None,
        target: str = "hit_tgt",
        opt_config: dict | None = None,
    ):
        super().__init__()

        self.name = name
        self.init = init
        self.encoder = encoder
        self.dense = dense
        self.target = target
        self.opt_config = opt_config
        self.times: list[float] = []
        self.num_hits: list[int] = []
        self.pos_enc = pos_enc

    def forward(self, x, labels=None, timing=False):  # noqa: ARG002
        if self.pos_enc:
            pe = self.pos_enc(x)

        x = self.init(x["hit"])

        if self.pos_enc:
            x += pe

        x = self.encoder(x)
        preds = self.dense(x).squeeze(-1)

        return {"hit_pred": preds}

    def loss(self, preds, labels):
        loss = None
        if labels:
            preds = preds["hit_pred"]
            tgt = labels["hit"][self.target]
            weight = 1 / tgt.float().mean()
            loss = F.binary_cross_entropy_with_logits(preds, tgt.type_as(preds), pos_weight=weight)
        return {"hit_pred": loss}

    def step(self, batch, **kwargs):
        inputs, labels = batch
        preds = self.forward(inputs, labels, **kwargs)
        loss = self.loss(preds, labels)
        loss["loss"] = sum(subloss for subloss in loss.values())
        return preds, labels, loss

    def training_step(self, batch):
        preds, labels, loss = self.step(batch)
        self.log_losses(loss, stage="train")
        self.log_metrics(preds, labels, stage="train")
        return loss["loss"]

    def validation_step(self, batch):
        preds, labels, loss = self.step(batch)
        self.log_losses(loss, stage="val")
        self.log_metrics(preds, labels, stage="val")
        return loss

    def test_step(self, batch):
        preds, _, _ = self.step(batch, timing=True)
        return preds

    def log_losses(self, loss, stage):
        kwargs = {"sync_dist": True, "batch_size": 1}
        self.log(f"{stage}/loss", loss["loss"], prog_bar=True, **kwargs)

    def log_metrics(self, preds, labels, stage):
        kwargs = {"sync_dist": True, "batch_size": 1}

        pred = preds["hit_pred"]
        tgt = labels["hit"]["tgt_pid"].bool()
        pred_true = pred.sigmoid() > 0.1
        self.log(f"{stage}/nh_total_pre", float(pred.shape[1]), **kwargs)
        self.log(f"{stage}/nh_total_post", float(pred_true.sum()), **kwargs)
        self.log(f"{stage}/nh_pred_true", pred_true.float().sum(), **kwargs)
        self.log(f"{stage}/nh_pred_false", (~pred_true).float().sum(), **kwargs)
        self.log(f"{stage}/nh_valid_pre", tgt.float().sum(), **kwargs)
        self.log(f"{stage}/nh_valid_post", (pred_true & tgt).float().sum(), **kwargs)
        self.log(f"{stage}/nh_noise_pre", (~tgt).float().sum(), **kwargs)
        self.log(f"{stage}/nh_noise_post", (pred_true & ~tgt).float().sum(), **kwargs)

        # accuracy
        self.log(f"{stage}/acc", (pred_true == tgt).half().mean(), **kwargs)

        # precision and recall
        tp = (pred_true * tgt).sum()
        self.log(f"{stage}/valid_recall", tp / tgt.sum(), **kwargs)
        self.log(f"{stage}/valid_precision", tp / pred_true.sum(), **kwargs)

        tn = ((~pred_true) * (~tgt)).sum()
        self.log(f"{stage}/noise_recall", tn / (~tgt).sum(), **kwargs)
        self.log(f"{stage}/noise_precision", tn / (~pred_true).sum(), **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam
        if self.opt_config["opt"] == "AdamW":
            optimizer = torch.optim.AdamW
        elif self.opt_config["opt"] == "Lion":
            optimizer = Lion
        else:
            raise ValueError(f"Unknown optimizer: {self.opt_config['opt']}")
        opt = optimizer(self.parameters(), lr=self.opt_config["initial_lr"], weight_decay=self.opt_config["weight_decay"])

        # 1cycle
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.opt_config["max_lr"],
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=self.opt_config["max_lr"] / self.opt_config["initial_lr"],
            final_div_factor=self.opt_config["initial_lr"] / self.opt_config["final_lr"],
            pct_start=float(self.opt_config["pct_start"]),
        )
        sch = {"scheduler": sch, "interval": "step"}

        return [opt], [sch]


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=HitFilter,
        datamodule_class=TrackMLDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
