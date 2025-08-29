import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.callbacks import Callback
from matplotlib.colors import ListedColormap


class AttnMaskLogger(Callback):
    def __init__(
        self,
        log_train: bool = True,
        log_val: bool = True,
        log_stats: bool = False,
    ):
        super().__init__()
        self.log_train = log_train
        self.log_val = log_val
        self.log_stats = log_stats

    def _log_attention_mask(self, pl_module, mask, step, layer, prefix="local_ca_mask"):
        """Helper method to create and log attention mask figures."""
        fig, ax = plt.subplots(constrained_layout=True, dpi=300)
        cmap = ListedColormap(["#002b7f", "#ffff33"])  # blue for 0, yellow for 1
        im = ax.imshow(mask.numpy().astype(int), aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        # Flip y-axis so lowest phi is at the bottom
        ax.invert_yaxis()
        # Add colorbar with clear labels
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.set_label("Attention Mask", rotation=270, labelpad=15)
        cbar.ax.set_yticklabels(["Masked (0)", "Used in Attention (1)"])
        # Add title with step and layer info
        ax.set_title(f"Attention Mask - Step {step}, Layer {layer}")
        # Add arrows to axis labels to indicate phi direction
        ax.set_xlabel("Hits (→ increasing φ)")
        ax.set_ylabel("Queries (→ increasing φ)")
        # Log directly to Comet
        logger = getattr(pl_module, "logger", None)
        if logger is not None and hasattr(logger, "experiment"):
            logger.experiment.log_figure(figure_name=f"{prefix}_step{step}_layer{layer}", figure=fig, step=step)
        plt.close(fig)

    def _log_attention_stats(self, pl_module, mask, step, layer, prefix="val"):
        """Log basic attention mask statistics."""
        try:
            # mask: shape [num_queries, num_constituents], dtype=bool or int
            hits_per_query = mask.sum(dim=1).cpu().numpy()  # shape: [num_queries]
            avg_hits_per_query = hits_per_query.mean()

            logger = getattr(pl_module, "logger", None)
            if logger is not None and hasattr(logger, "experiment"):
                logger.experiment.log_metrics(
                    {
                        f"{prefix}/attn_mask_avg_hits_per_query_layer{layer}": float(avg_hits_per_query),
                        f"{prefix}/attn_mask_max_hits_per_query_layer{layer}": float(np.max(hits_per_query)),
                        f"{prefix}/attn_mask_min_hits_per_query_layer{layer}": float(np.min(hits_per_query)),
                        f"{prefix}/attn_mask_std_hits_per_query_layer{layer}": float(np.std(hits_per_query)),
                    },
                    step=step,
                )
            else:
                print(f"[AttnMaskLogger] Step {step} Layer {layer} - Avg hits per query: {avg_hits_per_query}")
        except (ValueError, AttributeError, TypeError) as e:
            print(f"[AttnMaskLogger] Error logging stats: {e}")

    def _process_attention_masks_from_outputs(self, pl_module, outputs, step, is_validation=False):
        """Process attention masks directly from the outputs dictionary."""
        prefix_suffix = "_val" if is_validation else "train"

        # Get only entries that contain "attn_mask"
        layer_outputs = {k: v for k, v in outputs.items() if k != "loss" and "attn_mask" in v}
        if not layer_outputs:
            return

        layer_indices = sorted(int(k.split("_")[1]) for k in layer_outputs)
        if not layer_indices:
            return

        for layer_name, l_out in outputs.items():
            if layer_name != "loss" and "attn_mask" in l_out:
                layer_index = int(layer_name.split("_")[1])

                # log only first and last layer
                if layer_index == max(layer_indices):
                    attn_mask = l_out["attn_mask"]
                    attn_mask_im = attn_mask[0].detach().cpu().clone().int()
                    self._log_attention_mask(pl_module, attn_mask_im, step, layer_index, f"local_ma_mask_{prefix_suffix}")
                    if self.log_stats:
                        self._log_attention_stats(pl_module, attn_mask_im, step, layer_index, f"local_ma_mask_{prefix_suffix}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.log_val:
            return
        self._process_attention_masks_from_outputs(pl_module, outputs, batch_idx, is_validation=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.log_train:
            return
        # only process if this batch is selected by the sampler
        if batch_idx % 1000 != 0:
            return
        self._process_attention_masks_from_outputs(pl_module, outputs, batch_idx, is_validation=False)
