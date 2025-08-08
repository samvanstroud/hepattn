import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from matplotlib.colors import ListedColormap


class AttnMaskLogger(Callback):
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

    def _log_attention_weights(self, pl_module, weights, step, layer, prefix="local_ca_weights"):
        """Helper method to create and log attention weights figures."""
        fig, ax = plt.subplots(constrained_layout=True, dpi=300)
        # Ensure weights is a numpy float array
        if isinstance(weights, torch.Tensor):
            weights_np = weights.detach().cpu().float().numpy()
        elif isinstance(weights, np.ndarray):
            weights_np = weights.astype(float)
        else:
            weights_np = np.array(weights, dtype=float)
        im = ax.imshow(weights_np, aspect="auto", cmap="viridis", interpolation="nearest")
        # Flip y-axis so lowest phi is at the bottom
        ax.invert_yaxis()
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Attention Weights", rotation=270, labelpad=15)
        # Add arrows to axis labels to indicate phi direction
        ax.set_xlabel("Hits (→ increasing φ)")
        ax.set_ylabel("Queries (→ increasing φ)")
        logger = getattr(pl_module, "logger", None)
        if logger is not None and hasattr(logger, "experiment"):
            logger.experiment.log_figure(figure_name=f"{prefix}_step{step}_layer{layer}", figure=fig, step=step)
        plt.close(fig)

    def _process_attention_masks(self, pl_module, is_validation=False):
        """Helper method to process and log attention masks from the model."""
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        prefix_suffix = "_val" if is_validation else ""

        # Log final attention masks from the main model
        if hasattr(model, "output_attn_masks_to_log"):
            for mask_info in model.output_attn_masks_to_log.values():
                probs = mask_info["probs"]
                mask = mask_info["mask"]
                step = mask_info["step"]
                layer = mask_info["layer"]
                self._log_attention_weights(pl_module, probs, step, layer, f"final_ca_weights_output{prefix_suffix}")
                self._log_attention_mask(pl_module, mask, step, layer, f"final_ca_mask_output{prefix_suffix}")
            delattr(model, "output_attn_masks_to_log")

        # Log attention masks from the decoder
        if hasattr(model, "decoder") and hasattr(model.decoder, "attn_masks_to_log"):
            for mask_info in model.decoder.attn_masks_to_log.values():
                mask = mask_info["mask"]
                step = mask_info["step"]
                layer = mask_info["layer"]
                self._log_attention_mask(pl_module, mask, step, layer, f"local_ca_mask{prefix_suffix}")
            # Clear after logging
            delattr(model.decoder, "attn_masks_to_log")

        # Log strided attention masks from the decoder
        if hasattr(model, "decoder") and hasattr(model.decoder, "strided_masks_to_log"):
            for mask_info in model.decoder.strided_masks_to_log.values():
                mask = mask_info["mask"]
                step = mask_info["step"]
                layer = mask_info["layer"]
                self._log_attention_mask(pl_module, mask, step, layer, f"strided_ca_mask{prefix_suffix}")
            # Clear after logging
            delattr(model.decoder, "strided_masks_to_log")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._process_attention_masks(pl_module, is_validation=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._process_attention_masks(pl_module, is_validation=False)


class AttentionStatsLogger(Callback):
    """Callback for logging basic attention statistics."""

    def __init__(self, log_train: bool = False, log_val: bool = True):
        """Initialize the attention stats logger.

        Args:
            log_train: Whether to log stats during training
            log_val: Whether to log stats during validation
        """
        super().__init__()
        self.log_train = log_train
        self.log_val = log_val

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
                print(f"[AttentionStatsLogger] Step {step} Layer {layer} - Avg hits per query: {avg_hits_per_query}")
        except (ValueError, AttributeError, TypeError) as e:
            print(f"[AttentionStatsLogger] Error logging stats: {e}")

    def _process_attention_stats(self, pl_module, is_validation=False):
        """Helper method to process and log attention statistics from the model."""
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        prefix_base = "val" if is_validation else "train"

        # Check for final attention masks from the main model
        if hasattr(model, "output_attn_masks_to_log"):
            for mask_info in model.output_attn_masks_to_log.values():
                step = mask_info["step"]
                mask = mask_info["mask"]
                layer = mask_info["layer"]
                self._log_attention_stats(pl_module, mask, step, layer, f"{prefix_base}_final")

        # Check for attention masks from the decoder
        if hasattr(model, "decoder") and hasattr(model.decoder, "attn_masks_to_log"):
            for mask_info in model.decoder.attn_masks_to_log.values():
                step = mask_info["step"]
                mask = mask_info["mask"]
                layer = mask_info["layer"]
                self._log_attention_stats(pl_module, mask, step, layer, prefix_base)

        # Check for strided attention masks from the decoder
        if hasattr(model, "decoder") and hasattr(model.decoder, "strided_masks_to_log"):
            for mask_info in model.decoder.strided_masks_to_log.values():
                step = mask_info["step"]
                mask = mask_info["mask"]
                layer = mask_info["layer"]
                self._log_attention_stats(pl_module, mask, step, layer, f"{prefix_base}_strided")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.log_train:
            return
        self._process_attention_stats(pl_module, is_validation=False)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.log_val:
            return
        self._process_attention_stats(pl_module, is_validation=True)
