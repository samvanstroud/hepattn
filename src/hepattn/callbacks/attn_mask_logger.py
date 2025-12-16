import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from matplotlib.colors import ListedColormap

from hepattn.utils.local_ca import auto_local_ca_mask


class AttnMaskLogger(Callback):
    def __init__(
        self,
        log_train: bool = True,
        log_val: bool = True,
        log_stats: bool = False,
        log_every_n_batches: int = 1000,
        lca_window_sizes: list[int] | None = None,
        log_diagonal_metrics: bool = True,
    ):
        super().__init__()
        self.log_train = log_train
        self.log_val = log_val
        self.log_stats = log_stats
        self.log_every_n_batches = log_every_n_batches
        self.lca_window_sizes = lca_window_sizes if lca_window_sizes is not None else [32, 64, 128, 512, 1024, 2048]
        self.log_diagonal_metrics = log_diagonal_metrics

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
            print(f"[AttnMaskLogger] Error logging attention stats: {e}")

    def _calculate_multi_lca_comparison_metrics(self, ma_mask):
        """Calculate LCA comparison metrics for multiple window sizes using dummy embeddings."""
        all_metrics = {}

        # Extract dimensions from the attention mask
        num_queries, num_hits = ma_mask.shape
        device = ma_mask.device

        # Create dummy embeddings with correct shapes
        # We use any tensor with the right shape[1] - the values don't matter for LCA mask generation
        dummy_q_embed = torch.zeros(1, num_queries, device=device)  # batch_size=1, num_queries
        dummy_kv_embed = torch.zeros(1, num_hits, device=device)  # batch_size=1, num_hits

        for window_size in self.lca_window_sizes:
            # Generate LCA mask for this window size
            lca_mask = auto_local_ca_mask(dummy_q_embed, dummy_kv_embed, window_size, wrap=True)
            lca_mask = lca_mask.squeeze(0)  # Remove batch dimension

            # Calculate efficiency: fraction of MA mask positions that are in LCA mask
            # Efficiency = (MA ∩ LCA) / MA
            ma_positions = ma_mask.sum()
            intersection = (ma_mask & lca_mask).sum()
            efficiency = float(intersection / ma_positions) if ma_positions > 0 else 0.0

            # Calculate purity: fraction of LCA mask positions that are in MA mask
            # Purity = (MA ∩ LCA) / LCA
            lca_positions = lca_mask.sum()
            purity = float(intersection / lca_positions) if lca_positions > 0 else 0.0

            # Store metrics with window size suffix
            window_suffix = f"_w{window_size}"
            all_metrics.update({
                f"attn_mask_lca_efficiency{window_suffix}": efficiency,
                f"attn_mask_lca_purity{window_suffix}": purity,
                f"attn_mask_intersection{window_suffix}": float(intersection),
            })

        return all_metrics

    def _calculate_distance_from_diagonal(self, mask):
        """Calculate how close the attention pattern is to a diagonal band."""
        # Using a vectorized PyTorch implementation for performance.
        num_queries, num_hits = mask.shape
        if num_queries == 0:
            return 0.0

        stride = num_hits / num_queries

        # Get indices of attended hits
        query_indices, hit_indices = torch.where(mask.bool())
        if query_indices.numel() == 0:
            return 0.0

        # Calculate expected diagonal hit index for each attended hit
        strided_query_indices = torch.round(query_indices.float() * stride)

        # Calculate distances
        distances = torch.abs(hit_indices.float() - strided_query_indices)

        # To calculate mean of means, we sum distances per query and divide by hits per query
        sum_distances_per_query = torch.zeros(num_queries, device=mask.device, dtype=torch.float)
        sum_distances_per_query.scatter_add_(0, query_indices, distances)

        hits_per_query = mask.sum(dim=1).float()

        # Queries with hits
        has_hits_mask = hits_per_query > 0
        if not has_hits_mask.any():
            return 0.0

        # Calculate average distance per query, avoiding division by zero
        avg_distance_per_query = torch.zeros_like(hits_per_query)
        avg_distance_per_query[has_hits_mask] = sum_distances_per_query[has_hits_mask] / hits_per_query[has_hits_mask]

        # Average of these averages
        avg_diagonal_distance = avg_distance_per_query[has_hits_mask].mean()

        # Normalize
        max_distance = num_hits / 2
        if max_distance == 0:
            return 0.0
        return float(avg_diagonal_distance / max_distance)

    def _calculate_diagonal_band_width(self, mask):
        """Calculate the average width of the diagonal band in the attention mask."""
        # Using a vectorized PyTorch implementation for performance.
        num_queries, num_hits = mask.shape
        if num_queries == 0:
            return 0.0

        has_hits = mask.any(dim=1)
        if not has_hits.any():
            return 0.0

        col_indices = torch.arange(num_hits, device=mask.device)
        masked_indices = col_indices.expand_as(mask)

        # For min, set non-attended to a large value
        first_hits = torch.where(mask.bool(), masked_indices, num_hits)
        min_attended_indices = torch.min(first_hits, dim=1).values[has_hits]

        # For max, set non-attended to a small value
        last_hits = torch.where(mask.bool(), masked_indices, -1)
        max_attended_indices = torch.max(last_hits, dim=1).values[has_hits]

        # Calculate widths
        widths = max_attended_indices - min_attended_indices + 1

        return float(widths.float().mean())

    def _calculate_attention_consistency(self, mask):
        """Calculate how consistent the attention pattern is across queries."""
        # Convert to numpy for easier processing
        mask_np = mask.cpu().numpy().astype(bool)

        # Calculate the number of hits each query attends to
        hits_per_query = mask_np.sum(axis=1)

        # Calculate consistency as 1 - coefficient of variation
        if len(hits_per_query) > 1:
            mean_hits = np.mean(hits_per_query)
            std_hits = np.std(hits_per_query)
            if mean_hits > 0:
                cv = std_hits / mean_hits
                consistency = max(0.0, 1.0 - cv)
                return float(consistency)

        return 1.0  # If all queries attend to the same number of hits

    def _log_diagonal_metrics(self, pl_module, ma_mask, step, layer, prefix="val"):
        """Log metrics comparing MA mask to LCA mask to measure diagonal consistency."""
        # Calculate basic metrics based on MA mask structure
        diagonal_distance = self._calculate_distance_from_diagonal(ma_mask)
        ma_positions = ma_mask.sum()
        diagonal_band_width = self._calculate_diagonal_band_width(ma_mask)
        consistency = self._calculate_attention_consistency(ma_mask)

        metrics = {
            f"{prefix}/attn_mask_distance_from_diagonal_layer{layer}": diagonal_distance,
            f"{prefix}/attn_mask_diagonal_band_width_layer{layer}": diagonal_band_width,
            f"{prefix}/attn_mask_consistency_layer{layer}": consistency,
            f"{prefix}/attn_mask_ma_positions_layer{layer}": float(ma_positions),
        }

        # Calculate LCA comparison metrics for all window sizes using dummy embeddings
        lca_metrics = self._calculate_multi_lca_comparison_metrics(ma_mask)
        metrics.update(lca_metrics)

        logger = getattr(pl_module, "logger", None)
        if logger is not None and hasattr(logger, "experiment"):
            logger.experiment.log_metrics(metrics, step=step)

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

                # log only last layer
                if layer_index == max(layer_indices):
                    attn_mask = l_out["attn_mask"]
                    attn_mask_im = attn_mask[0].detach().cpu().clone().int()
                    self._log_attention_mask(pl_module, attn_mask_im, step, layer_index, f"local_ma_mask_{prefix_suffix}")
                    if self.log_stats:
                        self._log_attention_stats(pl_module, attn_mask_im, step, layer_index, f"local_ma_mask_{prefix_suffix}")

                    # Log diagonal metrics if enabled
                    if self.log_diagonal_metrics:
                        self._log_diagonal_metrics(pl_module, attn_mask_im, step, layer_index, f"local_ma_mask_{prefix_suffix}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.log_val:
            return
        self._process_attention_masks_from_outputs(pl_module, outputs, batch_idx, is_validation=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.log_train:
            return
        # only process if this batch is selected by the sampler
        if batch_idx % self.log_every_n_batches != 0:
            return
        self._process_attention_masks_from_outputs(pl_module, outputs, batch_idx, is_validation=False)
