import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from typing import Optional, Dict, Any, Union, Tuple


def cyclic_diff(phi1: float, phi2: float) -> float:
    """
    Compute the cyclic difference between two phi values.
    Returns the smallest angular difference in the range [0, π].
    """
    try:
        diff = phi1 - phi2
        # Wrap to [-π, π] first
        wrapped_diff = np.arctan2(np.sin(diff), np.cos(diff))
        # Take absolute value to get smallest angular distance [0, π]
        return float(np.abs(wrapped_diff))
    except (TypeError, ValueError) as e:
        return float('nan')


class AttentionMaskVisualizer(Callback):
    """Callback for visualizing attention masks as heatmaps."""
    
    def __init__(self, log_train: bool = True, log_val: bool = True):
        """
        Initialize the attention mask visualizer.
        
        Args:
            log_train: Whether to log attention masks during training
            log_val: Whether to log attention masks during validation
        """
        super().__init__()
        self.log_train = log_train
        self.log_val = log_val

    def _log_attention_mask(self, pl_module, mask, step, layer, prefix="local_ca_mask"):
        """Create and log attention mask heatmap."""
        fig, ax = plt.subplots(constrained_layout=True, dpi=300)
        try:
            cmap = ListedColormap(['#002b7f', '#ffff33'])  # blue for 0, yellow for 1
            im = ax.imshow(mask.numpy().astype(int), aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            # Add colorbar with clear labels
            cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
            cbar.set_label('Attention Mask', rotation=270, labelpad=15)
            cbar.ax.set_yticklabels(['Masked (0)', 'Used in Attention (1)'])
            # Add title with step and layer info
            ax.set_title(f'Attention Mask - Step {step}, Layer {layer}')
            ax.set_xlabel('Input Constituents')
            ax.set_ylabel('Queries')
            # Log directly to Comet
            logger = getattr(pl_module, 'logger', None)
            if logger is not None and hasattr(logger, 'experiment'):
                logger.experiment.log_figure(
                    figure_name=f"{prefix}_step{step}_layer{layer}",
                    figure=fig,
                    step=step
                )
        except Exception as e:
            print(f"[AttentionMaskVisualizer] Error logging attention mask: {e}")
        finally:
            plt.close(fig)
            del fig

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.log_train:
            return
            
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        if hasattr(model, 'attn_masks_to_log'):
            for mask_info in model.attn_masks_to_log.values():
                mask = mask_info["mask"]
                step = mask_info["step"]
                layer = mask_info["layer"]
                self._log_attention_mask(pl_module, mask, step, layer, "local_ca_mask")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.log_val:
            return
            
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        if hasattr(model, 'attn_masks_to_log'):
            for mask_info in model.attn_masks_to_log.values():
                mask = mask_info["mask"]
                step = mask_info["step"]
                layer = mask_info["layer"]
                self._log_attention_mask(pl_module, mask, step, layer, "local_ca_mask_val")


class AttentionStatsLogger(Callback):
    """Callback for logging basic attention statistics."""
    
    def __init__(self, log_train: bool = False, log_val: bool = True):
        """
        Initialize the attention stats logger.
        
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
            
            logger = getattr(pl_module, 'logger', None)
            if logger is not None and hasattr(logger, 'experiment'):
                logger.experiment.log_metrics({
                    f"{prefix}/attn_mask_avg_hits_per_query_layer{layer}": float(avg_hits_per_query),
                    f"{prefix}/attn_mask_max_hits_per_query_layer{layer}": float(np.max(hits_per_query)),
                    f"{prefix}/attn_mask_min_hits_per_query_layer{layer}": float(np.min(hits_per_query)),
                    f"{prefix}/attn_mask_std_hits_per_query_layer{layer}": float(np.std(hits_per_query)),
                }, step=step)
            else:
                print(f"[AttentionStatsLogger] Step {step} Layer {layer} - Avg hits per query: {avg_hits_per_query}")
        except Exception as e:
            print(f"[AttentionStatsLogger] Error logging stats: {e}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.log_train:
            return
            
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        if hasattr(model, 'attn_masks_to_log'):
            for mask_info in model.attn_masks_to_log.values():
                mask = mask_info["mask"]
                step = mask_info["step"]
                layer = mask_info["layer"]
                self._log_attention_stats(pl_module, mask, step, layer, "train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.log_val:
            return
            
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        if hasattr(model, 'attn_masks_to_log'):
            for mask_info in model.attn_masks_to_log.values():
                mask = mask_info["mask"]
                step = mask_info["step"]
                layer = mask_info["layer"]
                self._log_attention_stats(pl_module, mask, step, layer, "val")


class PhiAnalysisLogger(Callback):
    """Callback for analyzing phi relationships and creating histograms."""
    
    def __init__(self, max_queries_to_log: int = 3, max_queries_to_process: int = 20, 
                 input_name: str = "hit"):
        """
        Initialize the phi analysis logger.
        
        Args:
            max_queries_to_log: Maximum number of queries to log detailed metrics for
            max_queries_to_process: Maximum number of queries to process for analysis
            input_name: Name of the input field (e.g., "hit", "pix")
        """
        super().__init__()
        self.max_queries_to_log = max_queries_to_log
        self.max_queries_to_process = max_queries_to_process
        self.input_name = input_name

    def _create_phi_histogram(self, phi_hits, q, layer, step, logger):
        """Create and log phi histogram for a query."""
        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            ax.hist(phi_hits, bins=20, color='#4ECDC4', alpha=0.7, edgecolor='#45B7AA', linewidth=1.2)
            ax.set_title(f'Phi Values of Hits for Query {q}', fontsize=12, fontweight='bold', pad=15)
            ax.set_xlabel('Phi Values', fontsize=10)
            ax.set_ylabel('Number of Hits', fontsize=10)
            ax.set_xlim(-np.pi, np.pi)  # Set range from -π to π
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if logger is not None and hasattr(logger, 'experiment'):
                logger.experiment.log_figure(figure_name=f"phi_hits_hist_query{q}_layer{layer}_step{step}", figure=fig, step=step)
        except Exception as e:
            print(f"[PhiAnalysisLogger] Error creating phi histogram for query {q}: {e}")
        finally:
            plt.close(fig)
            del fig

    def _create_distribution_histogram(self, data, title, xlabel, ylabel, color, edge_color, 
                                     figure_name, layer, step, logger, xlim=None):
        """Create and log a distribution histogram."""
        if not data:
            print(f"[PhiAnalysisLogger] No data for histogram: {title}")
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            ax.hist(data, bins=20, color=color, alpha=0.7, edgecolor=edge_color, linewidth=1.2, range=xlim)
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            if xlim:
                ax.set_xlim(xlim)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Add a vertical line at zero for reference
            if xlim and xlim[0] <= 0 <= xlim[1]:
                ax.axvline(x=0, color='#F18F01', linestyle='--', alpha=0.8, linewidth=2, label='Zero Difference')
                ax.legend()
            if logger is not None and hasattr(logger, 'experiment'):
                logger.experiment.log_figure(figure_name=f"{figure_name}_layer{layer}_step{step}", figure=fig, step=step)
        except Exception as e:
            print(f"[PhiAnalysisLogger] Error creating histogram {title}: {e}")
        finally:
            plt.close(fig)
            del fig

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        if not hasattr(model, 'attn_masks_to_log'):
            return
            
        for mask_info in model.attn_masks_to_log.values():
            try:
                mask = mask_info["mask"]
                step = mask_info["step"]
                layer = mask_info["layer"]
                
                # Extract batch data
                inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
                phi_constituents = inputs[f"{self.input_name}_phi"][0].detach().cpu().numpy()  # [num_constituents]
                
                # Get the regressed phi values from the model attribute
                regressed_phi = getattr(model, "last_regressed_phi", None)
                # Get the assigned query phi values from the model
                query_phi = getattr(model, "last_query_phi", None)

                # Collect for histograms
                stds_per_query = []
                diff_regressed_meanhit = []
                diff_regressed_query = []
                diff_meanhit_query = []
                
                mask_np = mask.cpu().numpy() if not isinstance(mask, np.ndarray) else mask
                useful_q_indices = np.where(mask_np.sum(axis=1) > 0)[0]

                # Limit to first queries to avoid excessive logging
                queries_to_process = useful_q_indices[:min(self.max_queries_to_process, len(useful_q_indices))]
                p = 0

                for q in queries_to_process:
                    try:
                        mask_q = mask[q].bool().cpu().numpy()  # [num_constituents]
                        if not mask_q.sum() > 0:
                            print(f"Query {q}: no hits in mask")
                            continue

                        num_selected = mask_q.sum()
                        print(f"Query {q}: Number of selected constituents: {num_selected}")

                        phi_hits = phi_constituents[mask_q]
                        if phi_hits.size > 0:
                            mean_phi = phi_hits.mean()
                            std_phi = phi_hits.std()
                        else:
                            print(f"Query {q}: not enough phi hits")
                            continue
                            
                        reg_phi = regressed_phi[q] if regressed_phi is not None else float('nan')
                        query_phi_val = query_phi[q] if query_phi is not None else float('nan')
                        
                        # Create phi histogram
                        self._create_phi_histogram(phi_hits, q, layer, step, getattr(pl_module, 'logger', None))
                        
                        diff_query_regressed = cyclic_diff(query_phi_val, reg_phi)
                        diff_query_meanhit = cyclic_diff(query_phi_val, mean_phi)

                        print("--------------------------------")
                        print(f"Query {q}:")
                        print("query_phi_val: ", query_phi_val)
                        print("reg_phi: ", reg_phi)
                        print("mean_phi: ", mean_phi)
                        print("std_phi: ", std_phi)
                        print("diff_query_regressed: ", diff_query_regressed)
                        print("diff_query_meanhit: ", diff_query_meanhit)

                        if not np.isnan(std_phi):
                            stds_per_query.append(std_phi)
                        if not np.isnan(mean_phi) and not np.isnan(reg_phi):
                            diff_regressed_meanhit.append(cyclic_diff(reg_phi, mean_phi))
                        if not np.isnan(reg_phi) and not np.isnan(query_phi_val):
                            diff_regressed_query.append(cyclic_diff(reg_phi, query_phi_val))
                        if not np.isnan(mean_phi) and not np.isnan(query_phi_val):
                            diff_meanhit_query.append(cyclic_diff(mean_phi, query_phi_val))
                        if p < self.max_queries_to_log:
                            log_data = {
                                f"val/query{q}_mean_hit_phi_layer{layer}": float(mean_phi),
                                f"val/query{q}_std_hit_phi_layer{layer}": float(std_phi),
                                f"val/query{q}_regressed_phi_layer{layer}": float(reg_phi),
                                f"val/query{q}_query_phi_layer{layer}": float(query_phi_val),
                                f"val/query{q}_query_minus_regressed_phi_layer{layer}": float(diff_query_regressed),
                                f"val/query{q}_query_minus_meanhit_phi_layer{layer}": float(diff_query_meanhit),
                                f"val/query{q}_mean_minus_regressed_phi_layer{layer}": float(cyclic_diff(mean_phi, reg_phi)),
                            }
                            p += 1
                            logger = getattr(pl_module, 'logger', None)
                            if logger is not None and hasattr(logger, 'experiment'):
                                logger.experiment.log_metrics(log_data, step=step)
                            else:
                                print(f"[PhiAnalysisLogger] Step {step} Layer {layer} Query {q} - Query phi: {query_phi_val}, Mean hit phi: {mean_phi}, Std: {std_phi}, Regressed phi: {reg_phi}, Query-Regressed: {diff_query_regressed}, Query-MeanHit: {diff_query_meanhit}, Mean-Regressed: {mean_phi - reg_phi}")
                    except Exception as e:
                        print(f"[PhiAnalysisLogger] Error processing query {q}: {e}")
                        continue
                        
                # Create distribution histograms
                logger = getattr(pl_module, 'logger', None)
                
                # 1. Histogram of std of phi of hits per query
                if stds_per_query:
                    avg_std = np.mean(stds_per_query)
                    if logger is not None and hasattr(logger, 'experiment'):
                        logger.experiment.log_metric(f"val/avg_std_hit_phi_layer{layer}", float(avg_std), step=step)
                    else:
                        print(f"[PhiAnalysisLogger] Step {step} Layer {layer} - Average std of hit phi: {avg_std}")
                    
                    self._create_distribution_histogram(
                        stds_per_query, 'Standard Deviation of Hit Phi Values per Query',
                        'Standard Deviation of Phi Values', 'Number of Queries',
                        '#2E86AB', '#1B4965', 'std_hit_phi_hist', layer, step, logger
                    )
                    
                # 2. Histogram of regressed phi - mean hit phi
                if diff_regressed_meanhit:
                    avg_diff = np.mean(diff_regressed_meanhit)
                    if logger is not None and hasattr(logger, 'experiment'):
                        logger.experiment.log_metric(f"val/avg_regressed_minus_meanhit_phi_layer{layer}", float(avg_diff), step=step)
                    else:
                        print(f"[PhiAnalysisLogger] Step {step} Layer {layer} - Average regressed-meanhit phi diff: {avg_diff}")
                    
                    self._create_distribution_histogram(
                        diff_regressed_meanhit, 'Regressed Phi - Mean Hit Phi Distribution',
                        'Regressed Phi - Mean Hit Phi (Angular Distance)', 'Number of Queries',
                        '#A23B72', '#6B2D5C', 'regressed_minus_meanhit_phi_hist', layer, step, logger, (0, np.pi)
                    )
                else:
                    print("no diff_regressed_meanhit")
                    
                # 3. Histogram of regressed phi - query phi
                if diff_regressed_query:
                    avg_diff = np.mean(diff_regressed_query)
                    if logger is not None and hasattr(logger, 'experiment'):
                        logger.experiment.log_metric(f"val/avg_regressed_minus_query_phi_layer{layer}", float(avg_diff), step=step)
                    else:
                        print(f"[PhiAnalysisLogger] Step {step} Layer {layer} - Average regressed-query phi diff: {avg_diff}")
                    
                    self._create_distribution_histogram(
                        diff_regressed_query, 'Regressed Phi - Query Phi Distribution',
                        'Regressed Phi - Query Phi (Angular Distance)', 'Number of Queries',
                        '#C73E1D', '#8B2635', 'regressed_minus_query_phi_hist', layer, step, logger, (0, np.pi)
                    )
                else:
                    print("no diff_regressed_query")
                    
                # 4. Histogram of mean hit phi - query phi
                if diff_meanhit_query:
                    avg_diff = np.mean(diff_meanhit_query)
                    if logger is not None and hasattr(logger, 'experiment'):
                        logger.experiment.log_metric(f"val/avg_meanhit_minus_query_phi_layer{layer}", float(avg_diff), step=step)
                    else:
                        print(f"[PhiAnalysisLogger] Step {step} Layer {layer} - Average meanhit-query phi diff: {avg_diff}")
                    
                    self._create_distribution_histogram(
                        diff_meanhit_query, 'Mean Hit Phi - Query Phi Distribution',
                        'Mean Hit Phi - Query Phi (Angular Distance)', 'Number of Queries',
                        '#7209B7', '#560BAD', 'meanhit_minus_query_phi_hist', layer, step, logger, (0, np.pi)
                    )
                else:
                    print("no diff_meanhit_query")
                    
            except Exception as e:
                print(f"[PhiAnalysisLogger] Error processing mask info: {e}")
                continue


class PositionalEncodingLogger(Callback):
    """Callback for logging positional encoding analysis."""
    
    def __init__(self, log_train: bool = False, log_val: bool = True):
        """
        Initialize the positional encoding logger.
        
        Args:
            log_train: Whether to log during training
            log_val: Whether to log during validation
        """
        super().__init__()
        self.log_train = log_train
        self.log_val = log_val

    def _log_positional_encodings(self, pl_module, step, layer):
        """Log positional encoding analysis."""
        try:
            model = pl_module.model if hasattr(pl_module, "model") else pl_module
            logger = getattr(pl_module, 'logger', None)
            
            # Get positional encoding data
            query_posenc = getattr(model, "last_query_posenc", None)
            key_posenc = getattr(model, "last_key_posenc", None)
            query_phi = getattr(model, "last_query_phi", None)
            key_phi = getattr(model, "last_key_phi", None)

            print("query_posenc: ", query_posenc)
            print("key_posenc: ", key_posenc)
            print("query_phi: ", query_phi)
            print("key_phi: ", key_phi)

            # Log dot product heatmap (commented out in original)
            # if query_posenc is not None and key_posenc is not None and query_phi is not None and key_phi is not None:
            #     # Implementation here...
            #     pass

            # Log correlation matrix heatmap (commented out in original)
            # if query_posenc is not None and key_posenc is not None:
            #     # Implementation here...
            #     pass
                
        except Exception as e:
            print(f"[PositionalEncodingLogger] Error logging positional encodings: {e}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.log_train:
            return
            
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        if hasattr(model, 'attn_masks_to_log'):
            for mask_info in model.attn_masks_to_log.values():
                step = mask_info["step"]
                layer = mask_info["layer"]
                self._log_positional_encodings(pl_module, step, layer)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.log_val:
            return
            
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        if hasattr(model, 'attn_masks_to_log'):
            for mask_info in model.attn_masks_to_log.values():
                step = mask_info["step"]
                layer = mask_info["layer"]
                self._log_positional_encodings(pl_module, step, layer)


# Legacy class for backward compatibility
class AttnMaskLogger(Callback):
    """Legacy callback that combines all attention mask logging functionality."""
    
    def __init__(self, max_queries_to_log: int = 3, max_queries_to_process: int = 200):
        """
        Initialize the legacy attention mask logger callback.
        
        Args:
            max_queries_to_log: Maximum number of queries to log detailed metrics for
            max_queries_to_process: Maximum number of queries to process for analysis
        """
        super().__init__()
        self.visualizer = AttentionMaskVisualizer()
        self.stats_logger = AttentionStatsLogger()
        self.phi_analyzer = PhiAnalysisLogger(max_queries_to_log, max_queries_to_process)
        self.posenc_logger = PositionalEncodingLogger()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Delegate to individual callbacks
        self.visualizer.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.stats_logger.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.phi_analyzer.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.posenc_logger.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Delegate to individual callbacks
        self.visualizer.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.stats_logger.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.phi_analyzer.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.posenc_logger.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)