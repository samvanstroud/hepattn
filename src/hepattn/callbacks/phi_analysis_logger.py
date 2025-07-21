import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
import numpy as np


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


class PhiAnalysisLogger(Callback):
    """Callback for analyzing phi relationships and creating histograms."""
    
    def __init__(self, max_queries_to_log: int = 1, max_queries_to_process: int = 20, 
                 input_name: str = "hit", regressed_phi: bool = False, queryPE: bool = False, log_every_n_steps: int = 100):
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
        self.regressed_phi = regressed_phi
        self.queryPE = queryPE
        self.log_every_n_steps = log_every_n_steps

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

    def on_validation_epoch_end(self, trainer, pl_module):
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        if not hasattr(model, 'attn_masks_to_log'):
            return

        for mask_info in model.attn_masks_to_log.values():
            try:
                step = mask_info["step"]
                if step % self.log_every_n_steps != 0:
                    continue
                mask = mask_info["mask"]
                layer = mask_info["layer"]

                phi_constituents = getattr(model, "last_key_phi", None)
                mask_np = mask.cpu().numpy() if not isinstance(mask, np.ndarray) else mask
                phi_constituents_np = phi_constituents.cpu().numpy() if hasattr(phi_constituents, 'cpu') else phi_constituents

                useful_q_indices = np.where(mask_np.sum(axis=1) > 0)[0]
                queries_to_process = useful_q_indices[:min(self.max_queries_to_process, len(useful_q_indices))]
                if len(queries_to_process) == 0:
                    continue

                selected_phi = np.where(mask_np[queries_to_process], phi_constituents_np, np.nan)
                mean_phi = np.nanmean(selected_phi, axis=1)
                std_phi = np.nanstd(selected_phi, axis=1)

                def cyclic_diff_vec(phi1, phi2):
                    diff = phi1 - phi2
                    wrapped_diff = np.arctan2(np.sin(diff), np.cos(diff))
                    return np.abs(wrapped_diff)

                # Prepare optional arrays
                reg_phi_sel = None
                query_phi_sel = None
                diff_regressed_meanhit = None
                diff_meanhit_query = None
                diff_regressed_query = None

                if self.regressed_phi:
                    regressed_phi = getattr(model, "last_regressed_phi", None)
                    reg_phi_sel = regressed_phi[queries_to_process]
                    diff_regressed_meanhit = cyclic_diff_vec(reg_phi_sel, mean_phi)
                if self.queryPE:
                    query_phi = getattr(model, "last_query_phi", None)
                    query_phi_np = query_phi.cpu().numpy() if hasattr(query_phi, 'cpu') else query_phi
                    query_phi_sel = query_phi_np[queries_to_process]
                    diff_meanhit_query = cyclic_diff_vec(mean_phi, query_phi_sel)
                if self.regressed_phi and self.queryPE:
                    diff_regressed_query = cyclic_diff_vec(reg_phi_sel, query_phi_sel)

                # Per-query logging/plotting for first N queries
                logger = getattr(pl_module, 'logger', None)
                for i, q in enumerate(queries_to_process[:self.max_queries_to_log]):
                    log_data = {
                        f"val/query{q}_mean_hit_phi_layer{layer}": float(mean_phi[i]),
                        f"val/query{q}_std_hit_phi_layer{layer}": float(std_phi[i]),
                    }
                    if self.regressed_phi:
                        log_data[f"val/query{q}_regressed_phi_layer{layer}"] = float(reg_phi_sel[i])
                        log_data[f"val/query{q}_mean_minus_regressed_phi_layer{layer}"] = float(diff_regressed_meanhit[i])
                    if self.queryPE:
                        log_data[f"val/query{q}_query_phi_layer{layer}"] = float(query_phi_sel[i])
                        log_data[f"val/query{q}_query_minus_meanhit_phi_layer{layer}"] = float(diff_meanhit_query[i])
                        if self.regressed_phi:
                            log_data[f"val/query{q}_query_minus_regressed_phi_layer{layer}"] = float(diff_regressed_query[i])
                    if logger is not None and hasattr(logger, 'experiment'):
                        logger.experiment.log_metrics(log_data, step=step)
                    else:
                        print(f"[PhiAnalysisLogger] Step {step} Layer {layer} Query {q} - {log_data}")

                    # Plot phi histogram for this query
                    phi_hits = selected_phi[i][~np.isnan(selected_phi[i])]
                    self._create_phi_histogram(phi_hits, q, layer, step, logger)

                # Distribution histograms and averages
                if len(std_phi) > 0:
                    avg_std = np.mean(std_phi)
                    if logger is not None and hasattr(logger, 'experiment'):
                        logger.experiment.log_metric(f"val/avg_std_hit_phi_layer{layer}", float(avg_std), step=step)
                    else:
                        print(f"[PhiAnalysisLogger] Step {step} Layer {layer} - Average std of hit phi: {avg_std}")
                    self._create_distribution_histogram(
                        std_phi, 'Standard Deviation of Hit Phi Values per Query',
                        'Standard Deviation of Phi Values', 'Number of Queries',
                        '#2E86AB', '#1B4965', 'std_hit_phi_hist', layer, step, logger
                    )

                if self.regressed_phi and diff_regressed_meanhit is not None and len(diff_regressed_meanhit) > 0:
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

                if self.regressed_phi and self.queryPE and diff_regressed_query is not None and len(diff_regressed_query) > 0:
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

                if self.queryPE and diff_meanhit_query is not None and len(diff_meanhit_query) > 0:
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