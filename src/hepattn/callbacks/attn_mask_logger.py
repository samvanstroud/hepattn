import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import sys

def cyclic_diff(phi1, phi2):
    """
    Compute the cyclic difference between two phi values.
    Returns the smallest angular difference in the range [0, π].
    """
    diff = phi1 - phi2
    # Wrap to [-π, π] first
    wrapped_diff = np.arctan2(np.sin(diff), np.cos(diff))
    # Take absolute value to get smallest angular distance [0, π]
    return np.abs(wrapped_diff)

class AttnMaskLogger(Callback):
    def __init__(self):
        super().__init__()
        self._reset_epoch_data()

    def _reset_epoch_data(self):
        self.stds_per_query = []
        self.diff_regressed_meanhit = []
        self.diff_regressed_query = []
        self.diff_meanhit_query = []

    def _log_attention_mask(self, pl_module, mask, step, layer, prefix="local_ca_mask"):
        """Helper method to create and log attention mask figures."""
        fig, ax = plt.subplots(constrained_layout=True, dpi=300)
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
        plt.close(fig)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):

        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        if hasattr(model, 'attn_masks_to_log'):
            for mask_info in model.attn_masks_to_log.values():
                mask = mask_info["mask"]
                step = mask_info["step"]
                layer = mask_info["layer"]
                self._log_attention_mask(pl_module, mask, step, layer, "local_ca_mask_val")

                # --- New: Log number of hits per mask and average ---
                # mask: shape [num_queries, num_constituents], dtype=bool or int
                hits_per_query = mask.sum(dim=1).cpu().numpy()  # shape: [num_queries]
                avg_hits_per_query = hits_per_query.mean()
                logger = getattr(pl_module, 'logger', None)
                if logger is not None and hasattr(logger, 'experiment'):
                    logger.experiment.log_metrics({
                        f"val/attn_mask_avg_hits_per_query_layer{layer}": float(avg_hits_per_query),
                        f"val/attn_mask_max_hits_per_query_layer{layer}": float(np.max(hits_per_query)),
                        f"val/attn_mask_min_hits_per_query_layer{layer}": float(np.min(hits_per_query)),
                        f"val/attn_mask_std_hits_per_query_layer{layer}": float(np.std(hits_per_query)),
                    }, step=step)
                else:
                    print(f"[AttnMaskLogger] Step {step} Layer {layer} - Hits per query: {hits_per_query}")
                    print(f"[AttnMaskLogger] Step {step} Layer {layer} - Avg hits per query: {avg_hits_per_query}")
                    
                # Set these to match your data/model:
                input_name = "hit"  # e.g. "pix" or "hit"
                inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
                phi_constituents = inputs[f"{input_name}_phi"][0].detach().cpu().numpy()  # [num_constituents]
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

                # Limit to first 20 queries to avoid excessive logging
                max_queries_to_log = 2
                max_queries_to_process = 50
                queries_to_process = useful_q_indices[:min(max_queries_to_process, len(useful_q_indices))]
                # queries_to_process = useful_q_indices
                p=0

                for q in queries_to_process:
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
                    
                    # log histogram of phi values
                    fig, ax = plt.subplots(figsize=(8, 5))
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
                    plt.close(fig)
                    
                    diff_query_regressed = cyclic_diff(query_phi_val, reg_phi)
                    diff_query_meanhit = cyclic_diff(query_phi_val, mean_phi)

                    # print("--------------------------------")
                    # print(f"Query {q}:")
                    # print("query_phi_val: ", query_phi_val)
                    # print("reg_phi: ", reg_phi)
                    # print("mean_phi: ", mean_phi)
                    # print("std_phi: ", std_phi)
                    # print("diff_query_regressed: ", diff_query_regressed)
                    # print("diff_query_meanhit: ", diff_query_meanhit)

                    if not np.isnan(std_phi):
                        stds_per_query.append(std_phi)
                    if not np.isnan(mean_phi) and not np.isnan(reg_phi):
                        diff_regressed_meanhit.append(cyclic_diff(reg_phi, mean_phi))
                    if not np.isnan(reg_phi) and not np.isnan(query_phi_val):
                        diff_regressed_query.append(cyclic_diff(reg_phi, query_phi_val))
                    if not np.isnan(mean_phi) and not np.isnan(query_phi_val):
                        diff_meanhit_query.append(cyclic_diff(mean_phi, query_phi_val))
                    if p<max_queries_to_log:
                        log_data = {
                            f"val/query{q}_mean_hit_phi_layer{layer}": float(mean_phi),
                            f"val/query{q}_std_hit_phi_layer{layer}": float(std_phi),
                            f"val/query{q}_regressed_phi_layer{layer}": float(reg_phi),
                            f"val/query{q}_query_phi_layer{layer}": float(query_phi_val),
                            f"val/query{q}_query_minus_regressed_phi_layer{layer}": float(diff_query_regressed),
                            f"val/query{q}_query_minus_meanhit_phi_layer{layer}": float(diff_query_meanhit),
                            f"val/query{q}_mean_minus_regressed_phi_layer{layer}": float(cyclic_diff(mean_phi, reg_phi)),
                        }
                        p+=1
                        if logger is not None and hasattr(logger, 'experiment'):
                            logger.experiment.log_metrics(log_data, step=step)
                        else:
                            print(f"[AttnMaskLogger] Step {step} Layer {layer} Query {q} - Query phi: {query_phi_val}, Mean hit phi: {mean_phi}, Std: {std_phi}, Regressed phi: {reg_phi}, Query-Regressed: {diff_query_regressed}, Query-MeanHit: {diff_query_meanhit}, Mean-Regressed: {mean_phi - reg_phi}")
                # --- Plot and log histograms ---
                # 1. Histogram of std of phi of hits per query
                if stds_per_query:
                    # Log average std
                    avg_std = np.mean(stds_per_query)
                    if logger is not None and hasattr(logger, 'experiment'):
                        logger.experiment.log_metric(f"val/avg_std_hit_phi_layer{layer}", float(avg_std), step=step)
                    else:
                        print(f"[AttnMaskLogger] Step {step} Layer {layer} - Average std of hit phi: {avg_std}")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(stds_per_query, bins=20, color='#2E86AB', alpha=0.7, edgecolor='#1B4965', linewidth=1.2)
                    ax.set_title('Standard Deviation of Hit Phi Values per Query', fontsize=14, fontweight='bold', pad=20)
                    ax.set_xlabel('Standard Deviation of Phi Values', fontsize=12)
                    ax.set_ylabel('Number of Queries', fontsize=12)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    if logger is not None and hasattr(logger, 'experiment'):
                        logger.experiment.log_figure(figure_name=f"std_hit_phi_hist_layer{layer}_step{step}", figure=fig, step=step)
                    plt.close(fig)
                # 2. Histogram of regressed phi - mean hit phi
                if diff_regressed_meanhit:
                    # Log average difference
                    avg_diff_regressed_meanhit = np.mean(diff_regressed_meanhit)
                    if logger is not None and hasattr(logger, 'experiment'):
                        logger.experiment.log_metric(f"val/avg_regressed_minus_meanhit_phi_layer{layer}", float(avg_diff_regressed_meanhit), step=step)
                    else:
                        print(f"[AttnMaskLogger] Step {step} Layer {layer} - Average regressed-meanhit phi diff: {avg_diff_regressed_meanhit}")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(diff_regressed_meanhit, bins=20, color='#A23B72', alpha=0.7, edgecolor='#6B2D5C', linewidth=1.2, range=(0, np.pi))
                    ax.set_title('Regressed Phi - Mean Hit Phi Distribution', fontsize=14, fontweight='bold', pad=20)
                    ax.set_xlabel('Regressed Phi - Mean Hit Phi (Angular Distance)', fontsize=12)
                    ax.set_ylabel('Number of Queries', fontsize=12)
                    ax.set_xlim(0, np.pi)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    # Add a vertical line at zero for reference
                    ax.axvline(x=0, color='#F18F01', linestyle='--', alpha=0.8, linewidth=2, label='Zero Difference')
                    ax.legend()
                    if logger is not None and hasattr(logger, 'experiment'):
                        logger.experiment.log_figure(figure_name=f"regressed_minus_meanhit_phi_hist_layer{layer}_step{step}", figure=fig, step=step)
                    plt.close(fig)
                else:
                    print("no diff_regressed_meanhit")
                # 3. Histogram of regressed phi - query phi
                if diff_regressed_query:
                    # Log average difference
                    avg_diff_regressed_query = np.mean(diff_regressed_query)
                    if logger is not None and hasattr(logger, 'experiment'):
                        logger.experiment.log_metric(f"val/avg_regressed_minus_query_phi_layer{layer}", float(avg_diff_regressed_query), step=step)
                    else:
                        print(f"[AttnMaskLogger] Step {step} Layer {layer} - Average regressed-query phi diff: {avg_diff_regressed_query}")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(diff_regressed_query, bins=20, color='#C73E1D', alpha=0.7, edgecolor='#8B2635', linewidth=1.2, range=(0, np.pi))
                    ax.set_title('Regressed Phi - Query Phi Distribution', fontsize=14, fontweight='bold', pad=20)
                    ax.set_xlabel('Regressed Phi - Query Phi (Angular Distance)', fontsize=12)
                    ax.set_ylabel('Number of Queries', fontsize=12)
                    ax.set_xlim(0, np.pi)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    # Add a vertical line at zero for reference
                    ax.axvline(x=0, color='#F18F01', linestyle='--', alpha=0.8, linewidth=2, label='Zero Difference')
                    ax.legend()
                    if logger is not None and hasattr(logger, 'experiment'):
                        logger.experiment.log_figure(figure_name=f"regressed_minus_query_phi_hist_layer{layer}_step{step}", figure=fig, step=step)
                    plt.close(fig)
                else:
                    print("no diff_regressed_query")
                # 4. Histogram of mean hit phi - query phi
                if diff_meanhit_query:
                    # Log average difference
                    avg_diff_meanhit_query = np.mean(diff_meanhit_query)
                    if logger is not None and hasattr(logger, 'experiment'):
                        logger.experiment.log_metric(f"val/avg_meanhit_minus_query_phi_layer{layer}", float(avg_diff_meanhit_query), step=step)
                    else:
                        print(f"[AttnMaskLogger] Step {step} Layer {layer} - Average meanhit-query phi diff: {avg_diff_meanhit_query}")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(diff_meanhit_query, bins=20, color='#7209B7', alpha=0.7, edgecolor='#560BAD', linewidth=1.2, range=(0, np.pi))
                    ax.set_title('Mean Hit Phi - Query Phi Distribution', fontsize=14, fontweight='bold', pad=20)
                    ax.set_xlabel('Mean Hit Phi - Query Phi (Angular Distance)', fontsize=12)
                    ax.set_ylabel('Number of Queries', fontsize=12)
                    ax.set_xlim(0, np.pi)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    # Add a vertical line at zero for reference
                    ax.axvline(x=0, color='#F18F01', linestyle='--', alpha=0.8, linewidth=2, label='Zero Difference')
                    ax.legend()
                    if logger is not None and hasattr(logger, 'experiment'):
                        logger.experiment.log_figure(figure_name=f"meanhit_minus_query_phi_hist_layer{layer}_step{step}", figure=fig, step=step)
                    plt.close(fig)
                else:
                    print("no diff_meanhit_query")
                # --- End histogram logging ---

                # --- Log dot product heatmap ---
                query_posenc = getattr(model, "last_query_posenc", None)
                key_posenc = getattr(model, "last_key_posenc", None)
                query_phi = getattr(model, "last_query_phi", None)
                key_phi = getattr(model, "last_key_phi", None)

                print("query_posenc: ", query_posenc)
                print("key_posenc: ", key_posenc)
                print("query_phi: ", query_phi)
                print("key_phi: ", key_phi)

                # if query_posenc is not None and key_posenc is not None and query_phi is not None and key_phi is not None:
                #     # Handle key_phi if it has batch dimension
                #     if key_phi.ndim > 1:
                #         key_phi = key_phi[0]  # Take first batch element
                    
                #     # Ensure both are 1D arrays
                #     if query_posenc.ndim > 1:
                #         query_posenc = query_posenc.flatten()
                #     if key_posenc.ndim > 1:
                #         key_posenc = key_posenc.flatten()
                    
                #     # Compute outer product to create 2D matrix
                #     # This creates a matrix where each element (i,j) is query_posenc[i] * key_posenc[j]
                #     dot_products = np.outer(query_posenc, key_posenc)  # [num_queries, num_keys]
                    
                #     print(f"Dot products shape: {dot_products.shape}")
                    
                #     fig, ax = plt.subplots(figsize=(8, 6))
                #     im = ax.imshow(dot_products, aspect='auto', origin='lower',
                #                     extent=[key_phi.min(), key_phi.max(), query_phi.min(), query_phi.max()])
                #     ax.set_xlabel('Key phi')
                #     ax.set_ylabel('Query phi')
                #     ax.set_title('Dot product of positional encodings')
                #     fig.colorbar(im, ax=ax, label='Dot product')
                #     if logger is not None and hasattr(logger, 'experiment'):
                #         logger.experiment.log_figure(figure_name=f"posenc_dotprod_heatmap_layer{layer}_step{step}", figure=fig, step=step)
                #     plt.close(fig)
                # else:
                #     print("Missing positional encoding data for heatmap:")
                #     print(f"  query_posenc: {query_posenc is not None}")
                #     print(f"  key_posenc: {key_posenc is not None}")
                #     print(f"  query_phi: {query_phi is not None}")
                #     print(f"  key_phi: {key_phi is not None}")
                # # --- End dot product heatmap logging ---

                # # --- Log correlation matrix heatmap ---
                # if query_posenc is not None and key_posenc is not None:
                #     input_space_min = -3
                #     input_space_max = 3
                #     num_samples = 100

                #     # Convert to tensors if they're numpy arrays
                #     if isinstance(query_posenc, np.ndarray):
                #         query_posenc_tensor = torch.from_numpy(query_posenc)
                #     else:
                #         query_posenc_tensor = query_posenc
                    
                #     if isinstance(key_posenc, np.ndarray):
                #         key_posenc_tensor = torch.from_numpy(key_posenc)
                #     else:
                #         key_posenc_tensor = key_posenc

                #     # Create correlation matrix
                #     correlation_matrix = torch.matmul(query_posenc_tensor.T, key_posenc_tensor) / query_posenc_tensor.shape[0]
                    
                #     # Convert back to numpy for plotting
                #     correlation_matrix_np = correlation_matrix.numpy()
                    
                #     # Create figure and plot
                #     fig, ax = plt.subplots(figsize=(8, 6))
                #     im = ax.imshow(correlation_matrix_np, aspect='auto', origin='lower')
                    
                #     # Add colorbar
                #     cbar = plt.colorbar(im, ax=ax, label='Correlation')
                    
                #     # Set tick labels
                #     num_tick_labels = 9
                #     x_ticks_labels = torch.linspace(input_space_min, input_space_max, num_tick_labels).tolist()
                #     x_ticks = torch.linspace(0, correlation_matrix_np.shape[1]-1, num_tick_labels).tolist()
                #     y_ticks = torch.linspace(0, correlation_matrix_np.shape[0]-1, num_tick_labels).tolist()
                    
                #     ax.set_xticks(x_ticks)
                #     ax.set_xticklabels([f'{x:.1f}' for x in x_ticks_labels])
                #     ax.set_yticks(y_ticks)
                #     ax.set_yticklabels([f'{x:.1f}' for x in x_ticks_labels])
                    
                #     ax.set_xlabel('Key Phi')
                #     ax.set_ylabel('Query Phi')
                #     ax.set_title('Positional Encoding Correlation Matrix')
                    
                #     # Log to Comet
                #     if logger is not None and hasattr(logger, 'experiment'):
                #         logger.experiment.log_figure(figure_name=f"posenc_correlation_matrix_layer_sam{layer}_step{step}", figure=fig, step=step)
                    
                #     plt.close(fig)
                # # --- End correlation matrix logging ---

            # Clear after logging
            delattr(model, 'attn_masks_to_log')


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        if hasattr(model, 'attn_masks_to_log'):
            for mask_info in model.attn_masks_to_log.values():
                mask = mask_info["mask"]
                step = mask_info["step"]
                layer = mask_info["layer"]
                self._log_attention_mask(pl_module, mask, step, layer, "local_ca_mask")
            # Clear after logging
            delattr(model, 'attn_masks_to_log')