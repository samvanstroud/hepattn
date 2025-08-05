# #!/usr/bin/env python3
# """
# Simplified positional encoding analysis callback for MaskFormer models.
# """

# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from lightning.pytorch.callbacks import Callback
# from pathlib import Path
# import warnings
# from typing import Dict


# def create_pos_enc_visualizations(
#     hit_posencs: torch.Tensor,
#     query_posencs: torch.Tensor,
#     hit_phi: torch.Tensor = None,
#     query_phi: torch.Tensor = None,
# ) -> Dict[str, plt.Figure]:
#     """
#     Create comprehensive positional encoding visualizations.
    
#     Args:
#         hit_posencs: Hit positional encodings [num_hits, dim]
#         query_posencs: Query positional encodings [num_queries, dim]
#         hit_phi: Phi values for hits (optional, for sorting)
#         query_phi: Phi values for queries (optional, for sorting)
        
#     Returns:
#         Dictionary of created figures
#     """

#     figures = {}
    
#     # Sort by phi values if available
#     if hit_phi is not None:
#         # Sort hits by phi
#         hit_sort_idx = torch.argsort(hit_phi)
#         hit_posencs_sorted = hit_posencs[hit_sort_idx]
#         hit_phi_sorted = hit_phi[hit_sort_idx]
#     else:
#         hit_posencs_sorted = hit_posencs
#         hit_phi_sorted = None
    
#     if query_phi is not None:
#         # Sort queries by phi
#         query_sort_idx = torch.argsort(query_phi)
#         query_posencs_sorted = query_posencs[query_sort_idx]
#         query_phi_sorted = query_phi[query_sort_idx]
#     else:
#         query_posencs_sorted = query_posencs
#         query_phi_sorted = None
    
#     # 1. Query positional encoding matrix
#     fig, ax = plt.subplots(figsize=(12, 8))
#     im = ax.imshow(query_posencs_sorted.T.numpy(), origin='lower', aspect='auto', cmap='viridis')
#     ax.set_title('Query Positional Encoding Matrix (Sorted by Phi)', fontsize=14, fontweight='bold')
#     ax.set_xlabel('Query Index (Sorted by Phi)', fontsize=12)
#     ax.set_ylabel('Positional Encoding Dimension', fontsize=12)
#     plt.colorbar(im, ax=ax)
    
#     figures['query_posenc'] = fig
    
#     # 2. Hit positional encoding matrix
#     fig, ax = plt.subplots(figsize=(12, 8))
#     im = ax.imshow(hit_posencs_sorted.T.numpy(), origin='lower', aspect='auto', cmap='viridis')
#     ax.set_title('Hit Positional Encoding Matrix (Sorted by Phi)', fontsize=14, fontweight='bold')
#     ax.set_xlabel('Hit Index (Sorted by Phi)', fontsize=12)
#     ax.set_ylabel('Positional Encoding Dimension', fontsize=12)
#     plt.colorbar(im, ax=ax)
    
#     figures['hit_posenc'] = fig
    
#     # 3. Dot product matrix (using sorted encodings)
#     dot_product = torch.matmul(query_posencs_sorted, hit_posencs_sorted.T) / hit_posencs_sorted.shape[0]
    
#     fig, ax = plt.subplots(figsize=(12, 8))
#     im = ax.imshow(dot_product.numpy(), origin='lower', aspect='auto')
#     ax.set_title('Query-Hit Dot Product Matrix (Sorted by Phi)', fontsize=14, fontweight='bold')
#     ax.set_xlabel('Hit Index (Sorted by Phi)', fontsize=12)
#     ax.set_ylabel('Query Index (Sorted by Phi)', fontsize=12)
    
#     plt.colorbar(im, ax=ax)
    
#     figures['dot_product'] = fig
    
#     # 4. Phi distribution plots
#     if hit_phi is not None or query_phi is not None:
#         fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#         fig.suptitle('Phi Distribution Analysis', fontsize=16, fontweight='bold')
        
#         # Query phi distribution
#         if query_phi is not None:
#             # Histogram
#             axes[0, 0].hist(query_phi.numpy(), bins=50, alpha=0.7, edgecolor='black', color='skyblue')
#             axes[0, 0].set_xlabel('Query Phi (radians)')
#             axes[0, 0].set_ylabel('Frequency')
#             axes[0, 0].set_title('Query Phi Distribution')
#             axes[0, 0].grid(True, alpha=0.3)
        
#         # Hit phi distribution
#         if hit_phi is not None:
#             # Histogram
#             axes[1, 0].hist(hit_phi.numpy(), bins=50, alpha=0.7, edgecolor='black', color='lightcoral')
#             axes[1, 0].set_xlabel('Hit Phi (radians)')
#             axes[1, 0].set_ylabel('Frequency')
#             axes[1, 0].set_title('Hit Phi Distribution')
#             axes[1, 0].grid(True, alpha=0.3)
        
#         # If both are available, add comparison
#         if hit_phi is not None and query_phi is not None:
#             # Overlay both distributions
#             axes[0, 1].hist(query_phi.numpy(), bins=50, alpha=0.7, label='Queries', color='skyblue')
#             axes[0, 1].hist(hit_phi.numpy(), bins=50, alpha=0.7, label='Hits', color='lightcoral')
#             axes[0, 1].set_xlabel('Phi (radians)')
#             axes[0, 1].set_ylabel('Frequency')
#             axes[0, 1].set_title('Query vs Hit Phi Distribution')
#             axes[0, 1].legend()
#             axes[0, 1].grid(True, alpha=0.3)
            
#             # Box plot comparison
#             axes[1, 1].boxplot([query_phi.numpy(), hit_phi.numpy()], labels=['Queries', 'Hits'])
#             axes[1, 1].set_ylabel('Phi (radians)')
#             axes[1, 1].set_title('Phi Distribution Comparison')
#             axes[1, 1].grid(True, alpha=0.3)
#         else:
#             # If only one type is available, add circular histograms
#             if query_phi is not None:
#                 # Circular histogram (polar plot) for queries
#                 ax_polar = fig.add_subplot(2, 2, 2, projection='polar')
#                 ax_polar.hist(query_phi.numpy(), bins=36, alpha=0.7, edgecolor='black', color='skyblue')
#                 ax_polar.set_title('Query Phi Distribution (Circular)', pad=20)
            
#             if hit_phi is not None:
#                 # Circular histogram (polar plot) for hits
#                 ax_polar2 = fig.add_subplot(2, 2, 4, projection='polar')
#                 ax_polar2.hist(hit_phi.numpy(), bins=36, alpha=0.7, edgecolor='black', color='lightcoral')
#                 ax_polar2.set_title('Hit Phi Distribution (Circular)', pad=20)
                
#         plt.tight_layout()
#         figures['phi_distributions'] = fig
    
#     # 5. Sorted phi plots to confirm sorting
#     if hit_phi_sorted is not None or query_phi_sorted is not None:
#         fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#         fig.suptitle('Sorted Phi Values Analysis', fontsize=16, fontweight='bold')
        
#         # Query phi sorted
#         if query_phi_sorted is not None:
#             # Line plot of sorted query phi
#             axes[0, 0].plot(range(len(query_phi_sorted)), query_phi_sorted.numpy(), 
#                            color='skyblue', linewidth=2, marker='o', markersize=3)
#             axes[0, 0].set_xlabel('Query Index (Sorted)')
#             axes[0, 0].set_ylabel('Query Phi (radians)')
#             axes[0, 0].set_title('Query Phi Values (Sorted)')
#             axes[0, 0].grid(True, alpha=0.3)
            
#             # Check if sorting is monotonic
#             is_monotonic = torch.all(torch.diff(query_phi_sorted) >= 0)
#             axes[0, 0].text(0.02, 0.98, f'Monotonic: {is_monotonic}', 
#                            transform=axes[0, 0].transAxes, verticalalignment='top',
#                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
#         # Hit phi sorted
#         if hit_phi_sorted is not None:
#             # Line plot of sorted hit phi
#             axes[1, 0].plot(range(len(hit_phi_sorted)), hit_phi_sorted.numpy(), 
#                            color='lightcoral', linewidth=2, marker='o', markersize=3)
#             axes[1, 0].set_xlabel('Hit Index (Sorted)')
#             axes[1, 0].set_ylabel('Hit Phi (radians)')
#             axes[1, 0].set_title('Hit Phi Values (Sorted)')
#             axes[1, 0].grid(True, alpha=0.3)
            
#             # Check if sorting is monotonic
#             is_monotonic = torch.all(torch.diff(hit_phi_sorted) >= 0)
#             axes[1, 0].text(0.02, 0.98, f'Monotonic: {is_monotonic}', 
#                            transform=axes[1, 0].transAxes, verticalalignment='top',
#                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
#         # Comparison of sorted values
#         if hit_phi_sorted is not None and query_phi_sorted is not None:
#             # Overlay sorted phi values
#             axes[0, 1].plot(range(len(query_phi_sorted)), query_phi_sorted.numpy(), 
#                            color='skyblue', linewidth=2, label='Queries', marker='o', markersize=3)
#             axes[0, 1].plot(range(len(hit_phi_sorted)), hit_phi_sorted.numpy(), 
#                            color='lightcoral', linewidth=2, label='Hits', marker='o', markersize=3)
#             axes[0, 1].set_xlabel('Index (Sorted)')
#             axes[0, 1].set_ylabel('Phi (radians)')
#             axes[0, 1].set_title('Sorted Phi Values Comparison')
#             axes[0, 1].legend()
#             axes[0, 1].grid(True, alpha=0.3)
            
#             # Difference between sorted query and hit phi ranges
#             query_range = query_phi_sorted.max() - query_phi_sorted.min()
#             hit_range = hit_phi_sorted.max() - hit_phi_sorted.min()
#             axes[1, 1].bar(['Query Range', 'Hit Range'], [query_range.item(), hit_range.item()], 
#                           color=['skyblue', 'lightcoral'])
#             axes[1, 1].set_ylabel('Phi Range (radians)')
#             axes[1, 1].set_title('Phi Value Ranges')
#             axes[1, 1].grid(True, alpha=0.3)
            
#             # Add text with range information
#             axes[1, 1].text(0.02, 0.98, f'Query: [{query_phi_sorted.min():.3f}, {query_phi_sorted.max():.3f}]\nHit: [{hit_phi_sorted.min():.3f}, {hit_phi_sorted.max():.3f}]', 
#                            transform=axes[1, 1].transAxes, verticalalignment='top',
#                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
#         plt.tight_layout()
#         figures['sorted_phi_analysis'] = fig
    
#     return figures


# class PositionalEncodingAnalysisLogger(Callback):
#     """Simple callback for analyzing positional encodings from MaskFormer models.
    
#     This callback logs positional encoding analysis once per training, accessing
#     the positional encodings directly from the model's instance variables.
#     """
    
#     def __init__(self, output_dir: str = "pos_enc_analysis"):
#         """
#         Initialize the positional encoding analysis logger.
        
#         Args:
#             output_dir: Directory to save analysis outputs
#         """
#         super().__init__()
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(exist_ok=True, parents=True)
        
#         # Track if we've logged this epoch
#         self._logged_this_epoch = False
        
#     def _create_pos_enc_visualizations(self, hit_posencs, query_posencs, logger, hit_phi=None, query_phi=None):
#         """Create and log positional encoding visualizations."""
        
#         if hit_posencs is None or query_posencs is None:
#             warnings.warn("No positional encodings available for visualization")
#             return
        
#         # Create visualizations with phi sorting
#         figures = create_pos_enc_visualizations(hit_posencs, query_posencs, hit_phi, query_phi)
        
#         # Log to experiment
#         if logger is not None and hasattr(logger, 'experiment'):
#             for name, fig in figures.items():
#                 logger.experiment.log_figure(
#                     figure_name=f"pos_enc_{name}",
#                     figure=fig,
#                 )
#             plt.close()
    
#     def on_train_start(self, trainer, pl_module):
#         """Log positional encoding analysis at the start of training."""
        
#         # Check if model has phi_analysis enabled
#         if not hasattr(pl_module.model, 'phi_analysis') or not pl_module.model.phi_analysis:
#             warnings.warn("Model has phi_analysis=False. Set phi_analysis=True in MaskFormer for positional encoding analysis.")
#             return
        
#         # Check if model has stored positional encodings
#         if not hasattr(pl_module.model, 'last_query_posenc') or not hasattr(pl_module.model, 'last_key_posenc'):
#             warnings.warn("Model does not have stored positional encodings. Run a forward pass first.")
#             return
        
#         # Get positional encodings from model and ensure consistent dtype
#         hit_posencs = torch.tensor(pl_module.model.last_key_posenc_sorted, dtype=torch.float16)
#         query_posencs = torch.tensor(pl_module.model.last_query_posenc, dtype=torch.float16)
        
#         # Get phi values if available
#         hit_phi = None
#         query_phi = None
#         if hasattr(pl_module.model, 'last_key_phi'):
#             hit_phi = torch.tensor(pl_module.model.last_key_phi, dtype=torch.float32)
#         else:
#             raise ValueError("No key phi")
#         if hasattr(pl_module.model, 'last_query_phi'):
#             query_phi = torch.tensor(pl_module.model.last_query_phi, dtype=torch.float32)
        
#         logger = getattr(pl_module, 'logger', None)
        
#         try:
#             self._create_pos_enc_visualizations(hit_posencs, query_posencs, logger, hit_phi, query_phi)
#             self._logged_this_epoch = True
            
#         except Exception as e:
#             warnings.warn(f"Error in positional encoding analysis: {e}")

#!/usr/bin/env python3

#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Simplified positional encoding analysis callback for MaskFormer models.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from pathlib import Path
import warnings
from typing import Dict


def create_pos_enc_visualizations(
    hit_posencs: torch.Tensor,
    query_posencs: torch.Tensor,
    hit_phi: torch.Tensor = None,
    query_phi: torch.Tensor = None,
) -> Dict[str, plt.Figure]:
    """
    Create comprehensive positional encoding visualizations.
    
    Args:
        hit_posencs: Hit positional encodings [num_hits, dim]
        query_posencs: Query positional encodings [num_queries, dim]
        hit_phi: Phi values for hits (optional, for sorting)
        query_phi: Phi values for queries (optional, for sorting)
        
    Returns:
        Dictionary of created figures
    """

    figures = {}
    
    # Sort by phi values if available
    if hit_phi is not None:
        # Sort hits by phi
        hit_sort_idx = torch.argsort(hit_phi)
        hit_posencs_sorted = hit_posencs[hit_sort_idx]
        hit_phi_sorted = hit_phi[hit_sort_idx]
    else:
        hit_posencs_sorted = hit_posencs
        hit_phi_sorted = None
    
    if query_phi is not None:
        # Sort queries by phi
        query_sort_idx = torch.argsort(query_phi)
        query_posencs_sorted = query_posencs[query_sort_idx]
        query_phi_sorted = query_phi[query_sort_idx]
    else:
        query_posencs_sorted = query_posencs
        query_phi_sorted = None
    
    # 1. Query positional encoding matrix
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(query_posencs_sorted.T.numpy(), origin='lower', aspect='auto', cmap='viridis')
    ax.set_title('Query Positional Encoding Matrix (Sorted by Phi)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Query Index (Sorted by Phi)', fontsize=12)
    ax.set_ylabel('Positional Encoding Dimension', fontsize=12)
    plt.colorbar(im, ax=ax)
    
    figures['query_posenc'] = fig
    
    # 2. Hit positional encoding matrix
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(hit_posencs_sorted.T.numpy(), origin='lower', aspect='auto', cmap='viridis')
    ax.set_title('Hit Positional Encoding Matrix (Sorted by Phi)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hit Index (Sorted by Phi)', fontsize=12)
    ax.set_ylabel('Positional Encoding Dimension', fontsize=12)
    plt.colorbar(im, ax=ax)
    
    figures['hit_posenc'] = fig
    
    # 3. Dot product matrix (using sorted encodings)
    dot_product = torch.matmul(query_posencs_sorted, hit_posencs_sorted.T) / hit_posencs_sorted.shape[0]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(dot_product.numpy(), origin='lower', aspect='auto')
    ax.set_title('Query-Hit Dot Product Matrix (Sorted by Phi)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hit Index (Sorted by Phi)', fontsize=12)
    ax.set_ylabel('Query Index (Sorted by Phi)', fontsize=12)
    
    plt.colorbar(im, ax=ax)
    
    figures['dot_product'] = fig
    
    # 4. Phi distribution plots
    if hit_phi is not None or query_phi is not None:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Phi Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Query phi distribution
        if query_phi is not None:
            # Histogram
            axes[0, 0].hist(query_phi.numpy(), bins=50, alpha=0.7, edgecolor='black', color='skyblue')
            axes[0, 0].set_xlabel('Query Phi (radians)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Query Phi Distribution')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Circular histogram (polar plot) for queries
            axes[0, 1] = fig.add_subplot(2, 4, 2, projection='polar')
            axes[0, 1].hist(query_phi.numpy(), bins=36, alpha=0.7, edgecolor='black', color='skyblue')
            axes[0, 1].set_title('Query Phi Distribution (Circular)', pad=20)
        
        # Hit phi distribution
        if hit_phi is not None:
            # Histogram
            axes[1, 0].hist(hit_phi.numpy(), bins=50, alpha=0.7, edgecolor='black', color='lightcoral')
            axes[1, 0].set_xlabel('Hit Phi (radians)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Hit Phi Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Circular histogram (polar plot) for hits
            axes[1, 1] = fig.add_subplot(2, 4, 6, projection='polar')
            axes[1, 1].hist(hit_phi.numpy(), bins=36, alpha=0.7, edgecolor='black', color='lightcoral')
            axes[1, 1].set_title('Hit Phi Distribution (Circular)', pad=20)
        
        # If both are available, add comparison
        if hit_phi is not None and query_phi is not None:
            # Overlay both distributions
            axes[0, 2].hist(query_phi.numpy(), bins=50, alpha=0.7, label='Queries', color='skyblue')
            axes[0, 2].hist(hit_phi.numpy(), bins=50, alpha=0.7, label='Hits', color='lightcoral')
            axes[0, 2].set_xlabel('Phi (radians)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Query vs Hit Phi Distribution')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # Box plot comparison
            axes[1, 2].boxplot([query_phi.numpy(), hit_phi.numpy()], labels=['Queries', 'Hits'])
            axes[1, 2].set_ylabel('Phi (radians)')
            axes[1, 2].set_title('Phi Distribution Comparison')
            axes[1, 2].grid(True, alpha=0.3)
            
            # Combined circular plot
            axes[0, 3] = fig.add_subplot(2, 4, 4, projection='polar')
            axes[0, 3].hist(query_phi.numpy(), bins=36, alpha=0.7, label='Queries', color='skyblue')
            axes[0, 3].hist(hit_phi.numpy(), bins=36, alpha=0.7, label='Hits', color='lightcoral')
            axes[0, 3].set_title('Combined Phi Distribution (Circular)', pad=20)
            axes[0, 3].legend()
        else:
            # If only one type is available, add placeholder for comparison plots
            if query_phi is not None:
                axes[0, 2].text(0.5, 0.5, 'No hit phi data\nfor comparison', 
                               ha='center', va='center', transform=axes[0, 2].transAxes,
                               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                axes[0, 2].set_title('Query vs Hit Phi Distribution')
                
                axes[1, 2].text(0.5, 0.5, 'No hit phi data\nfor comparison', 
                               ha='center', va='center', transform=axes[1, 2].transAxes,
                               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                axes[1, 2].set_title('Phi Distribution Comparison')
                
                # Combined circular plot (just queries)
                axes[0, 3] = fig.add_subplot(2, 4, 4, projection='polar')
                axes[0, 3].hist(query_phi.numpy(), bins=36, alpha=0.7, color='skyblue')
                axes[0, 3].set_title('Query Phi Distribution (Circular)', pad=20)
            
            if hit_phi is not None:
                axes[0, 2].text(0.5, 0.5, 'No query phi data\nfor comparison', 
                               ha='center', va='center', transform=axes[0, 2].transAxes,
                               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                axes[0, 2].set_title('Query vs Hit Phi Distribution')
                
                axes[1, 2].text(0.5, 0.5, 'No query phi data\nfor comparison', 
                               ha='center', va='center', transform=axes[1, 2].transAxes,
                               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                axes[1, 2].set_title('Phi Distribution Comparison')
                
                # Combined circular plot (just hits)
                axes[0, 3] = fig.add_subplot(2, 4, 4, projection='polar')
                axes[0, 3].hist(hit_phi.numpy(), bins=36, alpha=0.7, color='lightcoral')
                axes[0, 3].set_title('Hit Phi Distribution (Circular)', pad=20)
                
        # Hide unused subplots
        axes[1, 3].set_visible(False)
                
        plt.tight_layout()
        figures['phi_distributions'] = fig
    
    # 5. Sorted phi plots to confirm sorting
    if hit_phi_sorted is not None or query_phi_sorted is not None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sorted Phi Values Analysis', fontsize=16, fontweight='bold')
        
        # Query phi sorted
        if query_phi_sorted is not None:
            # Line plot of sorted query phi
            axes[0, 0].plot(range(len(query_phi_sorted)), query_phi_sorted.numpy(), 
                           color='skyblue', linewidth=2, marker='o', markersize=3)
            axes[0, 0].set_xlabel('Query Index (Sorted)')
            axes[0, 0].set_ylabel('Query Phi (radians)')
            axes[0, 0].set_title('Query Phi Values (Sorted)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Check if sorting is monotonic
            is_monotonic = torch.all(torch.diff(query_phi_sorted) >= 0)
            axes[0, 0].text(0.02, 0.98, f'Monotonic: {is_monotonic}', 
                           transform=axes[0, 0].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hit phi sorted
        if hit_phi_sorted is not None:
            # Line plot of sorted hit phi
            axes[1, 0].plot(range(len(hit_phi_sorted)), hit_phi_sorted.numpy(), 
                           color='lightcoral', linewidth=2, marker='o', markersize=3)
            axes[1, 0].set_xlabel('Hit Index (Sorted)')
            axes[1, 0].set_ylabel('Hit Phi (radians)')
            axes[1, 0].set_title('Hit Phi Values (Sorted)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Check if sorting is monotonic
            is_monotonic = torch.all(torch.diff(hit_phi_sorted) >= 0)
            axes[1, 0].text(0.02, 0.98, f'Monotonic: {is_monotonic}', 
                           transform=axes[1, 0].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Comparison of sorted values
        if hit_phi_sorted is not None and query_phi_sorted is not None:
            # Overlay sorted phi values
            axes[0, 1].plot(range(len(query_phi_sorted)), query_phi_sorted.numpy(), 
                           color='skyblue', linewidth=2, label='Queries', marker='o', markersize=3)
            axes[0, 1].plot(range(len(hit_phi_sorted)), hit_phi_sorted.numpy(), 
                           color='lightcoral', linewidth=2, label='Hits', marker='o', markersize=3)
            axes[0, 1].set_xlabel('Index (Sorted)')
            axes[0, 1].set_ylabel('Phi (radians)')
            axes[0, 1].set_title('Sorted Phi Values Comparison')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Difference between sorted query and hit phi ranges
            query_range = query_phi_sorted.max() - query_phi_sorted.min()
            hit_range = hit_phi_sorted.max() - hit_phi_sorted.min()
            axes[1, 1].bar(['Query Range', 'Hit Range'], [query_range.item(), hit_range.item()], 
                          color=['skyblue', 'lightcoral'])
            axes[1, 1].set_ylabel('Phi Range (radians)')
            axes[1, 1].set_title('Phi Value Ranges')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add text with range information
            axes[1, 1].text(0.02, 0.98, f'Query: [{query_phi_sorted.min():.3f}, {query_phi_sorted.max():.3f}]\nHit: [{hit_phi_sorted.min():.3f}, {hit_phi_sorted.max():.3f}]', 
                           transform=axes[1, 1].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        figures['sorted_phi_analysis'] = fig
    
    return figures


class PositionalEncodingAnalysisLogger(Callback):
    """Simple callback for analyzing positional encodings from MaskFormer models.
    
    This callback logs positional encoding analysis once per training, accessing
    the positional encodings directly from the model's instance variables.
    """
    
    def __init__(self, output_dir: str = "pos_enc_analysis"):
        """
        Initialize the positional encoding analysis logger.
        
        Args:
            output_dir: Directory to save analysis outputs
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Track if we've logged this epoch
        self._logged_this_epoch = False
        
    def _create_pos_enc_visualizations(self, hit_posencs, query_posencs, logger, hit_phi=None, query_phi=None):
        """Create and log positional encoding visualizations."""
        
        if hit_posencs is None or query_posencs is None:
            warnings.warn("No positional encodings available for visualization")
            return
        
        # Create visualizations with phi sorting
        figures = create_pos_enc_visualizations(hit_posencs, query_posencs, hit_phi, query_phi)
        
        # Log to experiment
        if logger is not None and hasattr(logger, 'experiment'):
            for name, fig in figures.items():
                logger.experiment.log_figure(
                    figure_name=f"pos_enc_{name}",
                    figure=fig,
                )
            plt.close()
    
    def on_train_start(self, trainer, pl_module):
        """Log positional encoding analysis at the start of training."""
        
        # Check if model has phi_analysis enabled
        if not hasattr(pl_module.model, 'phi_analysis') or not pl_module.model.phi_analysis:
            warnings.warn("Model has phi_analysis=False. Set phi_analysis=True in MaskFormer for positional encoding analysis.")
            return
        
        # Check if model has stored positional encodings
        if not hasattr(pl_module.model, 'last_query_posenc') or not hasattr(pl_module.model, 'last_key_posenc'):
            warnings.warn("Model does not have stored positional encodings. Run a forward pass first.")
            return
        
        # Get positional encodings from model and ensure consistent dtype
        hit_posencs = torch.tensor(pl_module.model.last_key_posenc_sorted, dtype=torch.float16)
        query_posencs = torch.tensor(pl_module.model.last_query_posenc, dtype=torch.float16)
        
        # Get phi values if available
        hit_phi = None
        query_phi = None
        if hasattr(pl_module.model, 'last_key_phi'):
            hit_phi = torch.tensor(pl_module.model.last_key_phi, dtype=torch.float32)
        else:
            raise ValueError("No key phi")
        if hasattr(pl_module.model, 'last_query_phi'):
            query_phi = torch.tensor(pl_module.model.last_query_phi, dtype=torch.float32)
        
        logger = getattr(pl_module, 'logger', None)
        
        try:
            self._create_pos_enc_visualizations(hit_posencs, query_posencs, logger, hit_phi, query_phi)
            self._logged_this_epoch = True
            
        except Exception as e:
            warnings.warn(f"Error in positional encoding analysis: {e}")