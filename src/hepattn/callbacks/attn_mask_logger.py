import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from matplotlib.colors import ListedColormap

class AttnMaskLogger(Callback):
    def _log_attention_mask(self, pl_module, model, step, layer, prefix="local_ca_mask"):
        """Helper method to create and log attention mask figures."""
        fig, ax = plt.subplots(constrained_layout=True, dpi=300)
        attn_mask = getattr(model, '_last_attn_mask', None)
        if attn_mask is None:
            return

        cmap = ListedColormap(['#4575b4', '#d73027'])  # blue for 0, red for 1
        im = ax.imshow(attn_mask.numpy().astype(int), aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            
        # im = ax.imshow(attn_mask.numpy(), aspect="auto", cmap='RdYlBu_r', vmin=0, vmax=1)
        
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
        
        # Clear the stored attention mask
        for attr_name in ['_last_attn_mask', '_last_attn_mask_step', '_last_attn_mask_layer']:
            if hasattr(model, attr_name):
                delattr(model, attr_name)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        
        # Directly check for stored attention mask
        if hasattr(model, '_last_attn_mask'):
            step = getattr(model, '_last_attn_mask_step', 0)
            layer = getattr(model, '_last_attn_mask_layer', 0)
            self._log_attention_mask(pl_module, model, step, layer, "local_ca_mask_val")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        step = getattr(model, "step_", 0)
        
        if step % 1000 == 0 and hasattr(model, '_last_attn_mask'):
            step = getattr(model, '_last_attn_mask_step', 0)
            layer = getattr(model, '_last_attn_mask_layer', 0)
            self._log_attention_mask(pl_module, model, step, layer, "local_ca_mask")