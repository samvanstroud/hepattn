import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback

class AttnMaskLogger(Callback):
    def _log_attention_mask(self, pl_module, model, step, layer, prefix="local_ca_mask"):
        """Helper method to create and log attention mask figures."""
        fig, ax = plt.subplots(constrained_layout=True, dpi=300)
        im = ax.imshow(model._last_attn_mask.numpy(), aspect="auto", cmap='RdYlBu_r', vmin=0, vmax=1)
        
        # Add colorbar with clear labels
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.set_label('Attention Mask', rotation=270, labelpad=15)
        cbar.ax.set_yticklabels(['Masked (0)', 'Used in Attention (1)'])
        
        # Add title with step and layer info
        ax.set_title(f'Attention Mask - Step {step}, Layer {layer}')
        ax.set_xlabel('Input Constituents')
        ax.set_ylabel('Queries')
        
        # Log directly to Comet
        if hasattr(pl_module, 'logger') and pl_module.logger is not None:
            if hasattr(pl_module.logger, 'experiment'):
                pl_module.logger.experiment.log_figure(
                    figure_name=f"{prefix}_step{step}_layer{layer}",
                    figure=fig,
                    step=step
                )
        
        plt.close(fig)
        
        # Clear the stored attention mask
        del model._last_attn_mask
        if hasattr(model, '_last_attn_mask_step'):
            del model._last_attn_mask_step
        if hasattr(model, '_last_attn_mask_layer'):
            del model._last_attn_mask_layer

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
        
        if step % 1000 == 0:
            # Directly check for stored attention mask
            if hasattr(model, '_last_attn_mask'):
                step = getattr(model, '_last_attn_mask_step', 0)
                layer = getattr(model, '_last_attn_mask_layer', 0)
                self._log_attention_mask(pl_module, model, step, layer, "local_ca_mask")