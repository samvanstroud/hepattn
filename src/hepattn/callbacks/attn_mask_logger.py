import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback

class AttnMaskLogger(Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        attn_mask_info = model.get_last_attention_mask() if hasattr(model, "get_last_attention_mask") else None
        
        if attn_mask_info is not None:
            attn_mask, step, layer = attn_mask_info
            plt.figure(constrained_layout=True, dpi=300)
            plt.imshow(attn_mask.numpy(), aspect="auto")
            if hasattr(model, "log_figure"):
                model.log_figure(f"local_ca_mask_val_step{step}_layer{layer}", plt.gcf(), step=step)
            plt.close()
            # Clear after logging
            model.clear_last_attention_mask()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        step = getattr(model, "step_", 0)
        
        if step % 1000 == 0:
            attn_mask_info = model.get_last_attention_mask() if hasattr(model, "get_last_attention_mask") else None
            
            if attn_mask_info is not None:
                attn_mask, step, layer = attn_mask_info
                plt.figure(constrained_layout=True, dpi=300)
                plt.imshow(attn_mask.numpy(), aspect="auto")
                pl_module.log_figure(f"local_ca_mask_step{step}_layer{layer}", plt.gcf(), step=step)
                plt.close()
                model.clear_last_attention_mask()