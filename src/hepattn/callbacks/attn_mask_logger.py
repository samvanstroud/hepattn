import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback

class AttnMaskLogger(Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        if hasattr(model, "_last_attn_mask"):
            attn_mask = model._last_attn_mask.numpy()
            step = getattr(model, "_last_attn_mask_step", 0)
            layer = getattr(model, "_last_attn_mask_layer", 0)
            plt.figure(constrained_layout=True, dpi=300)
            plt.imshow(attn_mask, aspect="auto")
            if hasattr(model, "log_figure"):
                model.log_figure(f"local_ca_mask_val_step{step}_layer{layer}", plt.gcf(), step=step)
            plt.close()
            # Optionally clear after logging
            del model._last_attn_mask


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        step = getattr(model, "step_") 
        if (step % 1000 == 0) and hasattr(model, "_last_attn_mask"):
            attn_mask = model._last_attn_mask.numpy()
            layer = getattr(model, "_last_attn_mask_layer", 0)
            step = getattr(model, "_last_attn_mask_step", 0)
            plt.figure(constrained_layout=True, dpi=300)
            plt.imshow(attn_mask, aspect="auto")
            pl_module.log_figure(f"local_ca_mask_step{step}_layer{layer}", plt.gcf(), step=step)
            plt.close()
            del model._last_attn_mask