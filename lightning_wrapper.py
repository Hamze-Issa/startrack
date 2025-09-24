import torch
import lightning as pl
import torch.optim as optim
from torchmetrics import MetricCollection
from loss_functions import LOSS_FUNCTIONS
from metrics import METRIC_FUNCTIONS
import matplotlib.pyplot as plt
from plotting import create_multichannel_figure
from tools import log_tensor_stats, get_joint_valid_mask

class GenericTask(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        # Loss
        loss_config = config['loss']
        loss_cls = LOSS_FUNCTIONS[loss_config['name']]
        self.loss_fn = loss_cls(**loss_config.get('params', {}))

        # Metrics
        self.metrics = {}
        for metric_key, metric_info in config['metrics'].items():
            metric_cls = METRIC_FUNCTIONS[metric_info['name']]
            self.metrics[metric_key] = metric_cls(**metric_info.get('params', {}))
        self.metrics = MetricCollection(self.metrics)

    def log_figure(self, input, mask, prediction, batch_idx, filename, input_channel_names=None, mask_channel_names=None, pred_channel_names=None, cmap='viridis'):
        ##### Logging images/figures: Logging imags in Mlflow is having troubles appearing in the main grid for now due to this issue https://github.com/mlflow/mlflow/issues/15760
        ##### So as of the time this code is written, you can only view logged images in the artifacts tab, but maybe they will fix it soon hopefully.
        fig = create_multichannel_figure(input, mask, prediction, input_channel_names, mask_channel_names, pred_channel_names, cmap)
        self.logger.experiment.log_figure(
            run_id=self.logger.run_id,
            figure=fig,
            artifact_file=f"{filename}_epoch{self.current_epoch}_batch{batch_idx}.png"
        )
        plt.close(fig)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, prefix, batch_idx):
        ##### Loss and Metric Calculation #####
        x, y = batch["image"], batch["mask"]
        x_valid = batch.get("sst_image_valid", None)
        y_hat_valid = batch.get("chl_mask_valid", None)
        
        # Forward pass
        y_hat = self(x)  # Shape: [B, C, H, W]

        # Mask y and y_hat with the joint masks from all the datasets
        joint_valid = get_joint_valid_mask(batch)
        if joint_valid is not None:
            y_hat_masked = torch.where(joint_valid, y_hat, torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype))
            y_masked = torch.where(joint_valid, y, torch.tensor(0.0, device=y.device, dtype=y.dtype))
        else:
            y_hat_masked = y_hat
            y_masked = y

        # Print statistics
        log_tensor_stats("x", x)
        log_tensor_stats("y", y)
        log_tensor_stats("y_hat", y_hat)
        log_tensor_stats("x_valid", x_valid)
        log_tensor_stats("y_hat_valid", y_hat_valid)
        log_tensor_stats("y_hat_masked", y_hat_masked)
        log_tensor_stats("y_masked", y_masked)
        
        # Calculate loss
        loss = self.loss_fn(y_hat, y.float())

        # Calculate metrics
        preds = y_hat.sigmoid()
        metrics = self.metrics(preds, y.int())
        

        ##### Logging parameters and images #####
        # Different logging for train/val
        if prefix == "train":
            self.log(f"{prefix}_loss", loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True) # Logging Loss
            self.log_dict({f"{prefix}_{k}": v for k,v in metrics.items()}, sync_dist=True, on_step=True, on_epoch=True) # Logging Metrics
        else:  # validation
            self.log(
                f"{prefix}_loss", 
                loss,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                prog_bar=True
            )
            self.log_dict(
                {f"{prefix}_{k}": v for k,v in metrics.items()},
                sync_dist=True,
                on_step=True,
                on_epoch=True
            )

        if batch_idx == 0: # change this if you want another batch or another logging logic
            self.log_figure(x[0, :].detach().cpu().numpy(), y[0, :].detach().cpu().numpy(), y_hat[0, :].detach().cpu().numpy(), batch_idx, 'composite')
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val", batch_idx)

    # Still under construction
    def predict_step(self, batch, batch_idx):
        x = batch["image"]
        y_hat = self(x)  # [B, C, H, W]
        # Optionally apply sigmoid or argmax for predicted classes/masks:
        preds = y_hat.sigmoid() # For binary/multilabel
        # preds = y_hat.argmax(dim=1) # For multiclass

        metadata = {key: batch[key] for key in ['crs', 'affine'] if key in batch}

        return {
            "prediction": preds.detach().cpu().numpy(),
            "metadata": metadata,
        }

    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(), 
            lr=self.config['model']['learning_rate']
        )
    
