import lightning as pl
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MetricCollection, JaccardIndex
from loss_functions import MaskedBCELoss
import matplotlib.pyplot as plt
from plotting import create_multichannel_figure
import os
import rasterio

def save_prediction_with_metadata(preds, metadata, batch_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, pred in enumerate(preds):
        profile = {
            'driver': 'GTiff',
            'height': pred.shape[-2],
            'width': pred.shape[-1],
            'count': pred.shape[0] if pred.ndim == 3 else 1,
            'dtype': pred.dtype,
            'crs': metadata.get('crs'),
            'transform': metadata.get('affine'),
        }
        print(profile)
        out_path = os.path.join(output_dir, f"prediction_{batch_idx}_{i}.tif")
        with rasterio.open(out_path, 'w', **profile) as dst:
            if pred.ndim == 2:
                dst.write(pred, 1)
            else:
                dst.write(pred)
        print(f"Saved {out_path}")

class GenericTask(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = self._get_loss_fn()
        self.metrics = MetricCollection({
            'iou': JaccardIndex(task='binary'),
            # Add other metrics as needed
        })

    def _get_loss_fn(self):
        loss_type = self.config['model'].get('loss', 'bce').lower()
        if loss_type == 'bce':
            return MaskedBCELoss() #nn.BCEWithLogitsLoss() # add whatever loss you want
        elif loss_type == 'dice':
            from segmentation_models_pytorch.losses import DiceLoss
            return DiceLoss(mode='binary')
        raise ValueError(f"Unknown loss: {loss_type}")

    def log_figure(self, input, mask, prediction, batch_idx, filename, input_channel_names=None, mask_channel_names=None, pred_channel_names=None, cmap='Blues'):
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
        nan_mask = batch.get("nan_mask", None)
        
        # Forward pass
        y_hat = self(x)  # Shape: [B, C, H, W]
        
        # Mask out NaNs
        if nan_mask is not None:
            y_hat = y_hat[nan_mask.bool()]
            y = y[nan_mask.bool()]
        
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
        # print('batch________________',batch)
        # print('metadata_______________', metadata)
        

        
        # save results
        # if self.config['testing'].get('save_predictions', True):
        #     output_dir = self.config['testing'].get('output_dir', './inference_outputs')
        #     save_prediction_with_metadata(preds, metadata, batch_idx, output_dir)
        return {
            "prediction": preds.detach().cpu().numpy(),
            "metadata": metadata,
        }

    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(), 
            lr=self.config['model']['learning_rate']
        )
    
