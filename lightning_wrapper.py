import torch
import lightning as pl
import torch.optim as optim
from torchmetrics import MetricCollection
from loss_functions import LOSS_FUNCTIONS
from metrics import METRIC_FUNCTIONS
import matplotlib.pyplot as plt
from plotting import create_multichannel_figure
from tools import log_tensor_stats, get_joint_valid_mask, save_predictions
from augmentations import create_augmentation_pipeline

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

        # Batch Augmentations
        if 'augmentations' in config:
            self.batch_augmentations = create_augmentation_pipeline(config['augmentations'])
        else:
            self.batch_augmentations = None

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
        # Apply augmentations
        if self.batch_augmentations is not None:
            # Select keys with tensors
            tensor_keys = [k for k, v in batch.items() if isinstance(v, torch.Tensor)]
            # Create input dict for Kornia augmentation
            inputs = {k: batch[k] for k in tensor_keys}
            # Apply augmentation jointly
            outputs = self.batch_augmentations(inputs)
            # Replace tensors in batch with augmented versions
            for k in tensor_keys:
                batch[k] = outputs[k]

        ##### Loss and Metric Calculation #####
        x, y = batch["image"], batch["mask"]
        x_valid = batch.get("image_sst_valid", None)
        y_valid = batch.get("mask_chl_valid", None)
        
        # Mask x and y with the joint masks from all the datasets
        joint_valid = get_joint_valid_mask(batch)
        if joint_valid is not None:
            x_masked = torch.where(joint_valid, x, torch.tensor(0.0, device=x.device, dtype=x.dtype))
            y_masked = torch.where(joint_valid, y, torch.tensor(0.0, device=y.device, dtype=y.dtype))
        else:
            x_masked = x
            y_masked = y

        # Forward pass
        y_hat = self(x_masked)  # Shape: [B, C, H, W]

        # # Print statistics
        # log_tensor_stats("x", x)
        # log_tensor_stats("y", y)
        # log_tensor_stats("y_hat", y_hat)
        # log_tensor_stats("x_valid", x_valid)
        # log_tensor_stats("y_valid", y_valid)
        # log_tensor_stats("y_masked", y_masked)

        # Calculate loss
        loss = self.loss_fn(y_hat, y_masked.float())

        # Calculate metrics
        # Apply activation if needed
        preds = y_hat.sigmoid() # For binary/multilabel
        # preds = y_hat.argmax(dim=1) # For multiclass
        # preds = y_hat #y_hat.sigmoid()
        metrics = self.metrics(preds, y_masked.int())
        

        ##### Logging parameters and images #####
        # Different logging for train/val
        if prefix == "train":
            self.log(f"{prefix}_loss", loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.config['training']['batch_size']) # Logging Loss
            self.log_dict({f"{prefix}_{k}": v for k,v in metrics.items()}, sync_dist=True, on_step=True, on_epoch=True, batch_size=self.config['training']['batch_size']) # Logging Metrics
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

        # if batch_idx == 0: # make a condition here if you need another logging logic
        self.log_figure(x[0, :].detach().cpu().numpy(), y[0, :].detach().cpu().numpy(), y_hat[0, :].detach().cpu().numpy(), batch_idx, 'composite')
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val", batch_idx)

    def predict_step(self, batch, batch_idx):
        x = batch["image"]            # [B, C, H, W]
        meta = batch["meta"]          # list or dict of metadata per sample
        y_hat = self(x)               # model forward pass

        # Mask y_hat with the joint masks from all the datasets
        joint_valid = get_joint_valid_mask(batch)
        if joint_valid is not None:
            y_hat_masked = torch.where(joint_valid, y_hat, torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype))
        else:
            y_hat_masked = y_hat

        # Apply activation if needed
        # preds = y_hat.sigmoid() # For binary/multilabel
        # preds = y_hat.argmax(dim=1) # For multiclass
        crop = int(x.shape[-1] / 4) # cropping the center parts of the predictions to avoid edge artifacts (compensated by an overlap of the same size made by the stride in the dataloader)
        save_dir = self.config['testing'].get('output_dir', "./inference_outputs")
        outputs = save_predictions(y_hat_masked, meta, save_dir, batch_idx, crop) # saving in format: "pred_{batch_idx}_{i}_{mint}_{maxt}.tif"
        # outputs = save_predictions(batch['mask'], meta, "./inference_masks", batch_idx, crop) # uncomment if you want to also save the labels

        return outputs

    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(), 
            lr=self.config['model']['learning_rate']
        )
