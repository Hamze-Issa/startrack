import os
import tempfile
from urllib.parse import urlparse

import matplotlib.pyplot as plt
# import planetary_computer
# import pystac
import torch
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer
from torchgeo.models import ResNet50_Weights
from torchgeo.trainers import SemanticSegmentationTask

from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from custom_geo_data_module import CustomGeoDataModule as gdm
from models.model_wrapper import LoopUnet
from lightning_wrapper import LightningWrapper

class IceVelocity_u(RasterDataset):
    filename_glob = 'ice_velocity_u_*.tif'
    # filename_regex = r'^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B0[\d])'
    # date_format = '%Y%m%dT%H%M%S'
    is_image = True
    # separate_files = True
    all_bands = ('1')
    # rgb_bands = ('B04', 'B03', 'B02')

    def plot(self, sample):
        image = sample['image'][0]
        image = torch.clamp(image / 10000, min=0, max=1).numpy()
        fig, ax = plt.subplots()
        ax.imshow(image)
        return fig
    
    def plot_rgb(self, sample):
        # Find the correct band index order
        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))
        image = sample['image'][rgb_indices].permute(1, 2, 0)
        image = torch.clamp(image / 10000, min=0, max=1).numpy()
        fig, ax = plt.subplots()
        ax.imshow(image)

        return fig
    
class IceVelocity_v(RasterDataset):
    filename_glob = 'ice_velocity_v_*.tif'
    # filename_regex = r'^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B0[\d])'
    # date_format = '%Y%m%dT%H%M%S'
    is_image = True
    # separate_files = True
    all_bands = ('1')
    # rgb_bands = ('B04', 'B03', 'B02')

class Calving(RasterDataset):
    filename_glob = '*.tif'
    # filename_regex = r'^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B0[\d])'
    # date_format = '%Y%m%dT%H%M%S'
    is_image = False
    # separate_files = True
    all_bands = ('1')
    # rgb_bands = ('B04', 'B03', 'B02')


ivu_root = '/mnt/experiment-3/AI4IS/data/gtiff_200m/ice_velocity_u/'
ivu = IceVelocity_u(ivu_root)
ivv_root = '/mnt/experiment-3/AI4IS/data/gtiff_200m/ice_velocity_v/'
ivv = IceVelocity_v(ivv_root)
calving_root = '/mnt/experiment-3/AI4IS/data/calving/'
calving = Calving(calving_root)

weights = ResNet50_Weights.SENTINEL1_GRD_MOCO
task = SemanticSegmentationTask(model='unet', backbone='resnet50', weights=weights, in_channels=2, task='binary', num_classes=None, num_labels=None, num_filters=3, loss='focal', class_weights=None, ignore_index=None, lr=0.001, patience=10, freeze_backbone=False, freeze_decoder=False)

# or for a custom model
model = LoopUnet()
lightning_model = LightningWrapper(model)


iv = ivu & ivv
combined_dataset = iv & calving

batch_size = 1
num_workers = 2
max_epochs = 5
fast_dev_run = False


# default_root_dir = os.path.join(tempfile.gettempdir(), 'experiments')
default_root_dir = '.'
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss', dirpath=default_root_dir, save_top_k=1, save_last=True
)
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=10)
logger = TensorBoardLogger(save_dir=default_root_dir, name='tutorial_logs')

trainer = Trainer(
    callbacks=[checkpoint_callback, early_stopping_callback],
    fast_dev_run=fast_dev_run,
    log_every_n_steps=1,
    logger=logger,
    min_epochs=1,
    max_epochs=max_epochs,
    use_distributed_sampler=False,
)

# Create the datamodule with your existing dataset
datamodule = gdm(
    dataset=combined_dataset,  # Pass the initialized dataset
    batch_size=1,
    patch_size=(100, 100),
    length=10, #1000,  # samples per epoch
    num_workers=4,
    split_ratios=(0.7, 0.2, 0.1),  # Custom train/val/test splits
    seed=42  # For reproducibility
)

# Setup (called automatically by Lightning if using Trainer)
datamodule.setup("fit")  # For training

# # Get dataloaders
# train_loader = datamodule.train_dataloader()
# val_loader = datamodule.val_dataloader()

# Or use with Lightning
trainer.fit(task, datamodule=datamodule)