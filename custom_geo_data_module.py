from torchgeo.datamodules import GeoDataModule
from torchgeo.samplers import RandomBatchGeoSampler, GridGeoSampler, RandomGeoSampler
import torch
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from torchgeo.datasets.splits import random_bbox_assignment
import typing

class CustomGeoDataModule(GeoDataModule):
    def __init__(
        self,
        config,
        dataset: object,  # Your already initialized dataset
        batch_size: int = 64,
        patch_size: Tuple[int, int] = (256, 256),
        length: Optional[int] = None,
        num_workers: int = 0,
        split_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),  # train, val, test
        seed: int = 42,  # For reproducible splits
    ) -> None:
        """Initialize with an existing dataset.
        
        Args:
            dataset: Your already initialized dataset object
            batch_size: Size of each batch
            patch_size: Size of patches to sample (height, width)
            length: Number of samples per epoch
            num_workers: Workers for data loading
            split_ratios: Ratios for train/val/test split
            seed: Random seed for reproducible splits
        """
        # We pass None as dataset_class since we're handling the dataset ourselves
        super().__init__(
            dataset_class=None,  # This satisfies the parent class requirement
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            num_workers=num_workers,
        )
        self.collate_fn = self._collate_fn
        self.dataset = dataset
        self.split_ratios = split_ratios
        self.seed = seed
        self.config = config
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_batch_sampler = None
        self.val_sampler = None
        self.test_sampler = None

    def _collate_fn(self, samples):
        from torchgeo.datasets import stack_samples
        return stack_samples(samples)

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers."""        
        # Split the dataset
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_bbox_assignment(
            self.dataset, 
            self.split_ratios,
            generator
        )
        
        # Set up samplers
        if stage in ["fit"]:
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset,
                self.patch_size,
                self.batch_size,
                self.length
            )
        if stage in ["fit", "validate"]:
            self.val_sampler = RandomGeoSampler( # Either use that one for random samples or the GridGeoSampler to cover the whole space but will take a lot more time
                self.val_dataset,
                size=self.config['training']['patch_size'],
                length=self.config['training']['val_samples'],  # RandomGeoSampler accepts length
            )
            # # Strided validation sampler with limited samples
            # self.val_sampler = GridGeoSampler(
            #     self.val_dataset,
            #     self.config['training']['patch_size'],
            #     stride=max(1, int(self.config['training']['patch_size'][0] * 0.8)),  # 80% overlap reduction
            #     length=self.config['training']['val_samples']  # Hard limit
            # )
        if stage in ["test"]:
            self.test_sampler = GridGeoSampler(
                self.test_dataset,
                self.patch_size,
                self.patch_size
            )
        if stage in ["predict"]:
            self.predict_sampler = GridGeoSampler(
                self.test_dataset,
                self.patch_size,
                self.patch_size
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn #torchgeo_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn #torchgeo_collate,
        )
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.predict_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn #torchgeo_collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn #torchgeo_collate,
        )
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Validate batch after transfer to device"""
        batch["image"] = torch.nan_to_num(batch["image"], nan=0.0)
        if "mask" in batch:
            batch["mask"] = torch.nan_to_num(batch["mask"], nan=0.0)
        return batch
