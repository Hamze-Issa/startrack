import torch
import matplotlib.pyplot as plt
from torchgeo.datasets import RasterDataset

# Here you can define whatever datasets you need (preferrably inheriting from Torchgeo's RasterDataset or VectorDataset)
# In the style of the below example datasets (Datasets1 and 2 and Mask_dataset which is for labels).
# Then don't forget to add your newly created dataset to the DATASET_CLASSES dictionary below so it can be used by the trainer. 

def process_sample(self, sample):
    # Process image and mask if they exist
    processed = {}
    for key in sample:
        if key == "image":
            # Handle image NaN values
            image = sample["image"]
            nan_mask = ~torch.isnan(image)
            processed["image"] = torch.nan_to_num(image, nan=0.0)
            processed[f'${self.name}_nan_mask'] = nan_mask.float()
        elif key == "mask":
            # Handle mask NaN values
            mask = sample["mask"]
            nan_mask = ~torch.isnan(mask)
            processed["mask"] = torch.nan_to_num(mask, nan=0.0)  # Or your ignore_index
            processed[f'${self.name}_nan_mask'] = nan_mask.float()
        else:
            # Preserve all other keys unchanged
            processed[key] = sample[key]
    return processed

class Dataset_1(RasterDataset):
    def __init__(self, root: str, name: str="dataset_1", filename_glob: str='dataset_1*.tif', is_image=True, bands=('1'), **kwargs):
        super().__init__(root, **kwargs)
        self.name = name
        self.filename_glob = filename_glob
        self.is_image = is_image
        self.all_bands = bands
    
    # # filename_regex = r'^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B0[\d])'
    # # date_format = '%Y%m%dT%H%M%S'
    # # separate_files = True
    # # rgb_bands = ('B04', 'B03', 'B02')

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        
        processed = process_sample(self, sample)
        return processed

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
    
class Dataset_2(RasterDataset):
    def __init__(self, root: str, name: str="dataset_2", filename_glob: str='dataset_2*.tif', is_image=True, bands=('1'), **kwargs):
        super().__init__(root, **kwargs)
        self.name = name
        self.filename_glob = filename_glob
        self.is_image = is_image
        self.all_bands = bands

    # # filename_regex = r'^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B0[\d])'
    # # date_format = '%Y%m%dT%H%M%S'
    # # separate_files = True
    # # rgb_bands = ('B04', 'B03', 'B02')

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        
        processed = process_sample(self, sample)
        return processed

class Mask_dataset(RasterDataset):
    def __init__(self, root: str, name: str="mask_dataset", filename_glob: str='*.tif', is_image=False, bands=('1'), **kwargs):
        super().__init__(root, **kwargs)
        self.name = name
        self.filename_glob = filename_glob
        self.is_image = is_image
        self.all_bands = bands
    
    # # filename_regex = r'^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B0[\d])'
    # # date_format = '%Y%m%dT%H%M%S'
    # # separate_files = True
    # # rgb_bands = ('B04', 'B03', 'B02')

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        
        processed = process_sample(self, sample)
        return processed

DATASET_CLASSES = {
    "Dataset_1": Dataset_1,
    "Dataset_2": Dataset_2,
    "Mask_dataset": Mask_dataset,
    # Add new key-class pairs
}