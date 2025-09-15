import torch
import matplotlib.pyplot as plt
from torchgeo.datasets import RasterDataset


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

class IceVelocity_u(RasterDataset):
    def __init__(self, root: str, name: str = "ice_velocity_u", **kwargs):
        super().__init__(root, **kwargs)
        self.name = name
        self.filename_glob = 'ice_velocity_u_*.tif'
        self.is_image = True
        self.all_bands = ('1')
    
    # filename_glob = 'ice_velocity_u_*.tif'
    # # filename_regex = r'^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B0[\d])'
    # # date_format = '%Y%m%dT%H%M%S'
    # is_image = True
    # # separate_files = True
    # all_bands = ('1')
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
    
class IceVelocity_v(RasterDataset):
    def __init__(self, root: str, name: str = "ice_velocity_v", **kwargs):
        super().__init__(root, **kwargs)
        self.name = name
        self.filename_glob = 'ice_velocity_v_*.tif'
        self.is_image = True
        self.all_bands = ('1')

    # filename_glob = 'ice_velocity_v_*.tif'
    # # filename_regex = r'^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B0[\d])'
    # # date_format = '%Y%m%dT%H%M%S'
    # is_image = True
    # # separate_files = True
    # all_bands = ('1')
    # # rgb_bands = ('B04', 'B03', 'B02')

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        
        processed = process_sample(self, sample)
        return processed

class Calving(RasterDataset):
    def __init__(self, root: str, name: str = "calving", **kwargs):
        super().__init__(root, **kwargs)
        self.name = name
        self.filename_glob = '*.tif'
        self.is_image = False
        self.all_bands = ('1')
    
    # filename_glob = '*.tif'
    # # filename_regex = r'^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B0[\d])'
    # # date_format = '%Y%m%dT%H%M%S'
    # is_image = False
    # # separate_files = True
    # all_bands = ('1')
    # # rgb_bands = ('B04', 'B03', 'B02')

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        
        processed = process_sample(self, sample)
        return processed
