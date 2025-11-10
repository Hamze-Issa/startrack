import torch
import matplotlib.pyplot as plt
from torchgeo.datasets import RasterDataset
from tools import log_tensor_stats, encode_time_to_sample, parse_meta
import re

# Here you can define whatever datasets you need (preferrably inheriting from Torchgeo's RasterDataset or VectorDataset)
# In the style of the below example datasets (Datasets1 and 2 and Mask_dataset which is for labels).
# Then don't forget to add your newly created dataset to the DATASET_CLASSES dictionary below so it can be used by the trainer. 

def process_sample(self, sample, nodata_value=None, replace_value=0.0):
    # If keep_meta is True (preferred to be false unless you're predicting and need the metadata to save predictions)
    # then this saves metadata in a safe way to keep them separate from tensor mutable data before entering Lightning
    # keep_meta is defaulted to True only in the prediction/testing script
    if self.keep_meta:
        sample['meta'] = parse_meta(sample)
    # Process image and mask if they exist
    processed = {}
    for key in sample:
        if key == "image" or key == "mask":
            tensor = sample[key]
            # Store a mask where values were valid (not nan/inf/nodata)
            valid_mask = ~torch.isnan(tensor) & torch.isfinite(tensor)

            # Replace NaN, positive Inf, negative Inf
            tensor = torch.nan_to_num(tensor, nan=replace_value, posinf=replace_value, neginf=replace_value)

            if nodata_value is not None:
                # Create a mask where tensor equals nodata_value
                nodata_mask = (tensor == nodata_value)
                # Update valid_mask to also exclude nodata values
                valid_mask = valid_mask & (~nodata_mask)
                # Replace those nodata values with replace_value
                tensor = torch.where(nodata_mask, torch.tensor(replace_value, dtype=tensor.dtype, device=tensor.device), tensor)
            # Store the resultant tensor that should have only valid values and anything else set to replace_value
            processed[key] = tensor
            processed[f'{key}_{self.name}_valid'] = valid_mask.float()
        else:
            # Preserve all other keys unchanged
            processed[key] = sample[key]
    return processed


class Dataset_1(RasterDataset):
    # filename_regex = r"sst_(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})\.tif"
    # date_format = "%Y%m%dT%H%M%S"
    def __init__(self, root: str, name: str="dataset_1", filename_glob: str='*.tif', filename_regex='dataset_1_(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})\.tif', date_format="%Y%m%dT%H%M%S", is_image=True, bands=('1'), nodata_value=None, replace_value=0.0, keep_meta=False, **kwargs):
        self.name = name
        self.filename_glob = filename_glob
        self.filename_regex = filename_regex
        self.date_format = date_format
        self.is_image = is_image
        self.all_bands = bands
        self.nodata_value = nodata_value
        self.replace_value = replace_value
        self.keep_meta = keep_meta
        super().__init__(root, **kwargs)
        # for attr, value in vars(self).items():
        #     print(f"{attr}: {value}")
    
    # # separate_files = True
    # # rgb_bands = ('B04', 'B03', 'B02')

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        

        processed = process_sample(self, sample, self.nodata_value, self.replace_value)
        processed_time_encoded = encode_time_to_sample(processed, index)
        return processed_time_encoded

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
    def __init__(self, root: str, name: str="dataset_2", filename_glob: str='*.tif', filename_regex='dataset_2_(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})\.tif', date_format="%Y%m%dT%H%M%S", is_image=True, bands=('1'), nodata_value=None, replace_value=0.0, keep_meta=False, **kwargs):
        self.name = name
        self.filename_glob = filename_glob
        self.filename_regex = filename_regex
        self.date_format = date_format
        self.is_image = is_image
        self.all_bands = bands
        self.nodata_value = nodata_value
        self.replace_value = replace_value
        self.keep_meta = keep_meta
        super().__init__(root, **kwargs)

    # # separate_files = True
    # # rgb_bands = ('B04', 'B03', 'B02')

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        
        processed = process_sample(self, sample, self.nodata_value, self.replace_value)
        return processed

class Mask_dataset(RasterDataset):
    def __init__(self, root: str, name: str="mask_dataset", filename_glob: str='*.tif', filename_regex='mask_dataset_(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})\.tif', date_format="%Y%m%dT%H%M%S", is_image=False, bands=('1'), nodata_value=None, replace_value=0.0, keep_meta=False, **kwargs):
        self.name = name
        self.filename_glob = filename_glob
        self.filename_regex = filename_regex
        self.date_format = date_format
        self.is_image = is_image
        self.all_bands = bands
        self.nodata_value = nodata_value
        self.replace_value = replace_value
        self.keep_meta = keep_meta
        super().__init__(root, **kwargs)

    # # separate_files = True
    # # rgb_bands = ('B04', 'B03', 'B02')

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if "image" in sample:
            sample["mask"] = sample.pop("image")

        processed = process_sample(self, sample, self.nodata_value, self.replace_value)
        return processed

DATASET_CLASSES = {
    "Dataset_1": Dataset_1,
    "Dataset_2": Dataset_2,
    "Mask_dataset": Mask_dataset,
    # Add new key-class pairs
}