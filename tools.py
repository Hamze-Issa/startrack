import torch
from datetime import datetime, timezone
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import torch.nn.functional as F


def update_config(cfg, keys, value):
    """Recursively update nested dictionary cfg with keys list and set to value."""
    d = cfg
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    # Try to interpret value as Python literal
    try:
        import ast
        value = ast.literal_eval(value)
    except Exception:
        pass
    d[keys[-1]] = value

def log_tensor_stats(name, tensor):
    if tensor is None:
        print(f"{name}: None")
        return
    # Cast to float for statistics (won't affect original tensor)
    tensor_stats = tensor.float()
    stats = (
        f"{name}: shape={tuple(tensor.shape)}, "
        f"min={tensor.min().item():.3g}, "
        f"max={tensor.max().item():.3g}, "
        f"mean={tensor_stats.mean().item():.3g}, "
        f"std={tensor_stats.std().item():.3g}, "
    )
    print(stats)


def get_joint_valid_mask(batch):
    # Find all keys ending with '_valid'
    valid_keys = [k for k in batch.keys() if k.endswith('_valid') and isinstance(batch[k], torch.Tensor)]
    if not valid_keys:
        return None
    # Start with the first mask, then logically AND with others
    joint_mask = batch[valid_keys[0]].bool()
    for k in valid_keys[1:]:
        joint_mask = joint_mask & batch[k].bool()
    return joint_mask

def encode_time_to_sample(sample, index):
    start_ts = datetime.fromtimestamp(index.mint, tz=timezone.utc)
    start_month = start_ts.month
    image = sample['image']

    time_feat = start_month
    # Expand to full image shape (H, W) if needed, e.g., as a constant "band"
    time_band = torch.full_like(image[:1], fill_value=time_feat)  # shape: (1, H, W)
    # Concatenate to input
    image_with_time = torch.cat([image, time_band], dim=0)
    sample["image"] = image_with_time
    # Optionally save the original time_feat for later
    sample["time_feature"] = time_feat
    return sample

def save_gtiff(img, metadata, save_dir, batch_idx, i, crop=None):
    """
    Save a GeoTIFF image from tensor and metadata.
    
    Args:
        img (Tensor): Prediction tensor of shape [C, H, W].
        metadata (dict): Dictionary containing 'crs' and 'bounds' info.
        save_dir (str or Path): Directory to save output.
        batch_idx (int): Batch index for filename.
        i (int): Sample index in batch for filename.
        crop (int or None): Optional number of pixels to crop symmetrically from each edge.
    
    Returns:
        out_path (Path): Path of saved GeoTIFF file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Center crop if requested
    if crop is not None and crop > 0:
        _, H, W = img.shape
        # croph = int(crop / 2)
        cropped_img = img[:, crop:H - crop, crop:W - crop]
        
        # Adjust bounds for cropping (shift by crop pixels converted to spatial units)
        bounds = metadata["bounds"]
        crs = metadata["crs"]
        xres = (bounds["maxx"] - bounds["minx"]) / W
        yres = (bounds["maxy"] - bounds["miny"]) / H
        
        new_bounds = {
            "minx": bounds["minx"] + crop * xres,
            "miny": bounds["miny"] + crop * yres,
            "maxx": bounds["maxx"] - crop * xres,
            "maxy": bounds["maxy"] - crop * yres,
            "mint": bounds.get("mint"),
            "maxt": bounds.get("maxt"),
        }
    else:
        cropped_img = img
        new_bounds = metadata["bounds"]
        crs = metadata["crs"]

    # Convert UNIX timestamps to strings
    mint_ts = new_bounds.get("mint")
    maxt_ts = new_bounds.get("maxt")

    mint_str = datetime.fromtimestamp(mint_ts, timezone.utc).strftime("%Y%m%dT%H%M%S") if mint_ts else "unknown"
    maxt_str = datetime.fromtimestamp(maxt_ts, timezone.utc).strftime("%Y%m%dT%H%M%S") if maxt_ts else "unknown"

    # Calculate affine transform for cropped bounds
    transform = from_bounds(
        new_bounds["minx"], new_bounds["miny"], new_bounds["maxx"], new_bounds["maxy"],
        cropped_img.shape[-1], cropped_img.shape[-2]
    )

    raster_meta = {
        "driver": "GTiff",
        "height": cropped_img.shape[-2],
        "width": cropped_img.shape[-1],
        "count": cropped_img.shape[0],
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
    }

    filename = f"pred_{batch_idx}_{i}_{mint_str}_{maxt_str}.tif"
    out_path = save_dir / filename

    with rasterio.open(out_path, "w", **raster_meta) as dst:
        dst.write(cropped_img.cpu().numpy())

    return out_path


def save_predictions(preds, meta, save_dir, batch_idx, crop):
    outputs = []
    for i in range(preds.shape[0]):
        img = preds[i]  # Single image
        metadata = meta[i]
        save_gtiff(img, metadata, save_dir, batch_idx, i, crop)

        outputs.append({
            "path": str(save_dir),
            "meta": meta,
        })
    
    return outputs

def parse_meta(sample):
    meta = {'crs': str(sample['crs']),
            'bounds': {
                'minx': float(sample['bounds'].minx),
                'miny': float(sample['bounds'].miny),
                'maxx': float(sample['bounds'].maxx),
                'maxy': float(sample['bounds'].maxy),
                'mint': float(sample['bounds'].mint),
                'maxt': float(sample['bounds'].maxt)
            }
        }
    return meta


def extract_patches(x, patch_size, stride):
    """
    Extract overlapping patches from input tensor x using unfold.
    Args:
        x: Tensor of shape [B, C, H, W]
        patch_size: int, size of square patch (kernel size)
        stride: int, stride between patches (controls overlap)
    Returns:
        patches: Tensor of shape [B * n_patches, C, patch_size, patch_size]
        n_patches_h: Number of patches vertically
        n_patches_w: Number of patches horizontally
    """
    B, C, H, W = x.shape
    unfold = torch.nn.Unfold(kernel_size=(patch_size, patch_size), stride=stride)
    patches = unfold(x)  # [B, C * patch_size^2, n_patches]
    n_patches = patches.shape[-1]
    n_patches_h = (H - patch_size) // stride + 1
    n_patches_w = (W - patch_size) // stride + 1

    # Reshape patches to [B, n_patches, C, patch_size, patch_size]
    patches = patches.transpose(1, 2).reshape(B * n_patches, C, patch_size, patch_size)
    return patches, n_patches_h, n_patches_w


def reconstruct_image(patches, n_patches_h, n_patches_w, patch_size, stride, output_size):
    """
    Reconstruct full image by folding overlapping patches and normalizing overlaps.
    Args:
        patches: Tensor of shape [B * n_patches, C, patch_size, patch_size]
        n_patches_h: int
        n_patches_w: int
        patch_size: int
        stride: int
        output_size: tuple (H, W)
    Returns:
        image: Tensor [B, C, H, W] reconstructed
    """
    B = patches.shape[0] // (n_patches_h * n_patches_w)
    C = patches.shape[1]
    # Reshape to [B, n_patches, C*patch_size*patch_size]
    patches = patches.reshape(B, n_patches_h * n_patches_w, C * patch_size * patch_size).transpose(1, 2)
    fold = torch.nn.Fold(
        output_size=output_size,
        kernel_size=(patch_size, patch_size),
        stride=stride
    )
    # Sum overlapping patches
    output = fold(patches)

    # Create divisor mask to normalize overlaps
    ones = torch.ones((B, C, output_size[0], output_size[1]), device=patches.device)
    divisor = fold(torch.nn.functional.unfold(ones, kernel_size=(patch_size, patch_size), stride=stride))
    output /= divisor

    return output