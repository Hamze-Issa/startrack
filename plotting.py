import matplotlib.pyplot as plt
import numpy as np

try:
    import cmocean
    ICE_CMAP = cmocean.cm.ice
except ImportError:
    ICE_CMAP = "Blues"

# Possible colormaps for various plots
# plt.imshow(data, cmap='viridis')         # General
# plt.imshow(data, cmap='YlGn')            # Vegetation
# plt.imshow(data, cmap='inferno')         # Thermal
# plt.imshow(data, cmap='cmocean.cm.thermal')  # Thermal (cmocean)

# import cmocean

# plt.imshow(data, cmap=cmocean.cm.thermal)       # Thermal
# plt.imshow(data, cmap=cmocean.cm.algae)         # Vegetation
# plt.imshow(data, cmap=cmocean.cm.rain)          # General/rainfall


def create_multichannel_figure(
    input_data,                # numpy array (C, H, W) or (H, W)
    mask_data,                 # numpy array (C2, H, W) or (H, W)
    pred_data,                 # numpy array (C3, H, W) or (H, W)
    input_channel_names=None,  # list of input channel names, length C
    mask_channel_names=None,   # list of mask channel names, length C2
    pred_channel_names=None,   # list of pred channel names, length C3
    cmap='viridis'               # You can change colormap (e.g. ICE_CMAP)
):
    # Expand single-channel arrays to (1, H, W)
    if input_data.ndim == 2:
        input_data = input_data[np.newaxis, ...]
    if mask_data.ndim == 2:
        mask_data = mask_data[np.newaxis, ...]
    if pred_data.ndim == 2:
        pred_data = pred_data[np.newaxis, ...]

    num_input = input_data.shape[0]
    num_mask = mask_data.shape[0]
    num_pred = pred_data.shape[0]

    # Default names if needed
    if input_channel_names is None or len(input_channel_names) != num_input:
        input_channel_names = [f"Input {i+1}" for i in range(num_input)]
    if mask_channel_names is None or len(mask_channel_names) != num_mask:
        mask_channel_names = [f"Mask {i+1}" for i in range(num_mask)]
    if pred_channel_names is None or len(pred_channel_names) != num_pred:
        pred_channel_names = [f"Prediction {i+1}" for i in range(num_pred)]

    # Arrange all channels side by side
    ncols = num_input + num_mask + num_pred
    fig, axs = plt.subplots(1, ncols, figsize=(6 * ncols, 6))

    # Plot inputs
    for i in range(num_input):
        im = axs[i].imshow(input_data[i], cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        axs[i].set_title(input_channel_names[i], fontsize=16)
        axs[i].axis('off')
        cb = fig.colorbar(im, ax=axs[i], orientation='vertical', pad=0.01)
        cb.set_label('Value', fontsize=12)

    # Plot masks
    for i in range(num_mask):
        idx = num_input + i
        im_mask = axs[idx].imshow(mask_data[i], cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        axs[idx].set_title(mask_channel_names[i], fontsize=16)
        axs[idx].axis('off')
        cb = fig.colorbar(im_mask, ax=axs[idx], orientation='vertical', pad=0.01)
        cb.set_label('Value', fontsize=12)

    # Plot predictions
    for i in range(num_pred):
        idx = num_input + num_mask + i
        im_pred = axs[idx].imshow(pred_data[i], cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        axs[idx].set_title(pred_channel_names[i], fontsize=16)
        axs[idx].axis('off')
        cb = fig.colorbar(im_pred, ax=axs[idx], orientation='vertical', pad=0.01)
        cb.set_label('Value', fontsize=12)

    plt.tight_layout()
    return fig
