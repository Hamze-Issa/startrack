import argparse
import yaml
import lightning as pl
from models.model_wrapper import ModelFactory
from lightning_wrapper import GenericTask
from datasets import IceVelocity_u, IceVelocity_v, Calving
from custom_geo_data_module import CustomGeoDataModule
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from tools import update_config
import torch
import mlflow
import os
import numpy as np
import rasterio

# def save_predictions_as_geotiff(predictions, input_paths, output_dir):
#     """
#     predictions: list of np.ndarray, (could be [C, H, W] or [H, W]; adjust as needed)
#     input_paths: list of file paths corresponding to each input image
#     output_dir: location to save geotiffs
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     for i, (pred, inp_path) in enumerate(zip(predictions, input_paths)):
#         with rasterio.open(inp_path) as src:
#             meta = src.meta.copy()
#             # Adjust for shape and dtype of pred
#             if pred.ndim == 2:
#                 meta.update(count=1, dtype=pred.dtype)
#                 write_array = pred
#             elif pred.ndim == 3:
#                 meta.update(count=pred.shape, dtype=pred.dtype)
#                 write_array = pred
#             else:
#                 raise ValueError("Unexpected prediction shape: {}".format(pred.shape))
#             out_path = os.path.join(output_dir, f"prediction_{i}.tif")
#             with rasterio.open(out_path, 'w', **meta) as dst:
#                 if pred.ndim == 2:
#                     dst.write(write_array, 1)
#                 else:  # multiple channels/bands
#                     dst.write(write_array)
#         print(f"Saved {out_path}")

def save_predictions_with_metadata(predictions, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, value in enumerate(predictions):
        pred, metadata = value['prediction'], value['metadata']
        profile = {
            'driver': 'GTiff',
            'height': pred.shape[-2],
            'width': pred.shape[-1],
            'count': pred.shape[0] if pred.ndim == 3 else 1,
            'dtype': pred.dtype,
            'crs': metadata.get('crs'),
            'transform': metadata.get('affine'),
        }
        out_path = os.path.join(output_dir, f"prediction_{i}.tif")
        with rasterio.open(out_path, 'w', **profile) as dst:
            if pred.ndim == 2:
                dst.write(pred, 1)
            else:
                dst.write(pred)
        print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Path to config YAML file")
    parser.add_argument("--override", nargs="*", default=[], help="Override config params, e.g., testing.save_predictions=True")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Update overrides
    for override_str in args.override:
        if "=" not in override_str:
            raise ValueError(f"Invalid override argument: {override_str}. Must be key=value format.")
        key, value = override_str.split("=", 1)
        keys = key.split(".")
        update_config(config, keys, value)

    mlflow.set_experiment(config['logging']['experiment_name'])
    mlflow.start_run(run_name=config['logging']['run_name'], tags=config['logging']['tags'],
                     description=config['logging']['description'], log_system_metrics=True)
    run_id = mlflow.active_run().info.run_id
    torch.set_float32_matmul_precision('medium')

    # Load model from checkpoint (assume path is in config['testing']['checkpoint_path'])
    assert os.path.isfile(config['testing']['checkpoint_path']), \
        f"Checkpoint at {config['testing']['checkpoint_path']} not found."
    config['model']['checkpoint_path'] = config['testing']['checkpoint_path']
    model = ModelFactory.create_model(config)
    task = GenericTask(model, config)

    # Prepare test dataset using same dataset construction
    ivu = IceVelocity_u(config['datasets']['ice_velocity_u']['root'])
    ivv = IceVelocity_v(config['datasets']['ice_velocity_v']['root'])
    calving = Calving(config['datasets']['calving']['root'])
    combined_dataset = (ivu & ivv) & calving

    # Create datamodule only for test set split; uses 'test' batch size, num_workers, length, etc.
    datamodule = CustomGeoDataModule(
        dataset=combined_dataset,
        config=config,
        batch_size=config['testing'].get('batch_size', 16),
        patch_size=tuple(config['testing'].get('patch_size', config['training']['patch_size'])),
        length=config['testing'].get('num_samples', 1000),
        num_workers=config['testing'].get('num_workers', 4),
        split_ratios=tuple(config['training']['split_ratios']),
        seed=config['testing'].get('seed', config['training']['seed']),
    )

    mlf_logger = MLFlowLogger(
        experiment_name=config['logging']['experiment_name'],
        run_id=run_id,
        log_model=False,  # No need to log model at inference
    )
    tb_logger = TensorBoardLogger(
        save_dir=config['logging']['tb_log_dir'],
        name=config['logging']['experiment_name']
    )   

    # Inference-specific settings
    limit_test_batches = config['testing'].get('num_samples', 1000) // config['testing'].get('batch_size', 16)

    trainer = pl.Trainer(
        logger=[mlf_logger, tb_logger],
        accelerator=config['testing'].get('accelerator', 'auto'),
        strategy=config['testing'].get('strategy', 'auto'),
        use_distributed_sampler=False,
        log_every_n_steps=config['testing'].get('log_every_n_steps', 10),
        limit_test_batches=limit_test_batches,
        enable_model_summary=False,
        max_epochs=1
    )

    # Run inference on the test set
    predictions = trainer.predict(task, datamodule)

    # # save results
    # if config['testing'].get('save_predictions', True):
    #     output_dir = config['testing'].get('output_dir', './inference_outputs')
    #     save_predictions_with_metadata(predictions, output_dir)
    #     print("Outputs saved to {output_dir}.")
    print(f"Inference complete.")

if __name__ == "__main__":
    main()
