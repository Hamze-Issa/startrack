import argparse
import yaml
import lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models.model_wrapper import ModelFactory
from lightning_wrapper import GenericTask
from lightning.pytorch.loggers import TensorBoardLogger
from datasets import DATASET_CLASSES
from custom_geo_data_module import CustomGeoDataModule
from lightning.pytorch.loggers import MLFlowLogger
from tools import update_config
import torch
import mlflow
import os

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

    dataset_objs = {}
    for key, dataset_config in config['datasets'].items():
        if key == 'combination':
            continue
        dataset_cls = DATASET_CLASSES[dataset_config['class']]
        # Pass all config values as kwargs except 'class'
        kwargs = {k: v for k, v in dataset_config.items() if k not in ['class']}
        kwargs['keep_meta'] = True
        dataset_objs[dataset_config['name']] = dataset_cls(**kwargs)

    comb_str = config['datasets']['combination']
    combined_dataset = eval(comb_str, {}, dataset_objs)

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

    print(f"Inference complete.")

if __name__ == "__main__":
    main()
