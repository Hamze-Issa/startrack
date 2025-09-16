import argparse
import yaml
import lightning as pl
from models.model_wrapper import ModelFactory
from lightning_wrapper import GenericTask
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from datasets import DATASET_CLASSES
from custom_geo_data_module import CustomGeoDataModule
from lightning.pytorch.loggers import MLFlowLogger
from tools import update_config
import torch
import mlflow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Path to config YAML file")
    # Accept overriding config params as a list of dot.notation=value strings
    parser.add_argument("--override", nargs="*", default=[], help="Override config params, e.g., training.max_epochs=10")
    args = parser.parse_args()

    # Load base config from file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override config params from command line
    for override_str in args.override:
        if "=" not in override_str:
            raise ValueError(f"Invalid override argument: {override_str}. Must be key=value format.")
        key, value = override_str.split("=", 1)
        keys = key.split(".")
        update_config(config, keys, value)

    # Setup for MLFlow logger and Torch precision
    mlflow.set_experiment(config['logging']['experiment_name'])
    mlflow.start_run(run_name=config['logging']['run_name'], tags=config['logging']['tags'],
                     description=config['logging']['description'], log_system_metrics=True)
    run_id = mlflow.active_run().info.run_id
    torch.set_float32_matmul_precision('medium')  # for more details on this see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

    # Initialize model
    model = ModelFactory.create_model(config)
    task = GenericTask(model, config)

    dataset_objs = {}
    for key, dataset_config in config['datasets'].items():
        if key == 'combination':
            continue
        dataset_cls = DATASET_CLASSES[dataset_config['class']]
        # Pass all config values as kwargs except 'class'
        kwargs = {k: v for k, v in dataset_config.items() if k not in ['class']}
        dataset_objs[dataset_config['name']] = dataset_cls(**kwargs)

    comb_str = config['datasets']['combination']
    combined_dataset = eval(comb_str, {}, dataset_objs)

    # Create datamodule
    datamodule = CustomGeoDataModule(
        dataset=combined_dataset,
        config=config,
        batch_size=config['training']['batch_size'],
        patch_size=tuple(config['training']['patch_size']),
        length=config['training']['samples_per_epoch'],
        num_workers=config['training']['num_workers'],
        split_ratios=tuple(config['training']['split_ratios']),
        seed=config['training']['seed'],
    )

    # Initialize logging 
    mlf_logger = MLFlowLogger(
        experiment_name=config['logging']['experiment_name'],
        run_id=run_id,
        log_model=True,
    )
    tb_logger = TensorBoardLogger(
        save_dir=config['logging']['tb_log_dir'],
        name=config['logging']['experiment_name']
    )

    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        log_every_n_steps=config['logging']['train_log_every_n_steps'],
        check_val_every_n_epoch=1,
        val_check_interval=config['logging']['val_log_every_n_steps'],
        limit_val_batches=config['training']['val_samples'] // config['training']['batch_size'],
        callbacks=[
            ModelCheckpoint(
                monitor=config['logging']['monitor'],
                save_top_k=config['logging']['save_top_k'],
                every_n_epochs=config['logging']['model_log_every_n_epochs']
            ),
            EarlyStopping(
                monitor=config['logging']['monitor'],
                patience=config['logging']['patience'],
            )
        ],
        logger=[mlf_logger], # you can add also the tensorboard logger (tb_logger) to this list if you want
        strategy="auto",
        accelerator="auto",
        use_distributed_sampler=False,
    )

    # Start training
    trainer.fit(task, datamodule=datamodule)


if __name__ == "__main__":
    main()
