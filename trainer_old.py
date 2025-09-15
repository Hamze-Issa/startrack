import yaml
import lightning as pl
from models.model_wrapper import ModelFactory
from lightning_wrapper import GenericSegmentationTask
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from datasets import IceVelocity_u, IceVelocity_v, Calving
from custom_geo_data_module import CustomGeoDataModule
from lightning.pytorch.loggers import MLFlowLogger
import torch
import mlflow

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Setup for MLFlow logger and Torch precision
mlflow.set_experiment(config['logging']['experiment_name'])
mlflow.start_run(run_name=config['logging']['run_name'], tags=config['logging']['tags'], description=config['logging']['description'], log_system_metrics=True)
run_id = mlflow.active_run().info.run_id
torch.set_float32_matmul_precision('medium') # for more details on this see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

# Initialize model
model = ModelFactory.create_model(config)
task = GenericSegmentationTask(model, config)

# Setup datasets
ivu = IceVelocity_u(config['datasets']['ice_velocity_u']['root'])
ivv = IceVelocity_v(config['datasets']['ice_velocity_v']['root'])
calving = Calving(config['datasets']['calving']['root'])
combined_dataset = (ivu & ivv) & calving

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
    check_val_every_n_epoch=1,  # Validate once per epoch
    val_check_interval=config['logging']['val_log_every_n_steps'],
    limit_val_batches=config['training']['val_samples'] // config['training']['batch_size'],
    callbacks=[
        ModelCheckpoint(
            monitor=config['logging']['monitor'],
            save_top_k=config['logging']['save_top_k'],
            every_n_epochs=config['logging']['model_log_every_n_epochs']  # checkpoint every N epochs
        ),
        EarlyStopping(
            monitor=config['logging']['monitor'],
            patience=config['logging']['patience'],
        )
    ],
    logger=[mlf_logger, tb_logger],
    strategy="auto", #"ddp_find_unused_parameters_false",
    accelerator="auto",
    use_distributed_sampler=False,
)

# Start training
trainer.fit(task, datamodule=datamodule)

        

