import os
import logging

import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from einops import rearrange

from energydiff.diffusion.dataset import ConditionalDataset1D
from energydiff.utils.initializer import (
    create_dataset, get_task_profile_condition
)
from energydiff.utils.argument_parser import inference_parser, save_config
from energydiff.utils.initializer import generate_random_id
from energydiff.models.nn_baseline import PLVAE1D

NUM_EPOCHS = 500

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

torch.set_float32_matmul_precision('high')

def reshape(profile):
    "profile: [B C L] -> [B 1 (L C)]"
    return rearrange(profile, 'B C L -> B 1 (L C)')

def main():
    # Load config from the DDPM training run
    config = inference_parser()  # take time id from DDPM run
    save_dir = generate_random_id()
    
    # Setup logging
    run_id = f"train-vae-{config.model.load_time_id}"
    os.makedirs('results', exist_ok=True)
    logging.basicConfig(
        filename=f'results/{run_id}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f"experiment starts: {run_id}")

    # Step 0: Load data using same configuration as DDPM
    dataset_collection = create_dataset(config.data)

    # Get profile and condition data
    all_profile, _ = get_task_profile_condition(
        dataset_collection,
        season=config.data.train_season,
        conditioning=False,  # VAE baseline doesn't use conditioning
    )

    # Create datasets
    trainset = ConditionalDataset1D(
        tensor=reshape(all_profile['train']),
        condition=None,
        transforms=[],
    )
    valset = ConditionalDataset1D(
        tensor=reshape(all_profile['val']),
        condition=None,
    )
    
    # Initialize Lightning module
    downsampling_factor = 3 if (trainset.sequence_length) % (2**4) != 0 else 4
    vae_kl_weight = 0.1
    hidden_dim = 512
    batch_size = 512
    pl_vae = PLVAE1D(
        seq_length=trainset.tensor.shape[-1],
        in_channels=trainset.num_channel,
        bottleneck_channels=64,  # can be made configurable
        hidden_dim=hidden_dim,  # can be made configurable
        downsampling_factor=downsampling_factor,  # can be made configurable
        trainset=trainset,
        valset=valset,
        batch_size=batch_size,
        val_batch_size=batch_size,
        lr=1e-4,  # can be made configurable
        kl_weight=vae_kl_weight,  # can be made configurable
        num_val_samples=64,
    )

    # Setup wandb logger and training
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        name=f"vae-{config.time_id}",
        save_dir='results',
        tags=['baseline', 'vae'],
    )

    # Checkpoint callback - save only the final model
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"results/{run_id}/{save_dir}",
        filename="final_model",
        save_last=True,
    )

    # Initialize trainer
    num_epochs = NUM_EPOCHS
    
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',
        strategy='ddp' if torch.cuda.device_count() > 1 else 'auto',
        max_epochs=num_epochs,  # can be made configurable
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        precision='16-mixed' if torch.cuda.is_available() else '32-true',
    )

    # Log number of parameters
    if trainer.is_global_zero:
        num_params = count_parameters(pl_vae.model)
        wandb_logger.experiment.config.update({
            "num_epochs": num_epochs,
            "model_params": num_params,
            "ddpm_run_id": config.wandb_id,
            "vae_config": {
                "bottleneck_channels": 64,
                "hidden_dim": hidden_dim,
                "downsampling_factor": downsampling_factor,
                "kl_weight": vae_kl_weight,
                "lr": 1e-4,
            }
        })

    # Train
    trainer.fit(pl_vae)
    
    # After training, upload the final model to wandb
    if trainer.is_global_zero:
        artifact = wandb.Artifact(
            name=f'vae-model-{wandb.run.id}',
            type='model',
            description='Trained VAE model'
        )
        artifact.add_file(f"results/{run_id}/{save_dir}/final_model.ckpt")
        wandb.log_artifact(artifact)
        config.vae_wandb_id = wandb.run.id
        save_config(config, config.time_id)
        print("updated config.vae_wandb_id:", config.vae_wandb_id)

    logging.info('Training complete.')
    return config.model.load_time_id

if __name__ == '__main__':
    time_id = main()
    print(f"TIMEID:{time_id}")