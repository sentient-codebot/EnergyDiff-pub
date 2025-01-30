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
from energydiff.models.nn_baseline import PLWGAN1D

NUM_EPOCHS = 500

torch.set_float32_matmul_precision('high')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def reshape(profile):
    "profile: [B C L] -> [B 1 (L C)]"
    return rearrange(profile, 'B C L -> B 1 (L C)')

def main():
    # Load config from the DDPM training run
    config = inference_parser()  # take time id from DDPM run
    save_dir = generate_random_id()
    
    # Setup logging
    run_id = f"train-gan-{config.model.load_time_id}"
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
        conditioning=False,  # GAN baseline doesn't use conditioning
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
    latent_dim = 100
    dim_base = 64
    batch_size = 512
    pl_gan = PLWGAN1D(
        latent_dim=latent_dim,
        out_channels=trainset.num_channel,
        dim_base=dim_base,
        in_channels=trainset.num_channel,
        sequence_length=trainset.sequence_length,
        trainset=trainset,
        valset=valset,
        batch_size=batch_size,
        val_batch_size=batch_size,
        lr=1e-4,  # learning rate
        betas=(0.5, 0.9),  # Adam betas
        n_critic=5,  # number of critic updates per generator update
        lambda_gp=10.0,  # gradient penalty coefficient
        num_val_samples=64,
    )

    # Setup wandb logger and training
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        name=f"gan-{config.time_id}",
        save_dir='results',
        tags=['baseline', 'gan'],
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
        max_epochs=num_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        precision='16-mixed' if torch.cuda.is_available() else '32-true',
    )

    # Log number of parameters and configuration
    if trainer.is_global_zero:
        g_params = count_parameters(pl_gan.generator)
        d_params = count_parameters(pl_gan.discriminator)
        wandb_logger.experiment.config.update({
            "num_epochs": num_epochs,
            "generator_params": g_params,
            "discriminator_params": d_params,
            "total_params": g_params + d_params,
            "ddpm_run_id": config.wandb_id,
            "gan_config": {
                "latent_dim": latent_dim,
                "g_dim_base": dim_base,
                "d_dim_base": dim_base,
                "lr": 1e-4,
                "betas": (0.5, 0.9),
                "n_critic": 5,
                "lambda_gp": 10.0,
            }
        })

    # Train
    trainer.fit(pl_gan)
    
    # After training, upload the final model to wandb
    if trainer.is_global_zero:
        artifact = wandb.Artifact(
            name=f'gan-model-{wandb.run.id}',
            type='model',
            description='Trained WGAN-GP model'
        )
        artifact.add_file(f"results/{run_id}/{save_dir}/final_model.ckpt")
        wandb.log_artifact(artifact)
        config.gan_wandb_id = wandb.run.id
        save_config(config, config.time_id)
        print("updated config.gan_wandb_id:", config.gan_wandb_id)

    logging.info('Training complete.')
    return config.model.load_time_id

if __name__ == '__main__':
    time_id = main()
    print(f"TIMEID:{time_id}")