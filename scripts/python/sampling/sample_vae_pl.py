import sys
import os
import logging

import torch
import wandb
from einops import rearrange

from energydiff.utils.argument_parser import inference_parser
from energydiff.utils.initializer import (
    create_dataset, get_task_profile_condition, 
    get_generated_filename
)
from energydiff.models.nn_baseline import PLVAE1D
from energydiff.marginal_carlibrate import calibrate

def main():
    # Load config from the training run
    config = inference_parser()  # take time id
    
    # Load VAE model using wandb
    api = wandb.Api()
    run = wandb.init(project=config.wandb_project, id=config.wandb_id, resume='must')
    
    # Find VAE checkpoint file
    assert hasattr(config, 'vae_wandb_id'), "No VAE wandb ID found in config"
    checkpoint_artifact = api.artifact(name=f"{config.wandb_project}/vae-model-{config.vae_wandb_id}:latest")
    print("checkpoint_artifact:", checkpoint_artifact.name)
    ckpt_dir = checkpoint_artifact.download()
    
    # Load trained model
    pl_model = PLVAE1D.load_from_checkpoint(os.path.join(ckpt_dir, 'final_model.ckpt'))
    pl_model.eval()
    
    # Get the run ID from training
    time_id = config.model.load_time_id
    run_id = f"train-vae-baseline-{time_id}"
    
    # Setup logging
    os.makedirs('results', exist_ok=True)
    logging.basicConfig(
        filename=f'results/sample_{run_id}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    
    # Load dataset and model
    dataset_collection = create_dataset(config.data)
    all_profile, _ = get_task_profile_condition(
        dataset_collection,
        season=config.data.train_season,
        conditioning=False
    )
    
    # Generate samples
    logging.info('Generating samples...')
    with torch.no_grad():
        # Sample from standard normal
        generated_samples = pl_model.generate_samples(config.sample.num_sample)
    
    # Save generated samples
    if config.data.vectorize:
        generated_samples = dataset_collection.inverse_vectorize_fn(
            generated_samples,
            style=config.data.style_vectorize
        )
    
    filename = get_generated_filename(config, model='vae')
    os.makedirs('generated_data/', exist_ok=True)
    save_path = os.path.join('generated_data/', filename)
    torch.save(generated_samples, save_path)
    logging.info(f'Saved to {save_path}')
    
    # Add to wandb as artifact (to original DDPM run)
    generated_samples_artifact = wandb.Artifact(
        name=f'vae_generated_data-{config.wandb_id}',
        type='generated_data',
        metadata=config.sample.to_dict(),
        description='Generated samples from VAE model'
    )
    generated_samples_artifact.add_file(save_path, name='generated_data.pt')
    run.log_artifact(generated_samples_artifact)
    
    # Calibrate generated samples
    _train_profile = rearrange(all_profile['train'], 'B C L -> B 1 (L C)')
    generated_calibrated = calibrate(_train_profile, generated_samples)
    calibrated_path = f'generated_data/{get_generated_filename(config, "vae-calibrated")}'
    torch.save(generated_calibrated, calibrated_path)
    logging.info(f'Saved calibrated samples to {calibrated_path}')

    # Add calibrated samples to wandb as artifact
    generated_calibrated_artifact = wandb.Artifact(
        name=f'vae_calibrated_generated_data-{config.wandb_id}',
        type='generated_data',
        metadata=config.sample.to_dict(),
        description='Calibrated generated samples from VAE model'
    )
    generated_calibrated_artifact.add_file(calibrated_path, name='generated_data.pt')
    run.log_artifact(generated_calibrated_artifact)

    with open('last_sampled.txt', 'w') as f:
        f.write(time_id)
        
    return time_id

if __name__ == '__main__':
    time_id = main()
    print(f"TIMEID:{time_id}")
    sys.exit(0)