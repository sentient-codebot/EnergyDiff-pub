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
from energydiff.diffusion.dpm_solver import DPMSolverSampler
from energydiff.utils.sample import dpm_solver_sample, ancestral_sample
from energydiff.diffusion import PLDiffusion1D
from energydiff.marginal_carlibrate import calibrate

def main():
    # Load config from the training run
    config = inference_parser() # take time id
    
    # Load model using wandb
    api = wandb.Api()
    run = wandb.init(project=config.wandb_project, id=config.wandb_id, resume='must')
    
    # Find the last checkpoint file
    checkpoint_artifact = api.artifact(name=f"{config.wandb_project}/model-{config.wandb_id}:latest") # or best
    ckpt_dir = checkpoint_artifact.download()
    
    # Load trained model
    pl_model = PLDiffusion1D.load_from_checkpoint(ckpt_dir+'/model.ckpt')
    pl_model.eval()
    
    # Get the run ID from training
    time_id = config.model.load_time_id
    if config.model.conditioning:
        run_id = f"train-diffusion-{config.diffusion.prediction_type}-cond-{time_id}"
    else:
        run_id = f"train-diffusion-{config.diffusion.prediction_type}-{config.data.train_season}-{time_id}"
    
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
        conditioning=config.model.conditioning
    )
    
    # Generate samples
    logging.info('Generating samples...')
    if config.sample.dpm_solver_sample and not config.diffusion.use_rectified_flow:
        dpm_sampler = DPMSolverSampler(pl_model.ema.ema_model)
        generated_samples = dpm_solver_sample(
            dpm_sampler,
            total_num_sample=config.sample.num_sample,
            batch_size=config.sample.val_batch_size,
            step=config.sample.num_sampling_step,
            shape=(pl_model.trainset.num_channel, pl_model.trainset.sequence_length),
            conditioning=None,
            cfg_scale=config.sample.cfg_scale,
            accelerator=None,
        )
    else:
        generated_samples = ancestral_sample(
            config.sample.num_sample,
            config.sample.val_batch_size,
            cond=None,
            cfg_scale=config.sample.cfg_scale,
            model=pl_model.ema.ema_model,
        )
    
    # Save generated samples
    if config.data.vectorize:
        generated_samples = dataset_collection.inverse_vectorize_fn(
            generated_samples, 
            style=config.data.style_vectorize
        )
    
    filename = get_generated_filename(config, model='ddpm')
    os.makedirs('generated_data/', exist_ok=True)
    save_path = os.path.join('generated_data/', filename)
    torch.save(generated_samples, save_path)
    logging.info(f'Saved to {save_path}')
    
    # Add to wandb as artifact
    generated_samples_artifact = wandb.Artifact(
        name=f'ddpm_generated_data-{config.wandb_id}',
        type='generated_data',
        metadata=config.sample.to_dict(),
        description='Generated samples from model'
    )
    generated_samples_artifact.add_file(save_path, name='generated_data.pt')
    run.log_artifact(generated_samples_artifact)
    
    # Calibrate generated samples
    # IMPORTANT need to align the shape of two profiles before calibration
    _train_profile = rearrange(all_profile['train'], 'B C L -> B 1 (L C)')
    generated_calibrated = calibrate(_train_profile, generated_samples)
    torch.save(generated_calibrated, f'generated_data/{get_generated_filename(config, "ddpm-calibrated")}')
    logging.info(f'Saved calibrated samples to generated_data/{get_generated_filename(config, "ddpm-calibrated")}')

    # Add to wandb as artifact
    generated_calibrated_artifact = wandb.Artifact(
        name=f'ddpm_calibrated_generated_data-{config.wandb_id}',
        type='generated_data',
        metadata=config.sample.to_dict(),
        description='Calibrated generated samples from model'
    )
    generated_calibrated_artifact.add_file(f'generated_data/{get_generated_filename(config, "ddpm-calibrated")}', name='generated_data.pt')
    run.log_artifact(generated_calibrated_artifact)

    # Save run ID for test script
    with open('last_sampled.txt', 'w') as f:
        f.write(time_id)
        
    return time_id

if __name__ == '__main__':
    time_id = main()
    print(f"TIMEID:{time_id}")
    sys.exit(0)