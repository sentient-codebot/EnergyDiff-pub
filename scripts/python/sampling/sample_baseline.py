import sys
import os
import logging
from datetime import datetime

import numpy as np
import torch
import wandb
from einops import rearrange
from sklearn.mixture import GaussianMixture

from energydiff.utils.argument_parser import inference_parser
from energydiff.utils.initializer import (
    create_dataset, get_task_profile_condition, 
    get_generated_filename
)
from energydiff.diffusion.typing import Float, Data1D
from energydiff.models.ellipitical_copula import EllipticalCopula

N_COMPONENTS = 10
MAX_TRAIN_SIZE = 100000
NUM_SAMPLE = 4000

def pre_copula(dataset: Data1D,) -> Float[np.ndarray, 'batch sequence']:
    'transform data shape for Copula model'
    channel = (dataset.shape[1]-1)//2
    return rearrange(dataset[:,channel,:], 'batch sequence -> sequence batch').cpu().numpy()

def pre_gmm(dataset: Data1D,) -> Float[np.ndarray, 'batch sequence']:
    'transform data shape for GMM model'
    channel = (dataset.shape[1]-1)//2
    return dataset[:,channel,:].cpu().numpy()

def save_data_artifact(
    config,
    model_name: str,
    generated_data: torch.Tensor,
    run,
    gmm_num_components: int=10,
):
    # Save generated samples
    filename = get_generated_filename(config, model=model_name, gmm_num_components=gmm_num_components)
    os.makedirs('generated_data/', exist_ok=True)
    save_path = os.path.join('generated_data/', filename)
    torch.save(generated_data, save_path)
    logging.info(f'Saved to {save_path}')
    
    # Add to wandb as artifact
    generated_samples_artifact = wandb.Artifact(
        name=f'{model_name}_generated_data-{config.wandb_id}',
        type='generated_data',
        metadata=config.sample.to_dict(),
        description='Generated samples from model'
    )
    generated_samples_artifact.add_file(save_path, name='generated_data.pt')
    run.log_artifact(generated_samples_artifact)

def main():
    # Load config from the training run
    config = inference_parser() # take time id
    
    # Load model using wandb
    api = wandb.Api()
    run = wandb.init(project=config.wandb_project, id=config.wandb_id, resume='must')
    
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
    config.data.vectorize = False # no need for baseline models
    dataset_collection = create_dataset(config.data)
    all_profile, _ = get_task_profile_condition(
        dataset_collection,
        season=config.data.train_season,
        conditioning=config.model.conditioning
    )
    
    # Generate samples: GMM
    try:
        logging.info('Generating samples using GMM...')
        train_seq = pre_gmm(all_profile['train'])
        model = GaussianMixture(
            n_components=N_COMPONENTS,
            covariance_type='full',
            max_iter=1000,
        )
        start = datetime.now()
        model.fit(train_seq[:min(MAX_TRAIN_SIZE, len(train_seq))])
        print(model)
        duration = datetime.now() - start
        print(f'Fitting time: {duration}')
        generated = model.sample(NUM_SAMPLE)
        generated = torch.from_numpy(generated[0])
        generated = rearrange(generated, 'batch sequence -> batch 1 sequence')
        
        save_data_artifact(config, 'gmm', generated, run, N_COMPONENTS)
    except Exception as e:
        logging.error(f'Error generating samples using GMM model: {e}')
    
    # Generate samples: Copula
    try:
        logging.info('Generating samples using t-Copula...')
        train_seq = pre_copula(all_profile['train'])
        model = EllipticalCopula(
            data_frame = train_seq[:min(MAX_TRAIN_SIZE, len(train_seq))],
            copula_type='t',
            interpolation='linear'
        )
        print(f'Fitting model for {config.data.dataset}')
        start = datetime.now()
        model.fit()
        print(model)
        duration = datetime.now() - start
        print(f'Fitting time: {duration}')
        generated = model.sample(NUM_SAMPLE)
        generated = torch.from_numpy(generated)
        generated = rearrange(generated, 'sequence batch -> batch 1 sequence')
        
        save_data_artifact(config, 'copula', generated, run)
    except Exception as e:
        logging.error(f'Error generating samples using Copula model: {e}')
    
    # Save run ID for test script
    with open('last_sampled.txt', 'w') as f:
        f.write(time_id)
        
    return time_id

if __name__ == '__main__':
    time_id = main()
    print(f"TIMEID:{time_id}")