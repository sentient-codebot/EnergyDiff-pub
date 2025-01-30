""" train a copula model, sample new data and save in shape (B, 1, L). 

    saved data is   - normalized 
                    - (B, 1, L) 
"""
import os
from datetime import datetime
import random

import torch
import numpy as np
from einops import rearrange

from energydiff.utils.initializer import get_task_profile_condition
from energydiff.dataset.utils import NAME_SEASONS
from energydiff.diffusion.typing import Float, Data1D
from energydiff.models.ellipitical_copula import EllipticalCopula
from energydiff.utils import generate_time_id
from energydiff.utils.argument_parser import inference_parser, save_config
from energydiff.utils.initializer import create_dataset, get_generated_filename

RESCALE = True # rescale data with mean and variance

def pre_copula(dataset: Data1D,) -> Float[np.ndarray, 'batch sequence']:
    'transform data shape for Copula model'
    channel = (dataset.shape[1]-1)//2
    return rearrange(dataset[:,channel,:], 'batch sequence -> sequence batch').cpu().numpy()

def main():
    # Step 0: Parse arguments
    MAX_TRAIN_SIZE = 10000 
    config = inference_parser()
    time_id = generate_time_id()
    season = config.data.train_season
    NUM_SAMPLE = config.sample.num_sample
    
    # Step 1: Load data
    config.data.vectorize = False
    data = create_dataset(config.data)
    all_profile, all_condition = get_task_profile_condition(data, season=season, conditioning=False)
    train_seq = pre_copula(all_profile['train'])
    # Step 2: Set up model
    if RESCALE:
        mean, std = train_seq.mean(), train_seq.std()
        train_seq = (train_seq - mean) / (std + 1e-7)
    
    model = EllipticalCopula(
        data_frame = train_seq[:min(MAX_TRAIN_SIZE, len(train_seq))],
        copula_type='t',
        interpolation='linear'
    )
    
    # Step 3: Fit model
    print("**************************************************************************")
    print(f'Fitting model for {config.data.dataset} {season}')
    start = datetime.now()
    model.fit()
    print(model)
    duration = datetime.now() - start
    print(f'Fitting time: {duration}')
    
    # Step 4: Sample from model
    generated = model.sample(NUM_SAMPLE)
    if RESCALE:
        generated = generated * (std+1e-7) + mean
    
    generated = torch.from_numpy(generated)
    generated = rearrange(generated, 'sequence batch -> batch 1 sequence')
    
    # Step 5: Save generated data
    filename = get_generated_filename(config, 'copula')
    torch.save(generated, os.path.join('generated_data', filename))
    print(f'Saved generated data to generated_data/{filename}')
    save_config(config, time_id)
    print(f'Saved configuration to results/configs/exp_config_{time_id}.yaml')
    
if __name__ == '__main__':
    main()