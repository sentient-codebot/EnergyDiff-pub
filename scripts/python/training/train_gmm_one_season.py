""" train a gmm model, sample new data and save in shape (B, 1, L). 

    saved data is   - normalized 
                    - (B, 1, L) 
"""
import os
from datetime import datetime
from copy import deepcopy

import torch
import numpy as np
from einops import rearrange

from energydiff.dataset.utils import NAME_SEASONS, PIT, standard_normal_cdf, standard_normal_icdf
from energydiff.diffusion.typing import Float, Data1D
from sklearn.mixture import GaussianMixture
from energydiff.utils import generate_time_id
from energydiff.utils.argument_parser import inference_parser, save_config
from energydiff.utils.initializer import create_dataset, get_task_profile_condition, get_generated_filename

def pre_gmm(dataset: Data1D,) -> Float[np.ndarray, 'batch sequence']:
    'transform data shape for GMM model'
    channel = (dataset.shape[1]-1)//2
    return dataset[:,channel,:].cpu().numpy()

def main():
    # Step 0: Parse arguments
    MAX_TRAIN_SIZE = 40000
    N_COMPONENTS = 10
    config = inference_parser()
    time_id = generate_time_id()
    season = config.data.train_season
    NUM_SAMPLE = config.sample.num_sample
    
    # Step 1: Load data
    config.data.vectorize = False
    data = create_dataset(config.data)
    pre_transforms = {}
    post_transforms = {}
    pit: PIT = data.dataset.pit
    if pit is not None:
        pre_transforms['pit'] = pit.transform
        pre_transforms['erf'] = standard_normal_icdf
        post_transforms['erf'] = standard_normal_cdf
        post_transforms['pit'] = pit.inverse_transform
        foobar = deepcopy(config.data)
        foobar.pit = False
        data = create_dataset(foobar)
    all_profile, all_condition = get_task_profile_condition(data, season=season, conditioning=False)
    train_seq = all_profile['train']
    for fn_name, trans_fn in pre_transforms.items():
        train_seq = trans_fn(train_seq)
    # Step 2: Set up model
    model = GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type='full',
        max_iter=1000,
    )
    
    # Step 3: Fit model
    print("**************************************************************************")
    print(f'Fitting GMM model for {season}')
    start = datetime.now()
    model.fit(pre_gmm(train_seq[:min(MAX_TRAIN_SIZE, len(train_seq))]))
    print(model)
    duration = datetime.now() - start
    print(f'Fitting time: {duration}')
    
    # Step 4: Sample from model
    generated = model.sample(NUM_SAMPLE)
    generated = torch.from_numpy(generated[0])
    generated = rearrange(generated, 'batch sequence -> batch 1 sequence')
    for trans_name, trans_fn in post_transforms.items():
        generated = trans_fn(generated)
    
    # Step 5: Save generated data
    filename = get_generated_filename(config, 'gmm', N_COMPONENTS)
    torch.save(generated, os.path.join('generated_data', filename))
    print(f'Saved generated data to generated_data/{filename}')
    save_config(config, time_id)
    print(f'Saved configuration to results/configs/exp_config_{time_id}.yaml')
    
if __name__ == '__main__':
    main()