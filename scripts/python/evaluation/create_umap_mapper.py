import os

import torch
import numpy as np

from energydiff.utils.eval import get_mapper_label
from energydiff.utils.initializer import create_dataset, get_task_profile_condition
from energydiff.utils.eval import FrechetUMAPDistance, MkMMD
from energydiff.utils.argument_parser import inference_parser

PICKLE = False

def main():
    exp_config = inference_parser()
    data_config = exp_config.data
    data_config.vectorize = False # we don't need to vectorize the data for umap mapper.
    
    dataset = create_dataset(exp_config.data)
    profile, condition = get_task_profile_condition(dataset, season=data_config.train_season)
    umap_train_profile = torch.cat([profile['val'], profile['test']], dim=0).squeeze(1).numpy() # shape (num_sample, seq_len)
    
    if PICKLE:
        full_dataset_name = get_mapper_label(data_config)
        fud = FrechetUMAPDistance.from_pickle('results/', label_mapper=full_dataset_name, n_components=1)
    else:
        print(f'umap_train_profile: {umap_train_profile.shape}')
        
        full_dataset_name = get_mapper_label(data_config)
        n_components = 24 if data_config.resolution == '1min' else 24
        
        fud = FrechetUMAPDistance(umap_train_profile, 
                                label_mapper=full_dataset_name,
                                n_components=n_components,
                                )
        fud.save_to_pickle('results/')
        fud_extreme = FrechetUMAPDistance(
            umap_train_profile,
            label_mapper=full_dataset_name,
            n_components=1,
        )
        fud_extreme.save_to_pickle('results/')
        
    # get fud score with the training data. 
    rng = np.random.default_rng(47)
    shuffle = lambda seq: seq[rng.permutation(len(seq))]
    shuffled_seq = shuffle(profile['val'])
    trial_data_source = shuffled_seq[:len(shuffled_seq)//2].squeeze(1).numpy()
    trail_data_target = shuffled_seq[len(shuffled_seq)//2:].squeeze(1).numpy() 
    # trail_data_target += + rng.normal(0, 0.1, size=trail_data_target.shape)
    
    fud_score = fud(trial_data_source, trail_data_target)
    print(f'fud_score: {fud_score}')
    
    mkmmd = MkMMD()
    mkmmd_score = mkmmd(torch.from_numpy(trial_data_source), torch.from_numpy(trail_data_target))
    print(f'mkmmd_score: {mkmmd_score}')
    pass
    
if __name__ == '__main__':
    main()