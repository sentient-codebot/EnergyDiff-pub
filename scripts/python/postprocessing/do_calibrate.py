r"""
Test all metrics for three models: GMM, copula, DDPM. 

Required arguments:
    --dataset: dataset name, should be one of ["heat_pump", "lcl_electricity"]
    --data_root: root directory of data
    --resolution: (required for heat pump, so far) resolution of data, one of [10s, 1min, 15min, 30min, 1h]
    --val_season: season of validation data, winter/spring/.../whole_year
    --load_runid: runid of DDPM model to load
    --num_sampling_step: number of sampling steps of DDPM model to load
    --num_diffusion_step: number of diffusion steps of DDPM model to load
    -- **all the process options to create the dataset, used to get the target data.**
        **they need to strictly the same as the options used to train the DDPM model.**
        **can be conveniently loaded by specifying a config file.**

(new) ddpm sample name:
"DDPM_{RUNID}_samples_{SEASON}_{SAMPLING_STEP}_{DIFFUSION_STEP}.pt"
"""
import os
import torch
import numpy as np
from energydiff.utils.argument_parser import inference_parser
from energydiff.utils.initializer import create_dataset, get_task_profile_condition, get_generated_filename
from energydiff.marginal_carlibrate import calibrate

def normalize(raw_data, mean, std):
    return (raw_data - mean) / (std+1e-8)

BALANCE_NUM_SAMPLE = False # use the same number of samples for target and source

COMBINE_VAL_TEST = True
SKIP_BASELINE = False

ENABLE_CUDA = False
LOAD_DATA = True
DENORMALIZE = False
NORMALIZE = False
PIT_DATA = False
FAST_RUN = False # randomly select 2000 samples from target data

def main():
    config = inference_parser()
    if config.model.load_runid is None:
        raise ValueError('please specify runid')
    target_dataset = config.data.dataset
    load_runid = config.model.load_runid 
    season = config.data.val_season
    resolution = config.data.resolution
    data = create_dataset(config.data)
    print('dataset scaling factor', data.dataset.scaling_factor)
    
    print('**************************************************************')
    print(f'Test for {target_dataset} dataset, resolution: {config.data.resolution}, season: {season}')
    print('**************************************************************')
    
    all_profile, all_condition = get_task_profile_condition(data, season=season, conditioning=False)
    target = all_profile['train']
    
    generated = {}
    
    generated['ddpm'] = torch.load(f'generated_data/{get_generated_filename(config, "ddpm")}', 
                                   map_location='cuda' if ENABLE_CUDA else 'cpu').float() # shape: [sample, 1 or 3, sequence]
    # generated['trn_sample'] = all_profile['train'][:2000]
    if os.path.exists(f'generated_data/{get_generated_filename(config, "ddpm-calibrated")}'):
        try:
            generated['ddpm-calibrated'] = torch.load(f'generated_data/{get_generated_filename(config, "ddpm-calibrated")}',
                                            map_location='cuda' if ENABLE_CUDA else 'cpu').float()
        except:
            print(f'load existing calibrated data failed. regenerating...')
        else:
            print(f"before:\t({generated['ddpm'].min().item():.4f}, {generated['ddpm'].max().item():.4f})")
            print(f"after:\t({generated['ddpm-calibrated'].min().item():.4f}, {generated['ddpm-calibrated'].max().item():.4f})")
            print(f'load existing calibrated data successful. complete.')
            return 
    
    generated['ddpm-calibrated'] = calibrate(target, generated['ddpm'])
    torch.save(generated['ddpm-calibrated'], f'generated_data/{get_generated_filename(config, "ddpm-calibrated")}')
    print(f'calibrated data saved. complete.')
        
if __name__ == '__main__':
    main()