"""plot generated and real sample histograms

debug with runid: 20240325-5428
"""
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

import numpy as np
import torch
from einops import rearrange

from energydiff.utils.argument_parser import inference_parser
from energydiff.utils.initializer import create_dataset, get_task_profile_condition, get_generated_filename
from energydiff.utils.plot import plot_scatter_marginal_2d

method_find_two_steps = 'select' 
assert method_find_two_steps in ['average', 'select']

def main():
    config = inference_parser()
    if config.model.load_runid is None:
        raise ValueError('please specify runid')
    target_dataset = config.data.dataset
    load_runid = config.model.load_runid 
    season = config.data.val_season
    resolution = config.data.resolution
    data = create_dataset(config.data)
    scaling_factor = data.dataset.scaling_factor
    print('dataset scaling factor', data.dataset.scaling_factor)
    
    print('**************************************************************')
    print(f'Test for {target_dataset} dataset, resolution: {config.data.resolution}, season: {season}')
    print('**************************************************************')
    
    all_profile, all_condition = get_task_profile_condition(data, season=season, conditioning=False)
    target = all_profile['test']
    
    if config.data.vectorize:
        target = data.inverse_vectorize_fn(target, style=config.data.style_vectorize)
    
    # load synthetic
    generated = {}
    try:
        generated['GMM'] = torch.load(f'generated_data/{get_generated_filename(config, "gmm", 10)}',
                                    map_location='cpu').float() # shape: [sample, 1, sequence]
    except:
        print('gmm not found')
    try:
        generated['t-Copula'] = torch.load(f'generated_data/{get_generated_filename(config, "copula")}',
                                        map_location='cpu').float() # shape: [sample, 1, sequence]
    except:
        print('copula not found')
    generated['EnergyDiff'] = torch.load(f'generated_data/{get_generated_filename(config, "ddpm")}', 
                                   map_location='cpu').float() # shape: [sample, 1 or 3, sequence]
    # generated['trn_sample'] = all_profile['train'][:2000]
    if os.path.exists(f'generated_data/{get_generated_filename(config, "ddpm-calibrated")}'):
        try:
            generated['EnergyDiff (Calibrated)'] = torch.load(f'generated_data/{get_generated_filename(config, "ddpm-calibrated")}',
                                            map_location='cpu').float()
        except Exception as error:
            print(f'load calibrated failed. \n{error}')
        else:
            print(f'load calibrated successful.')
    else:
        print(f'calibrated data does not exist.')
        
    # denormalize
    for k, v in generated.items():
        generated[k] = data.denormalize_fn(v, *scaling_factor).squeeze(1).cpu().numpy() # [b, l]
    generated['Real Data'] = data.denormalize_fn(target, *scaling_factor).squeeze(1).cpu().numpy() # [b, l]
    
    # remove unnecessary items
    if 'GMM' in generated:
        generated.pop('GMM')
    if 't-Copula' in generated:
        generated.pop('t-Copula')
        
    if config.model.load_runid == '20240402-4210':
        # it's in fact GMM data. "pretending" it's our model (for convenience). 
        # now we change its name back
        generated['GMM'] = generated.pop('EnergyDiff')
        generated['GMM (Calibrated)'] = generated.pop('EnergyDiff (Calibrated)')
        generated['Real Data'] = generated.pop('Real Data') # just to keep the order consistent
    

    if method_find_two_steps == 'average':
    # choice 1: average to 2d
        for k, v in generated.items():
            generated[k] = np.mean(rearrange(v, 'b (seg l) -> b seg l', seg=2), axis=-1) # [b, seg]
    elif method_find_two_steps == 'select':
    # choice 2: select two steps
        for k, v in generated.items():
            generated[k] = v[:, 114:116]
    else:
        pass
    
    # plot
    fig = plot_scatter_marginal_2d(
        dict_samples=generated,
        xlabel='Power [-]',
        ylabel='Power [-]',
        save_dir='results/figures/scatter_marginal/',
        save_label=f'_{config.model.load_runid}',
    )
    print('complete.')
    
    
if __name__ == '__main__':
    main()