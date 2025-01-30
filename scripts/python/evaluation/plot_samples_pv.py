"""plot generated and real samples

debug with runid: 20240325-5428
"""
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

import numpy as np
import torch

from energydiff.utils.argument_parser import inference_parser
from energydiff.utils.initializer import create_dataset, get_task_profile_condition, get_generated_filename
from energydiff.utils.plot import plot_sampled_data, plot_sampled_data_pv

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
        generated['gmm'] = torch.load(f'generated_data/{get_generated_filename(config, "gmm", 10)}',
                                    map_location='cpu').float() # shape: [sample, 1, sequence]
    except:
        print('gmm not found')
    try:
        generated['copula'] = torch.load(f'generated_data/{get_generated_filename(config, "copula")}',
                                        map_location='cpu').float() # shape: [sample, 1, sequence]
    except:
        print('copula not found')
    generated['ddpm'] = torch.load(f'generated_data/{get_generated_filename(config, "ddpm")}', 
                                   map_location='cpu').float() # shape: [sample, 1 or 3, sequence]
    # generated['trn_sample'] = all_profile['train'][:2000]
    if os.path.exists(f'generated_data/{get_generated_filename(config, "ddpm-calibrated")}'):
        try:
            generated['ddpm-calibrated'] = torch.load(f'generated_data/{get_generated_filename(config, "ddpm-calibrated")}',
                                            map_location='cpu').float()
        except Exception as error:
            print(f'load calibrated failed. \n{error}')
        else:
            print(f'load calibrated successful.')
    else:
        print(f'calibrated data does not exist.')
        
    # denormalize
    target = data.denormalize_fn(target, *scaling_factor).squeeze(1) # [b, l]
    for k, v in generated.items():
        generated[k] = data.denormalize_fn(v, *scaling_factor).squeeze(1) # [b, l]
    
    # plot
    # plot_sampled_data(dataset_each_season={config.data.val_season: target}, 
    #                   save_filepath=f'results/figures/samples/samples_{config.model.load_runid}.pdf',
    #                   samples_each_season={config.data.val_season: generated['ddpm']},
    #                   num_samples_to_plot=10)
    fig = plot_sampled_data_pv(samples={'Real Data': target[:,:1440//3], 'GMM': generated['gmm'][:,:1440//3], 'EnergyDiff': generated['ddpm'][:,:1440//3]},
                         save_filepath=f'results/figures/samples/samples_{config.model.load_runid}.pdf',
                         num_samples_to_plot=100)
    plt.show()
    pass
    
if __name__ == '__main__':
    main()