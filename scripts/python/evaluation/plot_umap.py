"""plot generated and real samples

debug with runid: 20240325-5428
"""
#%%
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

import numpy as np
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

from umap import UMAP

from energydiff.utils.argument_parser import inference_parser
from energydiff.utils.initializer import create_dataset, get_task_profile_condition, get_generated_filename
from energydiff.utils.plot import plot_embedding_comparison

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
    target = torch.cat([all_profile['test'], all_profile['val'], all_profile['train']], dim=0) # [b, 1, l]
    # target = all_profile['train']
    
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
        generated['copula'] = torch.load(f'generated_data/{get_generated_filename(config, "copula")}',
                                        map_location='cpu').float() # shape: [sample, 1, sequence]
    except:
        print('copula not found')
    generated['EnergyDiff'] = torch.load(f'generated_data/{get_generated_filename(config, "ddpm")}', 
                                   map_location='cpu').float() # shape: [sample, 1 or 3, sequence]
    # generated['trn_sample'] = all_profile['train'][:2000]
    # if os.path.exists(f'generated_data/{get_generated_filename(config, "ddpm-calibrated")}'):
    #     try:
    #         generated['Ours-calibrated'] = torch.load(f'generated_data/{get_generated_filename(config, "ddpm-calibrated")}',
    #                                         map_location='cpu').float()
    #     except Exception as error:
    #         print(f'load calibrated failed. \n{error}')
    #     else:
    #         print(f'load calibrated successful.')
    # else:
    #     print(f'calibrated data does not exist.')
        
    # denormalize
    for k, v in generated.items():
        generated[k] = data.denormalize_fn(v, *scaling_factor).squeeze(1).cpu().numpy()[:1000] # [b, l]
    generated['Real Data'] = data.denormalize_fn(target, *scaling_factor).squeeze(1) # [b, l]
    
    #%% umap
    emb_type = 'UMAP'
    emb_model = UMAP(
        n_components=2,
    )
    emb_model.fit(generated['Real Data'])
    emb_data = {}
    for k, v in generated.items():
        emb_data[k] = emb_model.transform(v)
        
    #%% tsne
    # emb_type = 't-SNE'
    # emb_model = TSNE(n_components=2, random_state=42, perplexity=10)
    # all_data = np.concatenate([v for v in generated.values()], axis=0)
    # _split = [v.shape[0] for v in generated.values()]
    # _split = np.cumsum(_split)[:-1]
    # _emb_data = emb_model.fit_transform(all_data)
    # _emb_data = np.split(_emb_data, _split, axis=0)
    # emb_data = {}
    # for k, v in generated.items():
    #     emb_data[k] = _emb_data.pop(0)
        
    #%% plot
    plot_embedding_comparison(embedding_dict=emb_data, embedding_type=emb_type, 
                              save_dir='results/figures/embeddings/', 
                              save_label=config.model.load_runid,
                              max_num_point=1500,
                              )
    
    
if __name__ == '__main__':
    main()