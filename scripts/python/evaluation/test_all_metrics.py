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
from einops import rearrange
from energydiff.dataset import NAME_SEASONS
from energydiff.utils.eval import MkMMD, kl_divergence, ws_distance, ks_2samp_test, only_central_dim, FrechetUMAPDistance, UMAPMkMMD, UMAPEvalCollection, \
    get_mapper_label, calculate_frechet, ks_test_d
from energydiff.utils.argument_parser import inference_parser
from energydiff.utils.initializer import create_dataset, get_task_profile_condition, get_generated_filename

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
    if COMBINE_VAL_TEST:
        target = torch.cat([all_profile['val'], all_profile['test']], dim=0) # shape: [sample, channel, sequence]
    else:
        target = all_profile['test']
    
    if config.data.vectorize:
        target = data.inverse_vectorize_fn(target, style=config.data.style_vectorize)
    
    generated = {}
    dataset_name = {
        'wpuq': "wpuq",
        "lcl_electricity": "lcl_electricity",
    }
    if not SKIP_BASELINE:
        try:
            generated['gmm'] = torch.load(f'generated_data/{get_generated_filename(config, "gmm", 10)}',
                                        map_location='cuda' if ENABLE_CUDA else 'cpu').float() # shape: [sample, 1, sequence]
        except:
            print('gmm not found')
        try:
            generated['copula'] = torch.load(f'generated_data/{get_generated_filename(config, "copula")}',
                                            map_location='cuda' if ENABLE_CUDA else 'cpu').float() # shape: [sample, 1, sequence]
        except:
            print('copula not found')
    generated['ddpm'] = torch.load(f'generated_data/{get_generated_filename(config, "ddpm")}', 
                                   map_location='cuda' if ENABLE_CUDA else 'cpu').float() # shape: [sample, 1 or 3, sequence]
    # generated['trn_sample'] = all_profile['train'][:2000]
    if os.path.exists(f'generated_data/{get_generated_filename(config, "ddpm-calibrated")}'):
        try:
            generated['ddpm-calibrated'] = torch.load(f'generated_data/{get_generated_filename(config, "ddpm-calibrated")}',
                                            map_location='cuda' if ENABLE_CUDA else 'cpu').float()
        except Exception as error:
            print(f'load calibrated failed. \n{error}')
        else:
            print(f'load calibrated successful.')
    else:
        print(f'calibrated data does not exist.')
    
    if BALANCE_NUM_SAMPLE:
        print('balancing number of samples')
        min_num_sample = min(target.shape[0], *[value.shape[0] for value in generated.values()])
        sampler = np.random.default_rng()
        target = target[sampler.permutation(target.shape[0])[:min_num_sample]]
        for key, value in generated.items():
            generated[key] = value[sampler.permutation(value.shape[0])[:min_num_sample]]
        
    
    if ENABLE_CUDA:
        try:
            print('trying cuda')
            target = target.to('cuda')
            for key, value in generated.items():
                generated[key] = value.to('cuda')
            device = torch.device('cuda')
        except RuntimeError:
            print('cuda is not available')
    else:
        for key, value in generated.items():
            generated[key] = value.cpu()
        device = torch.device('cpu')
    
    if FAST_RUN:
        rng = np.random.default_rng()
        indices = rng.permutation(target.shape[0])[:min(2000, target.shape[0])]
        target = target[indices]

    for key, value in generated.items():
        if generated[key].shape[1] > 1: # channel > 1
            # NOTE should either just take the middle dimension or flatten
            # mid_dim = (generated[key].shape[1] -1) // 2
            # generated[key] = generated[key][:, mid_dim, :]
            generated[key] = rearrange(generated[key], 'b c l -> b (l c)')
    
    for key, source in generated.items():
        print('==============================================================')
        print(f'model: {key}')
        print(f'--------------------------------------------------------------')
    
        target, source = map(
            lambda x: rearrange(x, 'b 1 s -> b s') if x.ndim == 3 else x,
            (target, source)
        )
        
        print(f'number of target samples: {target.shape[0]}')
        print(f'number of source samples: {source.shape[0]}')
        print(f'--------------------------------------------------------------')
        
        # --- all eval fn ---
        eval_fn = {}
        
        # --- mkmmd ---
        
        mkmmd = MkMMD(
            kernel_type='rbf',
            num_kernel=1,
            kernel_mul=2.0,
            coefficient='auto',
            # fix_sigma=60.,
        ).to(device)
        # mkmmd = torch.compile(mkmmd)
        eval_fn['mkmmd'] = mkmmd
        # _sum = 0.
        # for i in range(50):
        #     _src = source[np.random.choice(source.shape[0], 150)]
        #     _tgt = target[np.random.choice(target.shape[0], 150)]
        #     _res = mkmmd(_src, _tgt)
        #     _sum += _res.clamp(0.)
        #     print(f'iter {i}: {_res:.4f}')
        # print(f'mkmmd: {(_sum / 50):.4f}')
        # print(f'mkmmd: {mkmmd(source, target):.4f} ')
        # print(f'central bandwidth: {mkmmd.bandwidth:.4f}')
        # print(f'{source.mean():.4f} {source.std():.4f}')
        # print(f'{target.mean():.4f} {target.std():.4f}')
        # exit()
        
        # print(f'central bandwidth: {mkmmd.bandwidth:.4f}')
        # print(f'mkmmd: {mkmmd_result:.4f}')
        
        # --- direct frechet distance ---
        eval_fn['d-fd'] = calculate_frechet
        
        # --- kl divergence, ws dist, ks test ---
        eval_fn['kl_div'] = kl_divergence
        eval_fn['ws_dist'] = ws_distance
        eval_fn['ks_test_d'] = ks_test_d
        
        # --- umap based metrics ---
        umap_eval = UMAPEvalCollection(full_dataset_name=get_mapper_label(config.data))
        umap_eval_seq = umap_eval.generate_eval_sequence()
        for fn_name, fn in umap_eval_seq:
            eval_fn[fn_name] = fn
        
        # --- run ---
        eval_res = {}
        for fn_name, func in eval_fn.items():
            eval_res[fn_name] = func(source, target)
        
        # --- print ---
        print(f'| mkmmd\t\t| d-fd\t\t| kl div\t| ws dist\t| ks test\t|')
        print(f'| {eval_res["mkmmd"]:.4f}\t'
              f'| {eval_res["d-fd"]:.4f}\t'
              f'| {eval_res["kl_div"]:.4f}\t'
              f'| {eval_res["ws_dist"]:.4f}\t'
              f'| {eval_res["ks_test_d"]:.4f}\t'
              )
        if 'umap_mkmmd' in eval_res:
            print(f'--------------------------------------------------------------')
            print(f'| u-mkmmd\t| fud\t\t| fud extreme\t| r-kl div\t| r-ws dist\t| r-ks test\t|')
            print(f'| {eval_res["umap_mkmmd"]:.4f}\t'
                f'| {eval_res["fud"]:.4f}\t'
                f'| {eval_res["fud_extreme"]:.4f}\t'
                f'| {eval_res["reduced_kl_divergence"]:.4f}\t'
                f'| {eval_res["reduced_ws_distance"]:.4f}\t'
                f'| {eval_res["reduced_ks_test_d"]:.4f}\t|')
        print(f'==============================================================')
    
if __name__ == '__main__':
    main()