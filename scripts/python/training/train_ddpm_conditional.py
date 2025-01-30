from datetime import datetime
import random
import os
from multiprocessing import cpu_count
from copy import deepcopy
from functools import partial
import logging

import torch
from torch import nn
import wandb
from energydiff.dataset import NAME_SEASONS, PIT, standard_normal_icdf, standard_normal_cdf
from energydiff.diffusion.dataset import ConditionalDataset1D
from energydiff.diffusion.dpm_solver import DPMSolverSampler
from energydiff.diffusion import Trainer1D, IntegerEmbedder, EmbedderWrapper, Zeros
from energydiff.utils.initializer import create_backbone, create_dataset, create_diffusion, \
    create_cond_embedder_wrapped, get_task_profile_condition, \
        create_rectified_flow
from energydiff.utils.eval import MkMMD, source_mean, source_std, target_mean, target_std, only_central_dim, kl_divergence, ws_distance, \
    ks_test_d, ks_test_p, UMAPEvalCollection, get_mapper_label, calculate_frechet

from energydiff.utils.argument_parser import argument_parser,save_config
from energydiff.utils import generate_time_id

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    config = argument_parser()
    exp_id = config.exp_id
    time_id = generate_time_id()
    # Step -1: Parse arguments
    train_season = config.data.train_season
    val_season = config.data.val_season
    
    conditioning = config.model.conditioning
    diffusion_objective = config.diffusion.prediction_type
    log_wandb = config.log_wandb
    val_batch_size = config.train.val_batch_size
    
    if conditioning:
        run_id = f"train-diffusion-{config.diffusion.prediction_type}-cond" \
            + '-' + time_id
    else:
        run_id = f"train-diffusion-{config.diffusion.prediction_type}-{train_season}" \
            + '-' + time_id
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Step 0: Load data
    dataset_collection = create_dataset(config.data)
    # log transforms
    pre_transforms = {}
    post_transforms = {}
    pit: PIT = dataset_collection.dataset.pit
    if pit is not None:
        pre_transforms['pit'] = pit.transform
        pre_transforms['erf'] = standard_normal_icdf
        post_transforms['erf'] = standard_normal_cdf
        post_transforms['pit'] = pit.inverse_transform
        _data_config = deepcopy(config.data)
        _data_config.pit = False
        dataset_collection = create_dataset(_data_config) # re-create dataset without pit. we do pit inside the diffusion

    #   get profile and condition per task (train, val, test)
    all_profile, all_condition = get_task_profile_condition(
        dataset_collection, 
        season=train_season, 
        conditioning=conditioning,
        lcl_use_fraction=0.01, # for LCL dataset
    )
   
    #    trainset
    trainset = ConditionalDataset1D(
        tensor=all_profile['train'],
        condition=all_condition['train'],
        transforms=list(pre_transforms.values()), # converting to list is necessary for pickling
    )
    print(trainset)
    valset = ConditionalDataset1D(
        tensor=all_profile['val'],
        condition=all_condition['val'],
    )
    
    # Step 1: Define model
    #   step 1.1: embedders
    cond_embedder = None
    sample_model_kwargs = {}
    num_condition_channel = None
    if conditioning:
        cond_embedder = create_cond_embedder_wrapped(
            dataset_collection=dataset_collection,
            dim_embedding=config.model.dim_base,
        )
        num_condition_channel = trainset.num_condition_channel
        _sample_cond = [float(NAME_SEASONS.index(val_season))]
        if config.data.dataset == 'cossmic' and 'area' in config.data.target_labels:
            _sample_cond.append(float(dataset_collection.condition_mapping['area'][config.data.val_area]))
        sample_model_kwargs = {
            'c': torch.tensor(_sample_cond, dtype=torch.float32).reshape(1, len(trainset.list_dim_cond), 1).to(device), # season
            'cfg_scale': config.train.val_sample_config.cfg_scale,
        }
        _profile, _ = get_task_profile_condition(dataset_collection, 
                                                 season=val_season, 
                                                 conditioning=False, 
                                                 area=config.data.val_area,)
        valset = ConditionalDataset1D(
            tensor=_profile['val'],
            condition=None,
        )
    
    # get data dimensions
    num_channel = trainset.num_channel
    seq_length = trainset.sequence_length

    #   step 1.2: backbone
    backbone_model = create_backbone(
        config.model,
        num_in_channel = num_channel,
        cond_embedder = cond_embedder,
        seq_length = seq_length, # only used by MLP
    ).to(device)
    backbone_model.compile()
    
    if config.model.resume and config.model.freeze_layers:
        backbone_model.freeze_layers()
    
    #   step 1.3: diffusion
    if not config.diffusion.use_rectified_flow:
        create_diffusion_base = partial(create_diffusion,
            base_model=backbone_model,
            seq_length=seq_length,
            ddpm_config=config.diffusion,
        ) # except `num_sampling_timestep`
        full_diffusion = create_diffusion_base(
            num_sampling_timestep=config.diffusion.num_diffusion_step
        ).to(device)
        spaced_diffusion_model = create_diffusion_base(
            num_sampling_timestep=config.train.val_sample_config.num_sampling_step,
        ).to(device)
    else:
        full_diffusion = create_rectified_flow(
            base_model=backbone_model,
            seq_length=seq_length,
            rf_config=config.diffusion,
            num_discretization_step=config.train.val_sample_config.num_sampling_step,
        ).to(device)
        spaced_diffusion_model = full_diffusion
            # essentially there's no spaced diffusion. for compatibility. 
    
    # Step 2: Define trainer
    #   validation
    mkmmd = MkMMD(
        kernel_type='rbf',
        num_kernel=1,
        kernel_mul=2.0,
        coefficient='auto'
    )
    umap_eval = UMAPEvalCollection(full_dataset_name=get_mapper_label(config.data))
    def pre_val_fn(x):
        """
        x: (batch_size, num_channel, seq_length)
        out_x: (batch_size, seq_length)
        """
        x = dataset_collection.inverse_vectorize_fn(x) # (batch_size, seq_length)
        x = x[:, 0, :] # (batch_size, seq_length)
        return x
    dict_eval_fn = {
        'MkMMD': mkmmd, # for 3d tensor, in dim=1, use the central channel
        'DirectFD': calculate_frechet,
        'source_mean': source_mean,
        'source_std': source_std,
        'target_mean': target_mean,
        'target_std': target_std,
        'kl_divergence': kl_divergence,
        'ws_distance': ws_distance,
        'ks_test_d': ks_test_d,
    }
    umap_eval_metrics = umap_eval.generate_eval_sequence()
    for metric_name, metric_fn in umap_eval_metrics:
        dict_eval_fn[metric_name] = metric_fn
    trainer = Trainer1D.from_config(
        config.train,
        full_diffusion,
        spaced_diffusion_model,
        dataset=trainset,
        val_dataset=valset,
        num_dataloader_workers=int(os.environ.get('SLURM_JOB_CPUS_PER_NODE', cpu_count())),
        max_val_batch=2,
        sample_model_kwargs=sample_model_kwargs,
        post_transforms=post_transforms.values(),
        pre_eval_fn=pre_val_fn,
        dict_eval_fn=dict_eval_fn,
        log_wandb=config.log_wandb,
        log_id=run_id
    )
    config.model.num_parameter = count_parameters(backbone_model)
    try:
        config.data.scaling_factor = list(map(lambda x: x.item(), dataset_collection.dataset.scaling_factor))
        print(f'scaling factor: {config.data.scaling_factor}')
    except AttributeError:
        pass
    if trainer.accelerator.is_main_process:
        os.makedirs('results', exist_ok=True)
        logging.basicConfig(filename=f'results/{run_id}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info(f"experiment starts: {run_id}")
        if log_wandb:
            wandb.init(
                project={
                    'wpuq': 'HeatDDPM',
                    'wpuq_trafo': "WPuQTrafoDDPm",
                    'wpuq_pv': 'WPuQPVDDPM',
                    'lcl_electricity': 'LCLDDPM',
                    'cossmic': 'CoSSMicDDPM'
                }[config.data.dataset],
                name=run_id,
                config=config.to_dict(),
            )
    else:
        logging.basicConfig(level=logging.CRITICAL+1)
    if config.model.resume:
        if config.model.load_runid is None or config.model.load_milestone is None:
            print("Warning: resume is True but runid or milestone is not specified. Will proceed without resuming.")
        else:
            load_milestone = config.model.load_milestone
            load_time_id = config.model.load_runid
            if conditioning:
                load_run_id = f"train-diffusion-{diffusion_objective}-cond" \
                    + '-' + load_time_id
            else:
                load_run_id = f"train-diffusion-{diffusion_objective}-{train_season}" \
                    + '-' + load_time_id
            trainer.load_model(
                milestone=load_milestone,
                directory=trainer.result_folder,
                log_id=load_run_id,
                ignore_init_final=True,
            )
            print(f'Successfully loaded model from milestone {load_milestone} of time_id {load_time_id}.')
            
    # save config
    save_config(config, time_id)
    logging.info(f'Saved configuration to results/configs/exp_config_{time_id}.yaml')
    
    # Step 3: Train
    logging.info('training initiated.')
    trainer.train()
    logging.info('training complete.')
    trainer.accelerator.wait_for_everyone()
    
    # Step 4: Sample data
    if trainer.accelerator.is_main_process: 
        from energydiff.utils.sample import dpm_solver_sample, ancestral_sample
        from energydiff.utils.initializer import get_generated_filename
        
        # generate samples
        logging.info(f'post-training: generating samples...')
        if config.sample.dpm_solver_sample and not config.diffusion.use_rectified_flow:
            dpm_sampler = DPMSolverSampler(trainer.ema.ema_model)
            generated_samples = dpm_solver_sample(
                dpm_sampler,
                total_num_sample=config.sample.num_sample,
                batch_size=config.sample.val_batch_size,
                step=config.sample.num_sampling_step,
                shape=(num_channel, seq_length),
                conditioning=None,
                cfg_scale=config.sample.cfg_scale,
                accelerator=None, # didn't distribute ema, must be none
            )
        else:
            generated_samples = ancestral_sample(
                config.sample.num_sample,
                config.sample.val_batch_size,
                cond=None,
                cfg_scale=config.sample.cfg_scale,
                model=trainer.ema.ema_model,
            ) # also applies to rectified flow
        target = torch.cat([all_profile['val'], all_profile['test']], dim=0)
        if config.data.vectorize:
            generated_samples = dataset_collection.inverse_vectorize_fn(generated_samples, style=config.data.style_vectorize)
            target = dataset_collection.inverse_vectorize_fn(target, style=config.data.style_vectorize)
        _config = deepcopy(config)
        _config.model.load_runid = time_id
        filename = get_generated_filename(_config, model = 'ddpm')
        os.makedirs('generated_data/', exist_ok=True)
        save_path = os.path.join(
            'generated_data/',
            filename
        )
        torch.save(
            generated_samples, 
            save_path
        )
        try:
            logging.info(f'Scaling factor: {dataset_collection.dataset.scaling_factor}')
        except:
            logging.info(f"Scaling factor: not available.")
        logging.info(f'Saved to {save_path}')
        # evaluate and log summary
        source = generated_samples[:2000].cpu()
        target = target[:].cpu()
        eval_res = {}
        for metric_name, metric_fn in dict_eval_fn.items():
            eval_res[metric_name] = metric_fn(source, target)
            logging.info(f'{metric_name}: {eval_res[metric_name]:.4f}')
        
        if log_wandb:
            wandb.log({'Test': eval_res})
    else:
        exit()

if __name__ == '__main__':
    main()