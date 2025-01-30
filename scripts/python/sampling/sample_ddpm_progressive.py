"""Generate very few samples (~10) and save every (few) reverse step(s). """
import os
from multiprocessing import cpu_count
from copy import deepcopy
from typing import Iterable, Callable

import torch
from energydiff.dataset import NAME_SEASONS, PIT
from energydiff.diffusion.dataset import ConditionalDataset1D
from energydiff.diffusion import Unet1D, Transformer1D, SpacedDiffusion1D, Trainer1D, \
    IntegerEmbedder, EmbedderWrapper, Zeros
from energydiff.utils.initializer import create_backbone, create_dataset, create_diffusion, \
    create_cond_embedder_wrapped, get_task_profile_condition, get_generated_filename, \
        create_rectified_flow
from energydiff.utils.sample import ancestral_sample_progressive, ConditionCrafter
from energydiff.utils.argument_parser import argument_parser, inference_parser

# NUM_SAMPLE = 2000
# SEASON_OR_COND = 'winter' 
# RUNID = '20231214-9806'
# MILESTONE = 8

def main():
    config = inference_parser()
    # Step -1: Parse arguments
    season = config.data.val_season
    if config.model.load_runid is None:
        raise ValueError('load_runid is not specified.')
    
    # conditioning
    conditioning = config.model.conditioning
    cfg_scale = config.sample.cfg_scale
    
    # diffusion
    diffusion_objective = config.diffusion.prediction_type
    
    # sample 
    num_sampling_step = config.sample.num_sampling_step
    val_batch_size = config.sample.val_batch_size
    
    # load
    load_runid = config.model.load_runid
    load_milestone = config.model.load_milestone
    season_or_cond = config.data.train_season if not config.model.conditioning else 'cond'
    
    run_id = f"train-diffusion-{diffusion_objective}-{season_or_cond}" \
        + '-' + load_runid
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Step 0: Load data
    dataset_collection = create_dataset(config.data)
    # log transforms
    pre_transforms = {}
    post_transforms = {}
    pit: PIT = dataset_collection.dataset.pit
    if pit is not None:
        pre_transforms['pit'] = pit.transform
        post_transforms['pit'] = pit.inverse_transform
        foobar = deepcopy(config.data)
        foobar.pit = False
        dataset_collection = create_dataset(foobar) # reload dataset without pit

    #   get profile and condition per task (train, val, test)
    all_profile, all_condition = get_task_profile_condition(dataset_collection, 
                                                            season=season, 
                                                            conditioning=conditioning, 
                                                            area=config.data.val_area)
    
    "the dataset here does not matter at all. they are just placeholder for defining trainer. "
    testset = ConditionalDataset1D(all_profile['test'], all_condition['test'])
    
    # Step 1: Define model
    #   step 1.1 embedders
    cond_embedder = None
    num_condition_channel = None
    if conditioning:
        cond_embedder = create_cond_embedder_wrapped(
            dataset_collection=dataset_collection,
            dim_embedding=config.model.dim_base
        )
        list_num_emb = [embedder.num_embedding for embedder in cond_embedder.list_embedder]
        dict_cond_num_emb = {
            # 'season': season_embedder.num_embedding
            cond_name: cond_num_emb for cond_name, cond_num_emb in zip(dataset_collection.dict_cond_dim.keys(), list_num_emb)
        }
    
    num_channel = testset.num_channel
    seq_length = testset.sequence_length
    #   step 1.2: backbone
    backbone_model = create_backbone(
        config.model,
        num_in_channel = num_channel,
        cond_embedder = cond_embedder,
    ).to(device)
    #   step 1.3: diffusion
    if not config.diffusion.use_rectified_flow:
        diffusion = create_diffusion(
            backbone_model,
            seq_length=seq_length,
            ddpm_config=config.diffusion,
            num_sampling_timestep=num_sampling_step,
        ).to(device)
        diffusion.compile()
    else:
        diffusion = create_rectified_flow(
            base_model=backbone_model,
            seq_length=seq_length,
            rf_config=config.diffusion,
            num_discretization_step=num_sampling_step,
        ).to(device)
    
    # Step 2: Define trainer
    "the trainer is only used to load models. we sample use different functions. "
    trainer = Trainer1D.from_config(
        config.train,
        diffusion_model=diffusion,
        spaced_diffusion_model=diffusion,
        dataset=testset,
        val_dataset=testset,
        log_id=run_id,
        num_dataloader_workers=int(os.environ.get('SLURM_JOB_CPUS_PER_NODE', cpu_count())),
        log_wandb = False,
        distribute_ema=True,
    )
    if trainer.accelerator.is_main_process:
        print(f'sampling {config.sample.num_sample} samples. ')
    # trainer.ema = trainer.accelerator.prepare(trainer.ema)
    trainer.load(milestone=load_milestone)

    # Step 3: Generate samples
    #   step 3.1: select conditions
    cond = None
    str_cond = season
    if conditioning:
        cond_crafter = ConditionCrafter(dict_cond_num_emb)
        dict_cond = {
            'season': torch.tensor(NAME_SEASONS.index(season)), # winter
        }
        if config.data.dataset == 'cossmic' and 'area' in config.data.target_labels:
            dict_cond['area'] = torch.tensor(dataset_collection.condition_mapping['area'][config.data.val_area])
        cond = cond_crafter(batch_size=val_batch_size, dict_cond=dict_cond)
        str_cond = [
            f'{cond_name}-{cond_value.cpu().item()}'
            for cond_name, cond_value in dict_cond.items() if cond_value is not None
        ]
        str_cond = '_'.join(str_cond)
    
    # sample either from EMA or Diffusion
    trainer.model.device = trainer.device
    sample_ddpm, steps = ancestral_sample_progressive(
        batch_size=val_batch_size,
        save_steps=[1000, 500, 100, 80, 60, 40, 20, 10, 5, 1],
        cond=cond,
        cfg_scale=cfg_scale if conditioning else 1., 
        trainer=trainer
    )
    
    if config.data.vectorize:
        b, t, c, l = sample_ddpm.shape
        sample_ddpm = dataset_collection.inverse_vectorize_fn(sample_ddpm.view(b*t, c, l), style=config.data.style_vectorize)
        n, c, l = sample_ddpm.shape
        sample_ddpm = sample_ddpm.view(b, t, c, l)
    
    if trainer.accelerator.is_main_process:
        filename = get_generated_filename(config, model = 'ddpm')
        save_path = os.path.join(
            'ddpm_progressive/',
            filename
        )
        torch.save(
            {'samples': sample_ddpm, 'steps': steps}, 
            save_path
        )
        print(f'saved to {save_path}')
    
    pass

if __name__ == '__main__':
    main()