import os
from multiprocessing import cpu_count
from copy import deepcopy
import logging

import torch
import pytorch_lightning as pl
from energydiff.dataset import NAME_SEASONS, PIT, standard_normal_icdf, standard_normal_cdf
from energydiff.diffusion.dataset import ConditionalDataset1D
from energydiff.diffusion import PLDiffusion1D
from energydiff.utils.initializer import create_dataset, \
    create_cond_embedder_wrapped, get_task_profile_condition, \
        create_rectified_flow, create_pl_trainer
from energydiff.utils.eval import MultiMetric, MkMMD, source_mean, source_std, target_mean, target_std, kl_divergence, ws_distance, \
    ks_test_d, UMAPEvalCollection, get_mapper_label, calculate_frechet

from energydiff.utils.argument_parser import argument_parser,save_config
from energydiff.utils import generate_time_id

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

torch.set_float32_matmul_precision('high')

def main():
    config = argument_parser()
    config.train.num_train_step = 1
    exp_id = config.exp_id
    if config.time_id is None:
        time_id = generate_time_id()
    else:
        time_id = config.time_id
    # Step -1: Parse arguments
    train_season = config.data.train_season
    val_season = config.data.val_season
    
    conditioning = config.model.conditioning
    diffusion_objective = config.diffusion.prediction_type
    log_wandb = config.log_wandb
    val_batch_size = config.train.val_batch_size
    config.time_id = time_id
    
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
        tensor=all_profile['train'][:1024],
        condition=None,
        transforms=list(pre_transforms.values()), # converting to list is necessary for pickling
    )
    print('train: ',trainset)
    valset = ConditionalDataset1D(
        tensor=all_profile['val'][:1024],
        condition=None,
    )
    print('val: ', valset)
    
    # Step 1: Define model
    #   step 1.1: embedders
    if conditioning:
        raise NotImplementedError
    
    # get data dimensions
    num_channel = trainset.num_channel
    seq_length = trainset.sequence_length
    config.model.num_in_channel = num_channel
    config.model.seq_length = seq_length
    
    # Step 2: Define trainer
    #   validation
    dict_eval_fn = {
        'source_mean': source_mean,
    }
    make_metrics = lambda: MultiMetric(dict_eval_fn)

    # Step 3: Train
    pl_diffusion = PLDiffusion1D(
        trainset=trainset,
        valset=valset,
        model_config=config.model,
        diffusion_config=config.diffusion,
        train_config=config.train,
        metrics_factory=make_metrics,
    )
    trainer = pl.Trainer(
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        max_steps=config.train.num_train_step,
        val_check_interval=10000,
        check_val_every_n_epoch=None,
        accelerator='cpu' if not torch.cuda.is_available() else 'auto',
        precision='bf16-mixed',
        profiler='simple',
        callbacks=[pl.callbacks.DeviceStatsMonitor()],
    )
    
    if trainer.is_global_zero:
        # init wandb to get the id
        trainer.logger.experiment
        # update config
        config.model.num_parameter = count_parameters(pl_diffusion.diffusion_model)
        try:
            config.data.scaling_factor = list(map(lambda x: x.item(), dataset_collection.dataset.scaling_factor))
            print(f'scaling factor: {config.data.scaling_factor}')
        except AttributeError:
            pass
    
        os.makedirs('results', exist_ok=True)
        logging.basicConfig(filename=f'results/{run_id}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info(f"experiment starts: {run_id}")
    else:
        logging.basicConfig(level=logging.CRITICAL+1)
    if config.model.resume:
        if config.model.load_runid is None or config.model.load_milestone is None:
            print("Warning: resume is True but runid or milestone is not specified. Will proceed without resuming.")
        else:
            load_milestone = config.model.load_milestone
        load_time_id = config.model.load_runid
        if conditioning:
            load_run_id = f"train-diffusion-{diffusion_objective}-cond-{load_time_id}"
        else:
            load_run_id = f"train-diffusion-{diffusion_objective}-{train_season}-{load_time_id}"
        
        checkpoint_path = os.path.join(trainer.log_dir, f"{load_run_id}/milestone_{load_milestone}.ckpt")
        pl_diffusion.load_from_checkpoint(checkpoint_path)
        print(f'Successfully loaded model from milestone {load_milestone} of time_id {load_time_id}.')
    
    # Step 3: Train
    logging.info('training initiated.')
    trainer.fit(pl_diffusion)
    logging.info('training complete.')
    
    return time_id

if __name__ == '__main__':
    main()