from datetime import datetime
import random
import os
from multiprocessing import cpu_count
from copy import deepcopy
from functools import partial

import torch
from torch import nn
import wandb
from energydiff.dataset import NAME_SEASONS, PIT, standard_normal_icdf, standard_normal_cdf
from energydiff.diffusion.dataset import ConditionalDataset1D
from energydiff.diffusion import Trainer1D, IntegerEmbedder, EmbedderWrapper, Zeros
from energydiff.diffusion.models_1d import KernelVelocity
from energydiff.utils.initializer import create_backbone, create_dataset, create_diffusion, \
    create_cond_embedder_wrapped, get_task_profile_condition, \
        create_rectified_flow
from energydiff.utils.eval import MkMMD, source_mean, source_std, target_mean, target_std, only_central_dim, kl_divergence, ws_distance, \
    ks_test_d, ks_test_p, UMAPEvalCollection, get_mapper_label, calculate_frechet

from energydiff.utils.argument_parser import argument_parser

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    config = argument_parser()
    exp_id = config.exp_id
    time_id = datetime.now().strftime("%Y%m%d" + "-" + f"{random.randint(0, 9999):04d}")
    # Step -1: Parse arguments
    data_root = config.data.root
    train_season = config.data.train_season
    
    conditioning = config.model.conditioning
    
    num_sampling_step = config.sample.num_sampling_step
    diffusion_objective = config.diffusion.prediction_type
    
    resume = config.model.resume
    freeze_layers = config.model.freeze_layers
    
    if conditioning:
        run_id = f"train-diffusion-{diffusion_objective}-cond" \
            + '-' + time_id
    else:
        run_id = f"train-diffusion-{diffusion_objective}-{train_season}" \
            + '-' + time_id
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Step 0: Load data
    dataset_collection = create_dataset(config.data)
    # log transforms
    pre_transforms = {}
    post_transforms = {}
    pit: PIT = dataset_collection.dataset_collection.pit
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
    )
    r"""which way of formulating condition is better? maybe I code all conditions as numbers and put them\
        in a tensor? and I separately store the mapping from semantically meaningful condition\
            values to the numbers? """
    # raise NotImplementedError("TODO: cossmic condition need to be re-formulated. Or reformulate \
    #     the condition of the other datasets. ")
   
    #    trainset
    trainset = ConditionalDataset1D(
        tensor=all_profile['train'],
        condition=all_condition['train'],
        transforms=pre_transforms.values(),
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
    
    # get data dimensions
    num_channel = trainset.num_channel
    seq_length = trainset.sequence_length

    #   step 1.2: backbone
    backbone_model = KernelVelocity(m=600, h=15)
    backbone_model.num_in_channel = num_channel
    backbone_model.conditioning = False
    
    if resume and freeze_layers:
        backbone_model.freeze_layers()
    
    #   step 1.3: diffusion
    rf = create_rectified_flow(
        base_model=backbone_model,
        seq_length=seq_length,
        rf_config=config.diffusion,
        num_discretization_step=num_sampling_step,
    ).to(device)
    
    backbone_model.train(all_profile['train'])
    backbone_model.to(device)
    
    samples = []
    samples.append(rf.sample(500))
    samples.append(rf.sample(500))
    samples.append(rf.sample(500))
    samples.append(rf.sample(500))
    samples = torch.cat(samples, dim=0)
    torch.save(samples, f'generated_data/ddpm_20240319-6666_cossmic_grid-import_residential_samples_1min_winter_{num_sampling_step}_1000.pt')
    pass

if __name__ == '__main__':
    main()