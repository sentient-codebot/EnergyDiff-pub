import os
import warnings
from functools import partial
from typing import List
from datetime import datetime
import random
import string

import torch
import numpy as np
import pytorch_lightning as pl
import wandb

from energydiff.diffusion import LossType, BetaScheduleType, ModelMeanType, ModelVarianceType
from energydiff.diffusion import Transformer1D, Unet1D, SpacedDiffusion1D, IntegerEmbedder, EmbedderWrapper, Zeros, DenoisingMLP1D
from energydiff.diffusion import space_timesteps
from energydiff.diffusion.rectified_flow import RectifiedFlow, RFPredictionType, RFScheduleType
from energydiff.dataset import NAME_SEASONS, TimeSeriesDataset, WPuQ, WPuQTrafo, WPuQPV, LCLElectricityProfile, CoSSMic, all_dataset, WPUQ_PV_DIRECTION_CODE

from .configuration import ExperimentConfig, DataConfig, CossmicDataConfig, ModelConfig, DDPMConfig, RectifiedFlowConfig, SampleConfig

def generate_random_id():
    # Get current date-time formatted string
    date_str = datetime.now().strftime("%Y%m%d%H%M")
    
    # Generate 12 random characters (letters + digits)
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
    
    # Combine with hyphen separator
    return f"{date_str}-{random_str}"

class WandbArtifactCleanerAlternative(pl.Callback):
    """this deletes all artifact versions except the latest and best"""
    def __init__(self, ):
        super().__init__()
        
    def on_validation_end(self, trainer, pl_module):
        if not wandb.run:
            return
        
        api = wandb.Api()
        artifact_type = "model"
        artifact_name = f"{wandb.run.entity}/{wandb.run.project}/model-{wandb.run.id}"
        
        try:
            # Get collection of artifacts
            all_versions = list(api.artifacts(artifact_type, artifact_name))
            _latest = list(api.artifacts(artifact_type, artifact_name, tags='latest'))
            _best = list(api.artifacts(artifact_type, artifact_name, tags='best'))
            
            # Delete all versions except the latest and best
            versions_to_keep = set(_latest + _best)
            for version in all_versions:
                if version not in versions_to_keep:
                    version.delete()
                    
        except Exception as e:
            print(f"Error cleaning artifacts: {e}")

class WandbArtifactCleaner(pl.Callback):
    def __init__(self, keep_n_latest=2):
        super().__init__()
        self.keep_n_latest = keep_n_latest
        
    def on_validation_end(self, trainer, pl_module):
        if not wandb.run:
            return
            
        api = wandb.Api()
        artifact_type = "model"
        artifact_name = f"{wandb.run.entity}/{wandb.run.project}/model-{wandb.run.id}"
        
        try:
            # Get all versions
            artifact_versions = api.artifact_versions(artifact_type, artifact_name)
            versions = sorted(artifact_versions, key=lambda x: x.created_at, reverse=True)
            
            # Find version marked as "best"
            best_version = next(
                (v for v in versions if "best" in v.aliases),
                None
            )
            
            # Keep N latest versions and best version
            versions_to_keep = set(versions[:self.keep_n_latest])
            if best_version:
                versions_to_keep.add(best_version)
            
            # Delete versions not in keep set
            for version in versions:
                if version not in versions_to_keep:
                    version.delete()
                    
        except Exception as e:
            print(f"Error cleaning artifacts: {e}")

class WandbModelLogger(pl.Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if not wandb.run:
            return
            
        # Create a temporary file to save the checkpoint
        checkpoint_path = os.path.join(trainer.default_root_dir, 'temp_checkpoint.ckpt')
        torch.save(checkpoint, checkpoint_path)
        
        try:
            current_score = trainer.checkpoint_callback.current_score
            
            artifact = wandb.Artifact(
                name=f"{wandb.run.id}_model",
                type="model",
                metadata={
                    "score": current_score
                }
            )
            
            artifact.add_file(checkpoint_path)
            
            aliases = ['latest']
            if trainer.checkpoint_callback.best_model_path == trainer.checkpoint_callback.last_model_path:
                aliases.append('best')
                
            wandb.log_artifact(artifact, aliases=aliases)
            
        finally:
            # Clean up temporary file
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

def create_wandb_logger(config: ExperimentConfig, run_id: str) -> pl.loggers.WandbLogger:
    r"""
    Setup and init the wandb logger.
    """
    wandb_logger = pl.loggers.WandbLogger(
        project={
            'wpuq': 'HeatDDPM',
            'wpuq_trafo': "WPuQTrafoDDPM",
            'wpuq_pv': 'WPuQPVDDPM',
            'lcl_electricity': 'LCLDDPM',
            'cossmic': 'CoSSMicDDPM'
        }[config.data.dataset],
        name=run_id,
        config=config.to_dict(),
        log_model='all', # to be combined with WandbArtifactCleaner
    )
    return wandb_logger

def create_pl_trainer(config: ExperimentConfig, run_id) -> pl.Trainer:
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="Validation/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    ckpt_artifact_cleaner = WandbArtifactCleaner(keep_n_latest=1)
    return pl.Trainer(
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        max_steps=config.train.num_train_step,
        logger=create_wandb_logger(config, run_id),
        val_check_interval=config.train.val_every,
        check_val_every_n_epoch=None,
        accelerator='cpu' if not torch.cuda.is_available() else 'auto',
        strategy='ddp',
        precision='bf16-mixed',
        profiler='simple',
        callbacks=[checkpoint_callback, ckpt_artifact_cleaner],
    )


def create_diffusion(
    base_model: torch.nn.Module,
    seq_length: int,
    ddpm_config: DDPMConfig,
    num_sampling_timestep: int|None = None,
    *,
    mse_loss: bool = True, # ALWAYS
    rescale_learned_variance: bool = True, # ALWAYS
    ddim_sampling_eta: float = 0., # ALWAYS
    auto_normalize: bool = False, # ALWAYS
    loss_only_central_channel: bool = False, # ALWAYS
    fft_loss_weight: float = 0., # ALWAYS
) -> SpacedDiffusion1D:
    """
    Create a Gaussian diffusion model from a base model.
    
    :param base_model: the base model
    :param seq_length: the length of the sequence
    :param num_timestep: the number of diffusion steps
    :param num_sampling_step: the number of sampling steps
    :param model_mean_type: the type of mean model
    :param model_var_type: the type of variance model
    :param loss_type: the type of loss
    :param beta_schedule_type: the type of beta schedule
    :param ddim_sampling_eta: the eta parameter of the ddim sampling
    :param auto_normalize: whether to automatically normalize the loss
    :param loss_only_central_channel: whether to compute the loss only on the central channel
    """
    if ddpm_config.beta_schedule_type == 'cosine':
        beta_schedule = BetaScheduleType.COSINE
    elif ddpm_config.beta_schedule_type == 'linear':
        beta_schedule = BetaScheduleType.LINEAR
    else:
        warnings.warn(f"Unknown beta schedule type {ddpm_config.beta_schedule_type}. Using cosine schedule.")
        beta_schedule = BetaScheduleType.COSINE
        
    if ddpm_config.prediction_type == 'pred_v':
        prediction_type = ModelMeanType.V
    elif ddpm_config.prediction_type == 'pred_x0':
        prediction_type = ModelMeanType.XSTART
    elif ddpm_config.prediction_type == 'pred_noise':
        prediction_type = ModelMeanType.NOISE
    else:
        raise ValueError(f"Unknown model mean type {prediction_type}.")
    
    if not mse_loss:
        loss_type = LossType.RESCALED_KL
    elif rescale_learned_variance:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE
        
    if ddpm_config.learn_variance:
        model_var_type = ModelVarianceType.LEARNED_RANGE
    elif ddpm_config.sigma_small:
        model_var_type = ModelVarianceType.FIXED_SMALL
    else:
        model_var_type = ModelVarianceType.FIXED_LARGE
        
    num_sampling_timestep = num_sampling_timestep or ddpm_config.num_diffusion_step
    return SpacedDiffusion1D(
        use_timesteps=space_timesteps(ddpm_config.num_diffusion_step, [num_sampling_timestep]),
        base_model=base_model,
        seq_length=seq_length,
        num_timestep=ddpm_config.num_diffusion_step,
        num_sampling_timestep=None,
        model_mean_type=prediction_type,
        model_variance_type=model_var_type,
        loss_type=loss_type,
        beta_schedule_type=beta_schedule,
        ddim_sampling_eta=ddim_sampling_eta,
        auto_normalize=auto_normalize,
        loss_only_central_channel=loss_only_central_channel,
        fft_loss_weight=fft_loss_weight,
    )


def create_cond_embedder(
    dict_cond_dim: dict[str, int],
    dim_embedding: int,
    dict_cond_kwargs: dict[str, dict] = {},
    ignore_conditions: list[str] = [],
) -> EmbedderWrapper:
    """
    Create a condition embedder for different conditions. 
    
    Required:
        - dict_cond_dim: a dictionary of condition name and its dimension.
        - dim_embedding: the dimension of the embedding. 
            - int, all conditions shall have the same embedding dimension. 
    Optional:
        - dict_kwargs: a dictionary of condition name and its kwargs for the embedder. 
            - e.g. 'annual_consumption': {'quantize': True, 'quantize_max_val': 20000., 'quantize_min_val': 0.}
        - ignore_conditions: a list of condition names to ignore. 
    """
    # Prepare condition embedder factory
    default_kwargs = {
        'season': {'num_embedding': 4, 'dropout': 0.1},
        'month': {'num_embedding': 12, 'dropout': 0.1},
        'year': {'num_embedding': 10, 'dropout': 0.1}, # year span <= 10
        'annual_consumption': {'num_embedding': 400, 'dropout': 0.1, \
            'quantize': True, 'quantize_max_val': 20000., 'quantize_min_val': 0.}, # annual consumption span <= 20000
        'area': {'num_embedding': 3, 'dropout': 0.1}, # area = 0, 1, 2 (0: industrial, 1: residential, 2: public)
    }
    for cond_name in dict_cond_kwargs.keys():
        default_kwargs[cond_name].update(dict_cond_kwargs[cond_name])
    _create_emb_collection = {
        cond_name: partial(IntegerEmbedder, **kwargs, dim_embedding=dim_embedding) \
            if cond_name not in ignore_conditions else partial(Zeros, **kwargs, dim_embedding=dim_embedding) \
                for cond_name, kwargs in default_kwargs.items()
    }
    # Create needed condition embedders
    list_embedder = [
        _create_emb_collection[cond_name]() \
            for cond_name in dict_cond_dim.keys()
    ]
    cond_embedder = EmbedderWrapper(
        list_embedder=list_embedder,
        list_dim_cond=list(dict_cond_dim.values()),
    )
    
    return cond_embedder

def create_cond_embedder_wrapped(dataset_collection: TimeSeriesDataset, dim_embedding: int, **kwargs) -> EmbedderWrapper:
    """Create Condition Embedder
    Arguments:
        - dataset: a TimeSeriesDataset object
        - dim_embedding: the dimension of the encoded embedding. 
        - kwargs: other arguments for create_cond_embedder
            - dict_cond_kwargs: a dictionary of condition name and its kwargs for the embedder.
            - ignore_conditions: a list of condition names to ignore.
    """
    if isinstance(dataset_collection, WPuQ):
        return create_cond_embedder(
            dict_cond_dim=dataset_collection.dict_cond_dim,
            dim_embedding=dim_embedding,
            dict_cond_kwargs={
                'annual_consumption': {'quantize': True, 'quantize_max_val': 20000., 'quantize_min_val': 0.},
                'year': {'num_embedding': 3} # year span = 3 (2018, 2019, 2020)
            },
            ignore_conditions=kwargs.get('ignore_conditions', [])
        )
    elif isinstance(dataset_collection, CoSSMic):
        return create_cond_embedder(
            dict_cond_dim=dataset_collection.dict_cond_dim,
            dim_embedding=dim_embedding,
            dict_cond_kwargs={
                'year': {'num_embedding': 5}, # year span = 5 (2015, 2016, 2017, 2018, 2019)
                'area': {'num_embedding': 3, 'dropout': 0.1}, # area = 0, 1, 2 (0: industrial, 1: residential, 2: public)
            },
            ignore_conditions=kwargs.get('ignore_conditions', [])
        )
    else:
        raise NotImplementedError(f"Conditional dataset {dataset_collection} not implemented.")


def create_dataset(data_config: DataConfig) -> TimeSeriesDataset:
    # Step 1: Setup dataset
    dataset_name = data_config.dataset
    assert dataset_name in all_dataset, f'Unknown dataset {dataset_name}'

    if dataset_name in ['heat_pump', 'wpuq']:
        dataset = WPuQ(
            root=data_config.root,
            resolution=data_config.resolution,
            load=data_config.load,
            normalize=data_config.normalize,
            pit_transform=data_config.pit,
            shuffle=data_config.shuffle,
            vectorize=data_config.vectorize,
            style_vectorize=data_config.style_vectorize,
            vectorize_window_size=data_config.vectorize_window_size,
        )
    elif dataset_name == 'wpuq_trafo':
        dataset = WPuQTrafo(
            root=data_config.root,
            resolution=data_config.resolution,
            load=data_config.load,
            normalize=data_config.normalize,
            pit_transform=data_config.pit,
            shuffle=data_config.shuffle,
            vectorize=data_config.vectorize,
            style_vectorize=data_config.style_vectorize,
            vectorize_window_size=data_config.vectorize_window_size,
        )
    elif dataset_name == 'wpuq_pv':
        dataset = WPuQPV(
            root=data_config.root,
            resolution=data_config.resolution,
            load=data_config.load,
            normalize=data_config.normalize,
            pit_transform=data_config.pit,
            shuffle=data_config.shuffle,
            vectorize=data_config.vectorize,
            style_vectorize=data_config.style_vectorize,
            vectorize_window_size=data_config.vectorize_window_size,
        )
    elif dataset_name == 'lcl_electricity':
        dataset = LCLElectricityProfile(
            root=data_config.root,
            load=data_config.load,
            resolution=data_config.resolution,
            normalize=data_config.normalize,
            pit_transform=data_config.pit,
            shuffle=data_config.shuffle,
            vectorize=data_config.vectorize,
            style_vectorize=data_config.style_vectorize,
            vectorize_window_size=data_config.vectorize_window_size,
        )
    elif dataset_name == 'cossmic':
        assert isinstance(data_config, CossmicDataConfig), 'CossmicDataConfig is required for CoSSMic dataset'
        dataset = CoSSMic(
            root=data_config.root,
            target_labels=data_config.target_labels,
            resolution=data_config.resolution,
            load=data_config.load,
            normalize=data_config.normalize,
            pit_transform=data_config.pit,
            shuffle=data_config.shuffle,
            vectorize=data_config.vectorize,
            style_vectorize=data_config.style_vectorize,
            vectorize_window_size=data_config.vectorize_window_size,
            sub_dataset_names=data_config.subdataset_names,
        )
    else:
        raise NotImplementedError(f'Dataset {dataset_name} not implemented')

    return dataset


def get_task_profile_condition(dataset: TimeSeriesDataset, **kwargs):
    """ 
    Required arguments:
        dataset: dataset object
        dataset-specific arguments:
            - HeatPumpProfile: [season]
            - LCLElectricityProfile: []
            - CoSSMic: [season, dataset_names] 
    Optional arguments:
        season: str, one of ['whole_year', 'winter', 'spring', 'summer', 'autumn']
        area: str, one of ['all', 'industrial', 'residential', 'public']
        dataset_names: list of str, e.g. ['pv_industrial', 'grid_import_residential']
    Return:
        final_profile: dict of torch.Tensor
        final_condition: dict of torch.Tensor
    """
    final_profile = {
        'train': None,
        'val': None,
        'test': None,
    }
    final_condition = {
        'train': None,
        'val': None,
        'test': None,
    }
    if type(dataset) is WPuQ or type(dataset) is WPuQTrafo:
        season = kwargs.get('season', None)
        if season is None:
            print('Warning: season is None, using whole_year')
            season = 'whole_year'
        # if args.conditioning:
            # assert season == 'whole_year', 'Conditioning only implemented for whole_year'
        for task in final_profile.keys():
            if season in NAME_SEASONS:
                    final_profile[task] = dataset.dataset.profile[task][season]
                    final_condition[task] = {'season': dataset.dataset.label[task][season]}
            elif season == 'whole_year':
                final_profile[task] = torch.cat(list(
                    dataset.dataset.profile[task].values()
                ), dim=0)
                final_condition[task] = {
                    'season': torch.cat(list(
                        dataset.dataset.label[task].values()
                        ), dim=0)
                }
    elif type(dataset) is WPuQPV:
        season = kwargs.get('season', None)
        direction = kwargs.get('direction', None)
        if season is None:
            print('Warning: season is None, using whole_year')
            season = 'whole_year'
        if direction is None:
            print('Warning: direction is None, using all_directions')
            direction = 'all_directions'
        # if args.conditioning:
            # assert season == 'whole_year', 'Conditioning only implemented for whole_year'
        for task in final_profile.keys():
            if season in NAME_SEASONS:
                final_profile[task] = dataset.dataset.profile[task][season]
                final_condition[task] = {'season': dataset.dataset.label[task][season]}
            elif season == 'whole_year':
                final_profile[task] = torch.cat(list(
                    dataset.dataset.profile[task].values()
                ), dim=0)
                final_condition[task] = {
                    'season': torch.cat(list(
                        dataset.dataset.label[task].values()
                        ), dim=0)
                }
        for task in final_profile.keys():
            if direction in WPUQ_PV_DIRECTION_CODE.keys():
                _dir = WPUQ_PV_DIRECTION_CODE[direction]
                _indices = final_condition[task][:, 1, 0] == _dir
                final_profile[task] = final_profile[task][_indices]
                final_condition[task] = final_condition[task][_indices]
            elif direction == 'all_directions':
                pass
            else:
                raise RuntimeError(f'Unknown direction {direction}. Must be one of {WPUQ_PV_DIRECTION_CODE.keys()} or "all_directions"')
    elif isinstance(dataset, LCLElectricityProfile):
        use_fraction = kwargs.get('lcl_use_fraction', 0.01)
        for task in final_profile.keys():
            indices = np.random.choice(
                len(dataset.dataset.profile[task]), 
                int(len(dataset.dataset.profile[task]) * use_fraction), 
                replace=False
            )
            final_profile[task] = dataset.dataset.profile[task][indices]
            # label shape: (N, d, 1) where d is the dimension of the label
            final_condition[task] = {
                'year': dataset.dataset.label[task][indices, 0:1, :],
                'month': dataset.dataset.label[task][indices, 1:2, :],
                'season': dataset.dataset.label[task][indices, 2:3, :],
            }
    elif isinstance(dataset, CoSSMic):
        # get and process configurations
        season = kwargs.get('season', None)
        area = kwargs.get('area', None)
        
        # season
        selected_seasons = []
        if season is None:
            print('Warning: season is None, using whole_year')
            season = 'whole_year'
        if season == 'whole_year':
            selected_seasons = []
            pass # no need to filter with season, leave mapped_seaon empty
        elif season in NAME_SEASONS:
            selected_seasons = [season]
        else:
            raise NotImplementedError(f'season {season} not implemented. must be one of {NAME_SEASONS} or "whole_year"')
        mapped_season = []
        if 'season' in dataset.process_option['target_labels']:
            for _sel_season in selected_seasons:
                _sel_season = dataset.map_label(season=_sel_season)
                mapped_season.append(_sel_season)
        
        # area
        selected_areas = []
        if area is None:
            print('Warning: area is None, using all areas')
            area = 'all'
        if area == 'all':
            selected_areas = []
            pass
        elif area in dataset.condition_mapping['area'].keys():
            selected_areas: List[str] = [area] # one area
        mapped_area = []
        if 'area' in dataset.process_option['target_labels']:
            for _sel_area in selected_areas:
                _sel_area = dataset.map_label(area=_sel_area)
                mapped_area.append(_sel_area)
        
        # get profile and condition from configurations
        for task in final_profile.keys():
            profile = dataset.dataset.profile[task]  # shape: (N, 1, T) where T is the length of the profile
            label = dataset.dataset.label[task] # shape: (N, d, 1) where d is the dimension of the label
            
            if mapped_area:
                area_dim = (~torch.isnan(mapped_area[0])).nonzero().item() # get the area dimension
                _mapped_area = torch.stack([_label[area_dim] for _label in mapped_area], dim=0) # all selected areas (num_sel_areas, )
                indices = torch.isin(label[:, area_dim, 0], _mapped_area)
                profile = profile[indices] # (*, 1, T)
                label = label[indices] # (*, d, 1)
            if mapped_season:
                season_dim = (~torch.isnan(mapped_season[0])).nonzero().item()
                _mapped_season = torch.stack([_label[season_dim] for _label in mapped_season], dim=0)
                indices = torch.isin(label[:, season_dim, 0], _mapped_season)
                profile = profile[indices]
                label = label[indices]

            assert len(profile) > 0, f'No profile found for task {task} with season {season} and area {area}'
            final_profile[task] = profile
            stacked_cond_chunk = label.split(
                list(dataset.dict_cond_dim.values()), dim=1 # channel_dim
            )
            final_condition[task] = {
                cond_name: cond_tensor for cond_name, cond_tensor in zip(
                    dataset.dict_cond_dim.keys(), stacked_cond_chunk
                )
            }# 
    else:
        raise NotImplementedError(f'Dataset {type(dataset)} not implemented')

    # # ugly but for backward compatibility. 
    # train_seq, val_seq, test_seq = final_profile['train'], final_profile['val'], final_profile['test']
    # train_cond, val_cond, test_cond = final_condition['train'], final_condition['val'], final_condition['test']

    return final_profile, final_condition

def get_generated_filename(exp_config: ExperimentConfig, model: str, gmm_num_components: int = 10):
    " filename = {model related} + {dataset related} + {resolution season} + {sampling related}"
    # assert model in ['copula', 'gmm', 'ddpm', 'ddpm-calibrated']
    filename = ''
    # model related
    if model == 'copula':
        filename += 'copula'
    elif model == 'gmm':
        filename += f'gmm_{gmm_num_components}_components'
    elif model in ['ddpm','ddpm-calibrated']:
        filename += f'{model}_{exp_config.model.load_time_id}'
    else:
        filename += f'{model}'
        # raise ValueError(f'Unknown model {model}')
    
    # dataset related
    if exp_config.data.dataset == 'cossmic':
        _cossmic_names = '-'.join(exp_config.data.subdataset_names)
        filename += f'_{exp_config.data.dataset}_{_cossmic_names}'
    else:
        filename += f'_{exp_config.data.dataset}'
    filename += '_samples' # convention
    
    # resolution season
    filename += f'_{exp_config.data.resolution}_{exp_config.data.val_season}'
    
    # sampling related
    if model in ['ddpm', 'ddpm-calibrated']:
        filename += f'_{exp_config.sample.num_sampling_step}_{exp_config.diffusion.num_diffusion_step}'
        if exp_config.sample.cfg_scale != 1.:
            filename += f'_cfg_{exp_config.sample.cfg_scale*10:.0f}'
            
    filename += '.pt'
    return filename

def create_rectified_flow(
    base_model: torch.nn.Module,
    seq_length: int,
    rf_config: RectifiedFlowConfig,
    num_discretization_step: int = 1000, # <- sample config
    rescale_t: bool = False, # ALWAYS
) -> RectifiedFlow:
    """
    Create a rectified flow model from a base model.
    
    :param base_model: the base model
    :param seq_length: the length of the sequence
    :param num_discretization_step: the number of discretization steps
    :param prediction_type: the type of prediction
    :param schedule_type: the type of schedule
    :param loc: the location parameter of the normal distribution
    :param scale: the scale parameter of the normal distribution
    :param num_sampling_timestep: the number of sampling steps
    """
    if rf_config.prediction_type == 'velocity':
        prediction_type = RFPredictionType.VELOCITY
    elif rf_config.prediction_type == 'noise':
        prediction_type = RFPredictionType.NOISE
    else:
        raise ValueError(f"Unknown prediction type {rf_config.prediction_type}.")
    
    if rf_config.schedule_type == 'logit_normal':
        schedule_type = RFScheduleType.LOGIT_NORMAL
    elif rf_config.schedule_type == 'cosmap':
        schedule_type = RFScheduleType.COSMAP
    elif rf_config.schedule_type == 'uniform':
        schedule_type = RFScheduleType.UNIFORM
    else:
        raise ValueError(f"Unknown schedule type {rf_config.schedule_type}.")
    
    return RectifiedFlow(
        base_model=base_model,
        seq_length=seq_length,
        num_discretization_step=num_discretization_step,
        prediction_type=prediction_type,
        schedule_type=schedule_type,
        rescale_t=rescale_t,
)
    
class Pipeline():
    r"""
    Create a general pipeline that handles training, sampling, saving, and loading.
    ---
    Components:
        - backbone: the backbone model
        - diffusion: the diffusion model (or rectified flow model)
        - dataset: the dataset
        - trainer: 
            - optimizer
            - acclerator
            - ema
    """
    def __init__(self):
        pass
    
    