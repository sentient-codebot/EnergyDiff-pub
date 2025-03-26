from dataclasses import dataclass, fields, field

from easy_ml_config import BaseConfig
        
@dataclass
class DataConfig(BaseConfig):
    dataset: str
    root: str
    resolution: str
    load: bool
    normalize: bool
    pit: bool
    shuffle: bool
    vectorize: bool
    style_vectorize: str
    vectorize_window_size: int
    train_season: str
    val_season: str
    target_labels: str
    scaling_factor: list[float]|None = None
    
@dataclass
class CossmicDataConfig(DataConfig):
    subdataset_names: str = 'grid_import_residential'
    val_area: str = 'all'
        
@dataclass
class ModelConfig(BaseConfig):
    model_class: str
    dim_base: int
    conditioning: bool = False
    cond_dropout: float = 0.1
    dropout: float = 0.1
    num_attn_head: int = 4
    dim_feedforward: int = 2048
    learn_variance: bool = False
    num_in_channel: int = -1 # -1: uninitialized
    seq_length: int = -1 # -1: uninitialized
    
    load_time_id: str|None = None
    load_milestone: int|None = None
    resume: bool = False
    freeze_layers: bool = False
    
@dataclass
class TransformerConfig(ModelConfig):
    num_encoder_layer: int = 6
    num_decoder_layer: int = 6
    
@dataclass
class UNetConfig(ModelConfig):
    dim_mult: tuple[int]|list[int] = (1,2,4,8)
    
@dataclass
class DiffusionConfig(BaseConfig):
    prediction_type: str
    use_rectified_flow: bool
    
@dataclass
class DDPMConfig(DiffusionConfig):
    num_diffusion_step: int = 1000
    learn_variance: bool = False
    sigma_small: bool = True
    beta_schedule_type: str = 'cosine'
    
    def __post_init__(self):
        assert self.beta_schedule_type in {
            'cosine',
            'linear',
        }
        assert self.prediction_type in {
            'pred_v',
            'pred_noise',
            'pred_x0'
        }
    
@dataclass
class RectifiedFlowConfig(DiffusionConfig):
    schedule_type: str = 'logit_normal' # t-sampler
    
    def __post_init__(self):
        self.use_rectified_flow = True
        assert self.prediction_type in {
            'noise',
            'velocity',
        }
        
    @property
    def num_diffusion_step(self):
        "for consistency in filenames. define as property to protect."
        return 1000 
    
@dataclass
class SampleConfig(BaseConfig):
    num_sample: int = 512
    val_batch_size: int = 256
    num_sampling_step: int = 50
    dpm_solver_sample: bool = True
    cfg_scale: float = 1.
    
@dataclass
class TrainConfig(BaseConfig):
    batch_size: int
    val_sample_config: SampleConfig = field(default_factory=SampleConfig)
    lr: float = 1e-4
    adam_betas: tuple[float] = (0.9, 0.999)
    gradient_accumulate_every: int = 1
    ema_update_every: int = 5
    ema_decay: float = 0.9999
    amp: bool = True
    mixed_precision_type: str = 'fp16'
    split_batches: bool = True
    num_train_step: int = 50000
    save_and_sample_every: int = 50000
    val_every: int = 1250
    val_batch_size: int = 256
    
    subconfigs = {'val_sample_config': SampleConfig}
    
    @classmethod
    def init_subconfig(cls, subconfig_name, subconfig_dict) -> SampleConfig|dict:
        if subconfig_name not in cls.subconfigs:
            return subconfig_dict
        if subconfig_name == 'val_sample_config':
            return SampleConfig.from_dict(subconfig_dict)
    
@dataclass
class ExperimentConfig(BaseConfig):
    exp_id: str
    data: DataConfig
    model: ModelConfig
    diffusion: DDPMConfig
    train: TrainConfig
    sample: SampleConfig
    log_wandb: bool = False
    time_id: str|None = None
    wandb_id: str|None = None # None: no wandb logging or has not been initialized
    wandb_project: str|None = None
    vae_wandb_id: str|None = None
    gan_wandb_id: str|None = None
    
    subconfigs = {
        'data': DataConfig,
        'model': ModelConfig,
        'diffusion': DDPMConfig,
        'train': TrainConfig,
        'sample': SampleConfig
    }
    
    @classmethod
    def init_subconfig(cls, subconfig_name, subconfig_dict) -> BaseConfig|dict:
        if subconfig_name not in cls.subconfigs:
            return subconfig_dict
        if subconfig_name == 'data':
            if subconfig_dict['dataset'] == 'cossmic':
                return CossmicDataConfig.from_dict(subconfig_dict)
            else:
                return DataConfig.from_dict(subconfig_dict)
        if subconfig_name == 'model':
            if subconfig_dict['model_class'] == 'unet':
                return UNetConfig.from_dict(subconfig_dict)
            elif subconfig_dict['model_class'] in ['transformer', 'gpt2']:
                return TransformerConfig.from_dict(subconfig_dict)
            else:
                return ModelConfig.from_dict(subconfig_dict)
        if subconfig_name == 'diffusion':
            if subconfig_dict['use_rectified_flow']:
                return RectifiedFlowConfig.from_dict(subconfig_dict)
            else:
                return DDPMConfig.from_dict(subconfig_dict)
        if subconfig_name == 'train':
            return TrainConfig.from_dict(subconfig_dict)
        if subconfig_name == 'sample':
            return SampleConfig.from_dict(subconfig_dict)
    
if __name__ == "__main__":
    model_config = ModelConfig.from_yaml("model_config.yaml")
    train_config = TrainConfig(batch_size=32)
    sample_config = SampleConfig(num_sample=100)
    
    exp_config = ExperimentConfig(exp_id=1,
                                    model=model_config,
                                    train=train_config,
                                    sample=sample_config)
    print(exp_config.model.num_layer)
    exp_config.to_yaml("exp_config.yaml")
    pass