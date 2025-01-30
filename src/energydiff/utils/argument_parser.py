from functools import partial
from argparse import ArgumentParser, Namespace
import os
import yaml
import warnings

from energydiff.utils import configuration

def str2tuple(str, dtype=int):
    " str: '[1,2,3]' "
    str = str.replace(" ", "")
    if str[0] == '[' and str[-1] == ']' or str[0] == '(' and str[-1] == ')':
        str = str[1:-1]
    return tuple(map(dtype, str.split(',')))

str2float_tuple = partial(str2tuple, dtype=float)
str2int_tuple = partial(str2tuple, dtype=int)
str2str_tuple = partial(str2tuple, dtype=str)

def str2bool(str):
    if isinstance(str, bool):
        return str
    if str.upper() in {'TRUE', 'T', '1'}:
        return True
    elif str.upper() in {'FALSE', 'F', '0'}:
        return False
    else:
        raise ValueError('Invalid boolean value')

def argument_parser() -> configuration.ExperimentConfig:
    "parse all the arguments here"
    config_parser = ArgumentParser(
        prog='HeatoDiff',
        description='parse yaml configs',
        add_help=False)
    config_parser.add_argument('--config', '--configs', default='configs/unet.yaml', type=str)
    
    parser = ArgumentParser(
        prog='HeatoDiff',
        description='train neural network for heat pump consumption profile generation'
    )
    
    # General Parameters
    parser.add_argument('--exp_id', default='0.0.0', type=str, help='experiment id')
    
    # Data Parameters
    # parser.add_argument('--data_path', default='/home/nlin/data/volume_2/heat_profile_dataset/2019_data_15min.hdf5', type=str, help='path to data')
    parser.add_argument('--dataset', default='not specified', type=str, help='dataset name, should be one of ["wpuq", "lcl_electricity"]')
    parser.add_argument('--data_root', default='/home/nlin/data/volume_2/heat_profile_dataset', type=str, help='root directory of data')
    parser.add_argument('--data_case', default='2019_data_15min.hdf5', type=str, help='case name of data') # NOTE: this is not used
    parser.add_argument('--target_labels', default=None, type=str2str_tuple, help='target labels of data, e.g. ["year","season"]')
    parser.add_argument('--resolution', default='15min', type=str, help='resolution of data, one of [10s, 1min, 15min, 30min, 1h]')
    parser.add_argument('--cossmic_dataset_names', default=[], type=str2str_tuple, help='dataset names of CoSSMic, e.g. ["grid-import_residential"]')
    parser.add_argument('--season', default='winter', type=str, help='season of data') # NOTE: this is not used
    parser.add_argument('--train_season', default='winter', type=str, help='season of training data, winter/spring/.../whole_year')
    parser.add_argument('--val_season', default='winter', type=str, help='season of validation data, winter/spring/.../whole_year')
    parser.add_argument('--val_area', default='all', type=str, help='area of validation data, industrial/residential/public')
    parser.add_argument('--load_data', default=True, type=str2bool, help='whether to load processed data')
    parser.add_argument('--normalize_data', default=True, type=str2bool, help='whether to normalize data')
    parser.add_argument('--pit_data', default=False, type=str2bool, help='whether to PIT data')
    parser.add_argument('--shuffle_data', default=True, type=str2bool, help='whether to shuffle data')
    parser.add_argument('--vectorize_data', default=False, type=str2bool, help='whether to vectorize data')
    parser.add_argument('--style_vectorize', default=False, type=str, help='whether to vectorize data with style, \
        should be one of ["chronological", "stft"]')
    parser.add_argument('--vectorize_window_size', default=3, type=int, help='window size for vectorization')
    # NOTE train/val/test ratio is not used anymore. 
    parser.add_argument('--train_ratio', default=0.7, type=float, help='ratio of training data')
    parser.add_argument('--val_ratio', default=0.15, type=float, help='ratio of validation data')
    parser.add_argument('--test_ratio', default=0.15, type=float, help='ratio of testing data')
    
    # Network Parameters
    parser.add_argument('--model_class', default='unet', type=str, help='model class, should be one of ["unet", "gpt2", "mlp]')
    parser.add_argument('--conditioning', default=False, type=str2bool, help='whether to use conditioning')
    parser.add_argument('--cond_dropout', default=0.1, type=float, help='dropout rate for conditioning')
    parser.add_argument('--dim_base', default=128, type=int, help='base dimension for convs')
    parser.add_argument('--dim_mult', default=(1, 2, 4, 8), type=str2int_tuple, help='dimension multiplier for convs')
    
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--num_attn_head', default=4, type=int, help='number of attention heads')
    # parser.add_argument('--type_transformer', default='gpt2', type=str, help='type of transformer, \
    #     should be one of ["gpt2", "transformer"]')
    parser.add_argument('--num_encoder_layer', default=6, type=int, help='number of encoder layers')
    parser.add_argument('--num_decoder_layer', default=6, type=int, help='number of decoder layers')
    parser.add_argument('--dim_feedforward', default=2048, type=int, help='dimension of feedforward layer')
    
    # Diffusion Parameters
    parser.add_argument('--use_rectified_flow', default=False, type=str2bool, help='whether to use rectified flow')
    parser.add_argument('--num_diffusion_step', default=1000, type=int, help='number of diffusion steps')
    parser.add_argument('--num_sampling_step', default=-1, type=int, help='number of sampling steps')
    parser.add_argument('--dpm_solver_sample', default=False, type=str2bool, help='whether to use dpm solver for sampling')
    parser.add_argument('--diffusion_objective', default='pred_v', choices=['pred_v', 'pred_noise', 'pred_x0'],
                        type=str, help='objective for diffusion, \
        should be one of ["pred_v", "pred_noise", "pred_x0"]')
    parser.add_argument('--learn_variance', default=False, type=str2bool, help='whether to learn variance')
    parser.add_argument('--sigma_small', default=True, type=str2bool, help='whether to use small sigma if not learned')
    parser.add_argument('--beta_schedule_type', default='cosine', type=str, help='type of beta schedule, \
        should be one of ["cosine", "linear"]')
    parser.add_argument('--ddim_sampling_eta', default=0., type=float, help='eta parameter of the ddim sampling')
    parser.add_argument('--cfg_scale', default=1., type=float, help='classfier-free guidance scale, default=1. (no guidance)')
    
    # Training Parameters
    #   batch size and optimizer
    parser.add_argument('--mse_loss', default=True, type=str2bool, help='whether to use MSE loss or KL loss')
    parser.add_argument('--rescale_learned_variance', default=True, type=str2bool, help='whether to rescale loss of learned variance')
    parser.add_argument('--only_central', default=False, type=str2bool, help='only use the central channel of output for loss.')
    parser.add_argument('--train_batch_size', default=480, type=int, help='batch size')
    parser.add_argument('--train_lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--adam_betas', default=(0.9,0.999), type=str2float_tuple, help='betas for adam optimizer')
    #   trainer and ema
    parser.add_argument('--gradient_accumulate_every', default=2, type=int, help='gradient accumulation steps')
    parser.add_argument('--ema_update_every', default=10, type=int, help='ema update steps')
    parser.add_argument('--ema_decay', default=0.995, type=float, help='ema decay')
    parser.add_argument('--amp', default=False, type=str2bool, help='whether to use amp')
    parser.add_argument('--mixed_precision_type', default='fp16', type=str, help='mixed precision type, should be one of ["fp16", "fp32"]')
    parser.add_argument('--split_batches', default=True, type=str2bool, help='whether to split batches for accelerator')
    #   train and logging steps
    parser.add_argument('--num_train_step', default=5000, type=int, help='number of training steps')
    parser.add_argument('--save_and_sample_every', default=500, type=int, help='save and sample every n steps')
    parser.add_argument('--val_every', default=1000, type=int, help='validate every n steps. save_and_sample_every must be a multiple of val_every')
    parser.add_argument('--num_sample', default=25, type=int, help='number of samples to generate every n steps')
    parser.add_argument('--log_wandb', default=True, type=str2bool, help='whether to log to wandb')
    
    # Sample and test Parameters
    parser.add_argument('--val_batch_size', default=None, type=int, help='batch size for validation/testing')
    parser.add_argument('--set_time_id', default=None, type=str, help='manually set a time id to start')
    parser.add_argument('--load_time_id', default=None, type=str, help='time id to load from')
    parser.add_argument('--load_milestone', default=10, type=int, help='milestone to load')
    parser.add_argument('--resume', default=False, action='store_true', help='whether to resume training from a milestone')
    parser.add_argument('--freeze_layers', default=False, action='store_true', help='whether to freeze layers of transformer')
    
    # Parsing arguments
    #   Step 0: Parse args in config yaml
    args, left_argv = config_parser.parse_known_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            config_yaml_dict = yaml.safe_load(f)
        
        yaml_argv = []
        for key, value in config_yaml_dict.items():
            yaml_argv.append('--' + key)
            yaml_argv.append(str(value))
        parser.parse_known_args(yaml_argv, namespace=args) # write config yaml values to args
    #   Step 1: Parse args in command line
    parser.parse_args(left_argv, namespace=args)
    
    # Post processing
    if args.dataset == 'lcl_electricity':
        args.resolution = '30min' # fixed. 
    if args.dataset == 'cossmic':
        assert len(args.cossmic_dataset_names) > 0, 'cossmic_dataset_names is empty.'
    if args.val_batch_size is None:
        args.val_batch_size = args.train_batch_size
    if args.dataset == 'not specified':
        raise ValueError('dataset is not specified.')
    if not args.shuffle_data:
        warnings.warn('shuffle_data is False.')
    #   train/val/test ratio
    # NOTE not used anymore
    _sum = args.train_ratio + args.val_ratio + args.test_ratio
    args.train_ratio /= _sum
    args.val_ratio /= _sum
    args.test_ratio /= _sum
    
    #   num sampling step
    if args.num_sampling_step == -1:
        args.num_sampling_step = args.num_diffusion_step
        
    # 2. store args into Config
    # 2.1 data
    data_config = configuration.DataConfig(
        dataset=args.dataset,
        root=args.data_root,
        resolution=args.resolution,
        load=args.load_data,
        normalize=args.normalize_data,
        pit=args.pit_data,
        shuffle=args.shuffle_data,
        vectorize=args.vectorize_data,
        style_vectorize=args.style_vectorize,
        vectorize_window_size=args.vectorize_window_size,
        train_season=args.train_season,
        val_season=args.val_season,
        target_labels=args.target_labels
    )
    if args.dataset == 'cossmic':
        data_config = configuration.CossmicDataConfig.inherit(
            data_config,
            subdataset_names=args.cossmic_dataset_names,
            val_area=args.val_area
        )
        
    # 2.2 model
    model_config = configuration.ModelConfig(
        model_class=args.model_class,
        dim_base=args.dim_base,
        conditioning=args.conditioning,
        cond_dropout=args.cond_dropout,
        dropout=args.dropout,
        num_attn_head=args.num_attn_head,
        dim_feedforward=args.dim_feedforward,
        learn_variance=args.learn_variance,
        load_time_id=args.load_time_id,
        load_milestone=args.load_milestone,
        resume=args.resume,
        freeze_layers=args.freeze_layers
    )
    if args.model_class in ['transformer','gpt2']:
        model_config = configuration.TransformerConfig.inherit(
            model_config,
            num_encoder_layer=args.num_encoder_layer,
            num_decoder_layer=args.num_decoder_layer
        )
    elif args.model_class == 'unet':
        model_config = configuration.UnetConfig.inherit(
            model_config,
            dim_mult=args.dim_mult
        )
        
    # 2.3 diffusion
    diffusion_config = configuration.DiffusionConfig(
        prediction_type=args.diffusion_objective,
        use_rectified_flow=args.use_rectified_flow
    )
    if args.use_rectified_flow:
        diffusion_config = configuration.RectifiedFlowConfig.inherit(
            diffusion_config,
        )
    else:
        diffusion_config = configuration.DDPMConfig.inherit(
            diffusion_config,
            num_diffusion_step=args.num_diffusion_step,
            learn_variance=args.learn_variance,
            sigma_small=args.sigma_small,
            beta_schedule_type=args.beta_schedule_type,
        )
        
    # 2.4 training
    val_sample_config = configuration.SampleConfig(
        num_sample=args.num_sample,
        val_batch_size=args.val_batch_size,
        num_sampling_step=args.num_sampling_step,
        dpm_solver_sample=args.dpm_solver_sample,
        cfg_scale=args.cfg_scale
    )
    train_config = configuration.TrainConfig(
        batch_size=args.train_batch_size,
        val_sample_config=val_sample_config,
        lr=args.train_lr,
        adam_betas=args.adam_betas,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_update_every=args.ema_update_every,
        ema_decay=args.ema_decay,
        amp=args.amp,
        mixed_precision_type=args.mixed_precision_type,
        split_batches=args.split_batches,
        num_train_step=args.num_train_step,
        save_and_sample_every=args.save_and_sample_every,
        val_every=args.val_every,
        val_batch_size=args.val_batch_size,
    )
    
    # 2.5 final sample config
    final_sample_config = configuration.SampleConfig(
        num_sample=4000,
        val_batch_size=args.val_batch_size,
        num_sampling_step=100 if not args.use_rectified_flow else 50,
        dpm_solver_sample=args.dpm_solver_sample,
        cfg_scale=args.cfg_scale
    )
    
    # 2.5 experiment
    exp_config = configuration.ExperimentConfig(
        exp_id=args.exp_id,
        data=data_config,
        model=model_config,
        diffusion=diffusion_config,
        train=train_config,
        sample=final_sample_config,
        log_wandb=args.log_wandb,
        time_id=args.set_time_id # str|None
    )
    
    return exp_config

def save_config(exp_config: configuration.ExperimentConfig, time_id: str) -> None:
    "save config to yaml"
    os.makedirs('results/configs', exist_ok=True)
    exp_config.to_yaml(f'results/configs/exp_config_{time_id}.yaml')
    
def inference_parser() -> configuration.ExperimentConfig:
    r"""Fast way to parse arguments for inference
    
    Required:
        - load_time_id: str
        - [dep] load_milestone: int
        
    **All unspecified arguments will be the same as the training arguments.**
    
    Preferrably to be specified:
        - val_batch_size: int
        - num_sample: int
        - num_sampling_step: int
        - dpm_solver_sample: bool
    
    """
    parser = ArgumentParser(
        prog='HeatoDiff Inference',
        description='load and inference diffusion. '
    )
    parser.add_argument('--load_time_id', type=str, help='time id to load')
    # parser.add_argument('--load_milestone', required=True, type=int, help='milestone to load')
    parser.add_argument('--val_batch_size', default=None, type=int, help='batch size for validation/testing')
    parser.add_argument('--num_sample', default=None, type=int, help='number of samples to generate every n steps')
    parser.add_argument('--num_sampling_step', default=None, type=int, help='number of sampling steps')
    parser.add_argument('--dpm_solver_sample', default=None, type=str2bool, help='whether to use dpm solver for sampling')
    
    args, _ = parser.parse_known_args()
    
    # load training config
    exp_config = configuration.ExperimentConfig.from_yaml(f'results/configs/exp_config_{args.load_time_id}.yaml')
    
    exp_config.model.load_time_id = args.load_time_id
    # exp_config.model.load_milestone = args.load_milestone
    if args.val_batch_size is not None:
        exp_config.sample.val_batch_size = args.val_batch_size
    if args.num_sample is not None:
        exp_config.sample.num_sample = args.num_sample
    if args.num_sampling_step is not None:
        exp_config.sample.num_sampling_step = args.num_sampling_step
    if args.dpm_solver_sample is not None:
        exp_config.sample.dpm_solver_sample = args.dpm_solver_sample
    
    return exp_config