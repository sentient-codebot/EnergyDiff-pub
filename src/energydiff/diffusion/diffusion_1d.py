import os
import inspect
import math
from pathlib import Path
from multiprocessing import cpu_count
from random import random
from functools import partial
from typing import Optional, Any, Optional, Callable, Sequence, Iterable
from .typing import Float, Data1D, Int

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim import Adam, AdamW
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce, einsum
from einops.layers.torch import Rearrange

import pytorch_lightning as pl
from accelerate import Accelerator 
from ema_pytorch import EMA

from tqdm.auto import tqdm
import wandb

from .version import __version__
from .rectified_flow import RectifiedFlow
from .utils import default, identify, discretized_gaussian_log_likelihood, continuous_gaussian_log_likelihood, normal_kl
from .models_1d import Transformer1D, Unet1D, DenoisingMLP1D
from ..utils.configuration import TrainConfig, ModelConfig, DiffusionConfig, DDPMConfig
from .typing import ModelMeanType, ModelVarianceType, LossType, BetaScheduleType, ModelPrediction
        
# diffusion modules
def extract(a: Tensor, t: Tensor, x_shape: torch.Size):
    """
        a: Float[Tensor, 'num_timestep, '], device cuda/cpu, dtype float64/32
        t: Int[Tensor, 'batch, ']
        x_shape: (batch, channel, sequence)
    """
    a = a.to(device=t.device, dtype=torch.float32)
    b, *_ = t.shape
    dim_x = len(x_shape)
    out = a.gather(-1, t)
    target_shape = (b, *(1 for _ in range(dim_x - 1)))
    out = out.reshape(*target_shape) + torch.zeros(x_shape, device=t.device)
    
    return out

# NOTE just for reference here, another more readable implementation for extract
# def _extract_into_tensor(arr, timesteps, broadcast_shape):
#     """
#     Extract values from a 1-D numpy array for a batch of indices.
#     :param arr: the 1-D numpy array.
#     :param timesteps: a tensor of indices into the array to extract.
#     :param broadcast_shape: a larger shape of K dimensions with the batch
#                             dimension equal to the length of timesteps.
#     :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
#     """
#     res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
#     while len(res.shape) < len(broadcast_shape):
#         res = res[..., None] # adding dimensions until equal
#     return res + torch.zeros(broadcast_shape, device=timesteps.device) # smartly use addition to broadcast
    
def linear_beta_schedule(num_timestep: int):
    scale = 1000. / num_timestep
    beta_start = scale * 1e-4
    beta_end = scale * 2e-2
    return torch.linspace(beta_start, beta_end, num_timestep, dtype=torch.float64) # shape: (num_timestep,)
    
def cosine_beta_schedule(num_timestep: int, s: float = 0.008):
    " as proposed in https://openreview.net/forum?id=-NEXDKk8gZ "
    num_step = num_timestep + 1
    x = torch.linspace(0, num_timestep, num_step, dtype=torch.float64) # shape: (num_step,)
    alpha_cumprod = torch.cos(
        (x/num_timestep+s)/(1+s) * math.pi/2
    ) ** 2 # shape: (num_step,)
    alpha_cumprod = alpha_cumprod / alpha_cumprod[0] # shouldn't [0] be 1.?
    beta_schedule = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1]) # shape: (num_timestep,)
    return torch.clip(beta_schedule, 0, 0.999) # shape: (num_timestep,)
    
def get_named_beta_schedule(beta_schedule_type: BetaScheduleType, num_timestep: int):
    if beta_schedule_type == BetaScheduleType.LINEAR:
        return linear_beta_schedule(num_timestep)
    elif beta_schedule_type == BetaScheduleType.COSINE:
        return cosine_beta_schedule(num_timestep)
    else:
        raise ValueError('type_beta_schedule must be one of linear, cosine')
    
def normalize_to_neg_one_to_one(x):
    return x * 2. - 1.

def unnormalize_to_zero_to_one(x):
    return (x + 1.) / 2.
    
class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        base_model: nn.Module|None = None,
        seq_length: int|None = None,
        num_timestep: int = 1000,
        num_sampling_timestep = None,
        model_mean_type: ModelMeanType = ModelMeanType.NOISE,
        model_variance_type: ModelVarianceType = ModelVarianceType.FIXED_SMALL,
        loss_type: LossType = LossType.MSE,
        beta_schedule_type: BetaScheduleType = BetaScheduleType.COSINE,
        beta_schedule: Optional[Tensor] = None,
        ddim_sampling_eta: float = 0.,
        auto_normalize: bool = False,
        loss_only_central_channel: bool = False,
        fft_loss_weight: float = 0.,
    ):
        super().__init__()
        self.model = base_model
        self.num_in_channel = base_model.num_in_channel
        self.self_condition = base_model.self_condition
        self.conditioning = base_model.conditioning
        
        self.seq_length = seq_length
        self.model_mean_type = model_mean_type
        self.model_variance_type = model_variance_type
        self.beta_schedule_type = beta_schedule_type
        self.loss_type = loss_type
        
        self.fft_loss: bool = fft_loss_weight > 0.
        self.fft_loss_weight: float = fft_loss_weight
        self.loss_only_central_channel = False # deprecated
        
        # Check arguments
        assert model_mean_type in ModelMeanType, \
            'objective must be ModelMeanType.X_START, ModelMeanType.NOISE, ModelMeanType.V'
        assert model_variance_type in ModelVarianceType, \
            'model_variance_type must be ModelVarianceType.FIXED_SMALL, ModelVarianceType.FIXED_LARGE, ModelVarianceType.LEARNED_RANGE'
        assert beta_schedule_type in BetaScheduleType, \
            'type_beta_schedule must be one of linear, cosine'
        
        # Calculate beta schedule
        if beta_schedule is not None:
            if beta_schedule.shape == (num_timestep,):
                beta_schedule = beta_schedule.clone().to(torch.float64)
        else:
            beta_schedule = get_named_beta_schedule(beta_schedule_type, num_timestep)
        
        alpha = 1. - beta_schedule
        alpha_cumprod = torch.cumprod(alpha, dim=0) # cumulative product
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.) # replace beginning with 1.
        
        self.num_timestep = int(num_timestep)
        
        # sampling related parameters
        self.num_sampling_timestep = default(num_sampling_timestep, num_timestep)
        
        assert self.num_sampling_timestep <= self.num_timestep
        # self.is_ddim_sampling = self.num_sampling_timestep < self.num_timestep
        self.is_ddim_sampling = False # turned off until fixed. using spaced sampling. 
        self.is_subseq_sampling = self.num_sampling_timestep < self.num_timestep
        sampling_subseq = torch.linspace(0, self.num_timestep-1, self.num_sampling_timestep, dtype=torch.long) # [0, S_1, ..., S_K]
        self.ddim_sampling_eta = ddim_sampling_eta
        
        # helper functions for float64 -> float32
        
        self.beta_schedule = beta_schedule  # shape: (num_timestep,)
        # self.sampling_subseq = sampling_subseq  # shape: (num_sampling_timestep,)
        self.alpha_cumprod = alpha_cumprod  # shape: (num_timestep,)
        self.alpha_cumprod_prev = alpha_cumprod_prev  # shape: (num_timestep,)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod) # shape: (num_timestep,)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - alpha_cumprod) # shape: (num_timestep,)
        self.log_one_minus_alpha_cumprod = torch.log(1. - alpha_cumprod) # shape: (num_timestep,)
        self.sqrt_recip_alpha_cumprod = torch.rsqrt(alpha_cumprod) # shape: (num_timestep,)
        self.sqrt_recipm1_alpha_cumprod = torch.sqrt(1. / alpha_cumprod - 1.) # shape: (num_timestep,)
        
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = beta_schedule * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)
        self.posterior_variance = posterior_variance # shape: (num_timestep,)
        
        # note: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.cat([posterior_variance[1:2], posterior_variance[1:]], dim=0).log()
        self.posterior_mean_coef1 = beta_schedule * torch.sqrt(alpha_cumprod_prev) / (1. - alpha_cumprod)
        self.posterior_mean_coef2 = (1. - alpha_cumprod_prev) * torch.sqrt(alpha) / (1. - alpha_cumprod)
        
        # whether to normalize
        #   original range: 0 ~ 1
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identify
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identify
        
    def predict_start_from_noise(self, 
                                 x_t: Data1D, 
                                 t: Int[Tensor, 'batch, '], 
                                 noise: Data1D
                                 ) -> Data1D:
        """
            As an example for `extract` function. self.sqrt_... is a tensor of shape (num_timestep, ).
        t is a tensor of shape (batch, ), so we need to extract the corresponding value from self.sqrt_...
        x_t is a tensor of shape (batch, num_in_channel, sequence), so we need to extract the corresponding value from noise.
        """
        return (
            extract(self.sqrt_recip_alpha_cumprod, t, x_t.shape) * x_t - 
            extract(self.sqrt_recipm1_alpha_cumprod, t, x_t.shape) * noise
        )
        
    def predict_noise_from_start(self, x_t, t, x0):
        return(
            (extract(self.sqrt_recip_alpha_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alpha_cumprod, t, x_t.shape)
        )
        
    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alpha_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alpha_cumprod, t, x_start.shape) * x_start
        )
    
    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alpha_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alpha_cumprod, t, x_t.shape) * v
        )
        
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t|x_0)
        :param x_start: x_0, shape (batch, channel, sequence)
        :param t: timestep, shape (batch, )
        
        :return mean, variance, log_variance
        """
        mean = extract(self.sqrt_alpha_cumprod, t, x_start.shape) * x_start
        variance = extract(self.sqrt_one_minus_alpha_cumprod ** 2, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alpha_cumprod, t, x_start.shape)
        return mean, variance, log_variance
        
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Get the distribution q(x_{t-1}|x_t, x_0)
        
        The returned var/log_var is the ModelVarType.FIXED_SMALL case. can choose to ignore it. 
        
        :param x_start: x_0, shape (batch, channel, sequence)
        :param x_t: x_t, shape (batch, channel, sequence)
        :param t: timestep, shape (batch, )
        
        :return mean, variance, log_variance
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape) # shape: (batch, 1, 1)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_sample(self, x_start, t, noise=None):
        """ 
        q(x_t | x_0), the forwardd (diffusion) process 
        
        :param x_start: x_0, shape (batch, channel, sequence)
        :param t: timestep, shape (batch, )
        
        :return x_t: noisy version of x_start, shape (batch, channel, sequence)
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        return (
            extract(self.sqrt_alpha_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alpha_cumprod, t, x_start.shape) * noise
        )
    
    def model_prediction(
        self,
        x: Data1D, # diffused data x_t
        t: Int[Tensor, 'batch, '], # timestep
        c: Optional[Data1D] = None, # condition
        x_self_cond: Optional[Data1D] = None, 
        cfg_scale: float = 1., # classifier-free guidance scale
        clip_x_start: bool = False,
        rederive_pred_noise: bool = False,
    ):
        # model_output = self.model(x, t, c, x_self_cond) # shape: (batch, dim_out, sequence)
        model_output = self.model.forward_with_cfg(x, t, c=c, x_self_cond=x_self_cond, cfg_scale=cfg_scale) # shape: (batch, dim_out, sequence)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identify
        
        # separate pred_mean and pred_variance
        if self.model_variance_type == ModelVarianceType.LEARNED_RANGE:
            assert model_output.shape == (x.shape[0], x.shape[1]*2, x.shape[2])
            model_output, model_var_factor = torch.split(model_output, x.shape[1], dim=1) # var_factor is a [-1,1] scalar for \
                # interpolating in the range of [FIXED_SMALL, FIXED_LARGE]
        else:
            model_var_factor = None
        
        # calculate noise and x_start
        if self.model_mean_type == ModelMeanType.NOISE:
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            
            # TODO: why do we calculate the noise again?
            if clip_x_start and not rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.model_mean_type == ModelMeanType.X_START:
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.model_mean_type == ModelMeanType.V:
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        else:
            raise ValueError('objective must be one of pred_noise, pred_x0, pred_v')
        
        return ModelPrediction(pred_noise, x_start, model_var_factor)
    
    def p_mean_variance(self, x, t, clip_denoised: bool = False, model_kwargs = None):
        """
        Apply the model to get p(x_{t-1}|x_t), as well as a prediction of the initial x_0
        
        :param x: x_t, shape (batch, channel, sequence)
        :param t: timestep, shape (batch, )
        :param clip_denoised: whether to clip the denoised x_{t-1} to [-1, 1]
        :param model_kwargs: kwargs for model_prediction, namely
            - c: condition information, Optional[Data1D] = None,
            - x_self_cond: Optional[Data1D] = None,
            - cfg_scale: float = 1.,
            
        :return dict: {
            'model_mean': model_mean,
            'model_variance': model_variance,
            'model_log_variance': model_log_variance,
            'pred_x_start': pred_x_start,
        }
        """
        # model_kwargs: c: Optional[Data1D] = None, x_self_cond: Optional[Data1D] = None, cfg_scale: float = 1., 
        model_kwargs = default(model_kwargs, {})
        pred = self.model_prediction(x, t, **model_kwargs)
        pred_x_start = pred.pred_x_start
        var_factor = pred.pred_var_factor # only used for ModelVarianceType.LEARNED_RANGE
        
        if self.model_variance_type == ModelVarianceType.LEARNED_RANGE:
            min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
            max_log = extract(torch.log(self.beta_schedule), t, x.shape)
            frac = (var_factor + 1.) / 2. # [-1, 1] -> [0, 1]
            model_log_variance = frac * max_log + (1. - frac) * min_log
            model_variance = model_log_variance.exp()
        else:
            model_variance, model_log_variance = {
                ModelVarianceType.FIXED_LARGE: (
                    torch.cat([self.posterior_variance[1:2], self.beta_schedule[1:]], dim=0),
                    torch.log(torch.cat([self.posterior_variance[1:2], self.beta_schedule[1:]], dim=0)),
                ),
                ModelVarianceType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_variance_type]
            model_variance = extract(model_variance, t, x.shape)
            model_log_variance = extract(model_log_variance, t, x.shape)
        
        if clip_denoised:
            pred_x_start = torch.clamp(pred_x_start, min=-1., max=1.)
            
        model_mean, _, _ = self.q_posterior_mean_variance(pred_x_start, x, t) # ignore the returned variance and log_variance
        
        assert model_mean.shape == model_log_variance.shape == pred_x_start.shape == x.shape
        
        return {
            'model_mean': model_mean,
            'model_variance': model_variance,
            'model_log_variance': model_log_variance,
            'pred_x_start': pred_x_start,
        }
        # return model_mean, model_variance, model_log_variance, pred_x_start
    
    @torch.no_grad()
    def p_sample(self, x: Data1D, t: int, clip_denoised = False, model_kwargs = None):
        """
        Sample x_{t-1} from p(x_{t-1}|x_t) using self.model at given timestep t
        :param x: x_t, shape (batch, channel, sequence)
        :param t: timestep, int
        :param clip_denoised: whether to clip the denoised x_{t-1} to [-1, 1]
        :param model_kwargs: kwargs for model_prediction, such as c, x_self_cond, cfg_scale
        
        :return: {
            'pred_x_prev': pred_x_prev,
            'pred_x_start': out['pred_x_start'],
        }
        """
        # model_kwargs:
        # c: Optional[Data1D] = None, x_self_cond = None, cfg_scale = 1., 
        b, *_ = x.shape
        batched_time = torch.full((b,), t, device=x.device, dtype=torch.long)
        out_dict = self.p_mean_variance(
            x = x,
            t = batched_time, 
            clip_denoised = clip_denoised,
            model_kwargs = model_kwargs,
        )
        noise = torch.randn_like(x) if t > 0 else 0. # N(0, 1)
        pred_x_prev = out_dict['model_mean'] + (0.5 * out_dict['model_log_variance']).exp() * noise
        
        return {
            'pred_x_prev': pred_x_prev,
            'pred_x_start': out_dict['pred_x_start'],
        }
    
    @torch.no_grad()
    def p_sample_loop(self, shape, noise = None, clip_denoised = False, model_kwargs = None):
        """
        Generate sample from model
        
        :param shape: ...
        :param noise: optional, (X_T) isotropic gaussian noise, shape (batch, channel, sequence)
        :param clip_denoised: whether to clip the denoised x_{t-1} to [-1, 1]
        :param model_kwargs: kwargs for model_prediction, such as c, x_self_cond, cfg_scale
        
        :return sample: x_0, shape (batch, channel, sequence)
        """
        for sample_out in self.p_sample_loop_progressive(shape, noise, clip_denoised, model_kwargs):
            sample = sample_out['pred_x_prev']
            
        return sample
    
    @torch.no_grad()
    def p_sample_loop_progressive(self, shape, noise = None, clip_denoised = False, model_kwargs = None):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        
        NOTE: when can call this function directly so that we can access all x_t
                when we do not need all x_t, we can all p_sample_loop
        """
        b = shape[0]
        device = next(self.model.parameters()).device
        model_kwargs = default(model_kwargs, {})
        
        x_t = default(noise, torch.randn(shape, device = device)) # x_t, t = T
        x_start = None
        
        # from T-1 to 0
        for t in tqdm(reversed(range(0, self.num_timestep)), desc='sampling loop time step', total=self.num_timestep):
            self_cond = x_start if self.self_condition else None
            model_kwargs['x_self_cond'] = self_cond
            sample_out = self.p_sample(x_t, t, clip_denoised = clip_denoised, model_kwargs = model_kwargs)
            if t == 0:
                sample_out['pred_x_prev'] = self.unnormalize(sample_out['pred_x_prev'])
            yield sample_out
            x_t, x_start = sample_out['pred_x_prev'], sample_out['pred_x_start']

    def subseq_p_mean_variance(self, x, s_t, clip_denoised = False, model_kwargs = None):
        """
        only applicable when ModelVarianceType == LEARNED_RANGE
        
        same usage as p_mean_variance
        """
        model_kwargs = default(model_kwargs, {})
        pred = self.model_prediction(x, s_t, **model_kwargs)
        pred_x_start = pred.pred_x_start
        var_factor = pred.pred_var_factor # only used for ModelVarianceType.LEARNED_RANGE
        # min_log = extract(self.subseq_posterior_log_variance_min_clipped, s_t, x.shape)
        # max_log = extract(self.subseq_posterior_log_variance_max_clipped, s_t, x.shape)
        
        max_var = (1 - extract(self.alpha_cumprod / self.alpha_cumprod_prev, torch.where(s_t>0, s_t, s_t+1), x.shape))
        min_var = extract((1-self.alpha_cumprod_prev)/(1-self.alpha_cumprod), torch.where(s_t>0, s_t, s_t+1), x.shape) * max_var
        max_log = torch.log(max_var)
        min_log = torch.log(min_var)
        
        frac = (var_factor + 1.) / 2. # [-1, 1] -> [0, 1]
        model_log_variance = frac * max_log + (1. - frac) * min_log
        model_variance = model_log_variance.exp()
        
        if clip_denoised:
            pred_x_start = torch.clamp(pred_x_start, min=-1., max=1.)
            
        model_mean, _, _ = self.q_posterior_mean_variance(pred_x_start, x, s_t) # ignore the returned variance and log_variance
        
        assert model_mean.shape == model_log_variance.shape == pred_x_start.shape == x.shape
        
        return {
            'model_mean': model_mean,
            'model_variance': model_variance,
            'model_log_variance': model_log_variance,
            'pred_x_start': pred_x_start,
        }
          
    @torch.no_grad()
    def sample(self, batch_size = 16, clip_denoised = False, model_kwargs = None, post_transforms: Iterable[Callable] = []):
        """
        Wrapped sample function, either p_sample_loop or ddim_sample_loop
        """
        if post_transforms:
            assert all([callable(transform) for transform in post_transforms])
        seq_length, num_channel = self.seq_length, self.num_in_channel

        sample_fn = self.p_sample_loop
        samples = sample_fn((batch_size, num_channel, seq_length), clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        
        for transform in post_transforms:
            samples = transform(samples)
        
        return samples
    
    def _vb_terms_bpd(self, x_start, x_t, t, model_mean, model_log_variance):
        """
        Calculate a term (at t) of the vb. 
        Using a discretized gaussian distribution (can also change to continuous). 
        The result units are bits per dimension. (log prob = bits, divide by #dimension)
        
        :param x_start: x_0, shape (batch, channel, sequence)
        :param x_t: noisy x_t, shape (batch, channel, sequence)
        :param t: timestep, shape (batch, )
        :param clip_denoised: whether to clip the denoised x_{t-1} to [-1, 1]
        :param model_kwargs: kwargs for model_prediction, such as c, x_self_cond, cfg_scale
        """
        target_mean, target_variance, target_log_variance_clipped = self.q_posterior_mean_variance(x_start, x_t, t)
        kl = normal_kl(
            target_mean, target_log_variance_clipped, model_mean, model_log_variance
        ) # shape (batch, channel, sequence)
        kl = reduce(kl, 'b ... -> b', 'mean') / math.log(2.0) # shape (batch, )
        
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, model_mean, model_log_variance, n_bits=8
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = reduce(decoder_nll, 'b ... -> b', 'mean') / math.log(2.0) # shape (batch, )
        
        output = torch.where((t==0), decoder_nll, kl)
        return output
    
    def train_losses(self, x_start, t, noise=None, model_kwargs=None):
        """
        Calculate the train losses. 
            - (rescaled) mse loss: calculated from mean
            - vb term: calculated from variance and (detached) mean
            
        :param x_start: x_0, shape (batch, channel, sequence)
        :param t: timestep, shape (batch, )
        :param noise: optional, (X_T) isotropic gaussian noise, shape (batch, channel, sequence)
        :param model_kwargs: kwargs for model_prediction, such as c, x_self_cond, cfg_scale
            !NOTE: model_kwargs[cfg_scale] should be set to 1. in training. 
            
        :return loss_terms: {
            'loss': loss, shape (batch, )
            'mse': mse, shape (batch, ), if applicable
            'vb': vb, shape (batch, ), if applicable
        }
        """
        model_kwargs = default(model_kwargs, {})
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start, t, noise)
        
        loss_terms = {}
        
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            # the 0 to T-1 terms of the vb loss 
            out = self.p_mean_variance(x_t, t, clip_denoised=False, model_kwargs=model_kwargs)
            loss_terms['loss'] = self._vb_terms_bpd(
                x_start, x_t, t, 
                model_mean=out['model_mean'],
                model_log_variance=out['model_log_variance'],
            )
            if self.loss_type == LossType.RESCALED_KL:
                loss_terms['loss'] *= self.num_timestep
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            out = self.p_mean_variance(x_t, t, clip_denoised=False, model_kwargs=model_kwargs)
            
            # calculate the vb term for variance
            if self.model_variance_type in [
                ModelVarianceType.LEARNED_RANGE, # alternative is ModelVarianceType.LEARNED, but I chose not to implement
            ]:
                loss_terms['vb'] = self._vb_terms_bpd(
                    x_start,
                    x_t,
                    t,
                    model_mean=out['model_mean'].detach(),  # x_{t-1} mean
                    model_log_variance=out['model_log_variance'],  # x_{t-1} log variance
                )
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    loss_terms['vb'] *= self.num_timestep / 10000.
            
            # calculate the mse term for mean    
            target = {
                ModelMeanType.NOISE: noise,
                ModelMeanType.X_START: x_start,
                ModelMeanType.V: self.predict_v(x_start, t, noise),
            }[self.model_mean_type]
            _noise = self.predict_noise_from_start(x_t, t, out['pred_x_start'])
            source = {
                ModelMeanType.NOISE: _noise,
                ModelMeanType.X_START: out['pred_x_start'],
                ModelMeanType.V: self.predict_v(out['pred_x_start'], t, _noise),
            }[self.model_mean_type] # re-calculating v in this way is a little bit detour. 
            assert target.shape == source.shape
            _mse_loss = reduce(F.mse_loss(source, target, reduction='none'), 'b ... -> b', 'mean') # shape (batch, )
            _fft_mse_loss = reduce(
                (torch.fft.fft(source, dim=-1, norm='ortho')-torch.fft.fft(target, dim=-1, norm='ortho')).abs().square(),
                'b ... -> b', 'mean'
            ) # shape: (batch, )
            loss_terms['mse'] = (1-self.fft_loss_weight) * _mse_loss + self.fft_loss_weight * _fft_mse_loss
            if 'vb' in loss_terms:
                loss_terms['loss'] = loss_terms['vb'] + loss_terms['mse']
            else:
                loss_terms['loss'] = loss_terms['mse']
        else:
            raise NotImplementedError(self.loss_type)
            
        return loss_terms    
    
    def forward(self, x_start, noise=None, model_kwargs=None):
        """
        Calculate the loss of x_t with randomly sampled t using the model. 
        :param x_start: x_0, shape (batch, channel, sequence)
        :param t: timestep, shape (batch, )
        :param noise: optional, (X_T) isotropic gaussian noise, shape (batch, channel, sequence)
        :param model_kwargs: kwargs for model_prediction, such as c, x_self_cond, cfg_scale
        
        :return loss: loss, scalar (,)
        """
        
        "Do pre-transforms in trainer. "
        
        b, channel, l = x_start.shape
        device = x_start.device
        assert l == self.seq_length, f'input sequence length must be {self.seq_length}'
        assert channel == self.num_in_channel, f'input channel must be {self.num_in_channel}'
        
        t = torch.randint(0, self.num_timestep, (b,), device=device) # different step for each sample in the batch
        # loss = self.train_losses(x_start, t, noise, model_kwargs)['loss']
        loss_terms = self.train_losses(x_start, t, noise, model_kwargs)
        
        for k, v in loss_terms.items():
            loss_terms[k] = v.mean()
        
        return loss_terms
    
# helper function for SpacedDiffusion1D
def space_timesteps(num_timesteps, section_counts):
    """
    Ref: https://github.com/facebookresearch/DiT/blob/main/diffusion/respace.py#L12
    
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

# Respaced Diffusion for Skip step sampling
class SpacedDiffusion1D(GaussianDiffusion1D):
    def __init__(self, use_timesteps, **kwargs):
        """
        :param use_timesteps: list of timesteps to use, e.g. [0, 1, 3, 7, 8, 16, 20, 51, 78]
        """
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_timestep = kwargs['num_timestep']
        alpha_cumprod_prev = 1.
        new_betas = []
        if kwargs['beta_schedule_type'] == BetaScheduleType.LINEAR:
            betas = linear_beta_schedule(kwargs['num_timestep'])
        elif kwargs['beta_schedule_type'] == BetaScheduleType.COSINE:
            betas = cosine_beta_schedule(kwargs['num_timestep'])
        else:
            raise NotImplementedError(kwargs['beta_schedule_type'])
        alpha = 1. - betas
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        for ori_timestep, alpha_cumprod_current in enumerate(alpha_cumprod):
            if ori_timestep in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod_current/alpha_cumprod_prev)
                alpha_cumprod_prev = alpha_cumprod_current
                self.timestep_map.append(ori_timestep)
        kwargs['num_timestep'] = len(self.use_timesteps)
        kwargs['num_sampling_timestep'] = len(self.use_timesteps)
        new_betas = torch.tensor(new_betas, dtype=torch.float32)
        kwargs['beta_schedule'] = new_betas
        
        # Init GaussianDiffusion1D
        super().__init__(**kwargs)
        # Re-wrap the model
        self.model = _WrappedModel(self.model, self.timestep_map, self.original_num_timestep)

class _WrappedModel(nn.Module):
    """
    wrap a nn.Module to let it use the original time step. e.g., sample every two steps
    ts =        0,      1,      2,      3,      4,      5,      6,      7
    t_use =     0,      2,      4,      6,      8,      10,     12,     14
    """
    def __init__(self, model, timestep_map, original_num_timestep):
        super().__init__()
        self.model = model
        self.timestep_map = timestep_map
        self.original_num_timestep = original_num_timestep
        
    def forward(self, x, ts, **kwargs):
        ts = ts.to(x.device)
        map_tensor = torch.tensor(self.timestep_map, dtype=ts.dtype, device=ts.device)
        new_ts = map_tensor[ts.long()]
        return self.model.forward(x, new_ts, **kwargs)
    
    def forward_with_cfg(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, dtype=ts.dtype, device=ts.device)
        new_ts = map_tensor[ts]
        return self.model.forward_with_cfg(x, new_ts, **kwargs)
    
# trainer modules
def cycle(dataloader: DataLoader):
    " infinitely yield data from dataloader "
    while True:
        for data in dataloader:
            yield data
            
def num_to_group(num: int, group_size: int):
    num_group = num // group_size
    remainder = num % group_size
    groups = [group_size] * num_group
    if remainder > 0:
        groups.append(remainder)
    return groups

def _dict_to(d: dict, target: Any):
    " do value.to(target) for every tensor value "
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(target)
    return d

class PLDiffusion1D(pl.LightningModule):
    "Pytorch Lightning wrapper for GaussianDiffusion1D"

    def __init__(
        self,
        # transformer model arguments
        model_config: ModelConfig|None = None,
        train_config: TrainConfig|None = None,
        # data
        trainset: Dataset|None = None,
        valset: Dataset|None = None,
        # diffusion model arguments
        diffusion_config: DiffusionConfig|None = None,
        # num_timestep: int = 1000,
        # model_mean_type: ModelMeanType = ModelMeanType.NOISE,
        # model_variance_type: ModelVarianceType = ModelVarianceType.FIXED_SMALL,
        # loss_type: LossType = LossType.MSE,
        # beta_schedule_type: BetaScheduleType = BetaScheduleType.COSINE,
        # PL optimizer arguments
        # validation arguments
        metrics_factory: Callable|None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['base_model', 'diffusion_model', 'metrics_factory'])
        
        if model_config.model_class != 'gpt2':
            raise NotImplementedError
        base_model = create_backbone(model_config)
        
        self.model_config = model_config
        self.train_config = train_config
        self.diffusion_config = diffusion_config
        
        self.diffusion_model = create_full_diffusion(
            base_model=base_model,
            seq_length=model_config.seq_length,
            ddpm_config=diffusion_config,
        )
        self.ema = EMA(
            self.diffusion_model,
            beta=train_config.ema_decay,
            update_every=train_config.ema_update_every,
            include_online_model=True,
        )
        self.lr = train_config.lr
        self.metrics = metrics_factory() if metrics_factory is not None else None
        self.trainset = trainset
        self.valset = valset
        self.train_config = train_config
        if self.valset is not None and self.metrics is not None:
            self._setup_sampler()
        
    def _setup_sampler(self):
        from energydiff.diffusion.dpm_solver import DPMSolverSampler
        _model = self.ema.ema_model
        sampler = DPMSolverSampler(_model)
        self._sampler = sampler
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=True,
            persistent_workers=True,
        )
    
    def val_dataloader(self):
        if self.valset is None:
            return None
        return torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.train_config.val_batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=True,
            persistent_workers=True,
        )

    # setup function
    def setup(self, stage: str) -> None:
       pass # nothing to setup
   
    # sample function
    def sample_for_validation(self, batch_size, clip_denoised: bool = False, model_kwargs: dict|None = None):
        # dpm solver
        _samples = self._sampler.sample(
            batch_size=batch_size,
            S = self.train_config.val_sample_config.num_sampling_step,
            shape = (self.ema.ema_model.num_in_channel, self.ema.ema_model.seq_length),
            cfg_scale = 1.,
            conditioning = None
        )[0]
        if clip_denoised:
            _samples = torch.clamp(_samples, min=-1., max=1.)
        return _samples
        # -- ancestral --
        # return self.ema.ema_model.sample(batch_size=batch_size,
        #                                    clip_denoised=clip_denoised,
        #                                    model_kwargs=model_kwargs)
   
    # training step
    def training_step(
        self,
        batch: Float[Tensor, "B C L"],
        batch_idx: int,  # -> step counter
    ) -> Float[Tensor, ""]:
        self.train()
        profile, condition = batch
        loss_terms = self.diffusion_model(x_start=profile)
        loss = loss_terms["loss"]
        mse = loss_terms["mse"].item() if "mse" in loss_terms else 0.0
        self.log(
            "Train/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "Train/MSE",
            mse,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update()

    # validation step
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        profile, condition = batch
        # val loss
        loss_terms = self.diffusion_model(x_start=profile)
        loss = loss_terms["loss"]
        mse = loss_terms["mse"].item() if "mse" in loss_terms else 0.0
        self.log(
            "Validation/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Validation/MSE",
            mse,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        if self.metrics is None:
            return
        
        # Each GPU gets a portion of the real data due to DDP
        # Generate synthetic samples matching the local batch size
        local_samples = self.sample_for_validation(batch_size=profile.shape[0])
        local_samples = rearrange(local_samples, "b c l -> b (l c)")
        
        # Process target data
        target = rearrange(profile, "b c l -> b (l c)")
        
        # Gather
        gathered_source = self.all_gather(local_samples)
        gathered_target = self.all_gather(target)
        
        print('gathered. update metric state on global zero')
        if self.trainer.is_global_zero:
            source_concat = torch.cat([batch for batch in gathered_source], dim=0)
            target_concat = torch.cat([batch for batch in gathered_target], dim=0)
            self.metrics.update(source_concat, target_concat)
        
        # if self.trainer.is_global_zero:
        #     # if any, first sample
        #     sample = self.sample_for_validation(batch_size=profile.shape[0])
        #     sample = rearrange(sample, "b c l -> b (l c)")
        #     # eval functions
        #     target = rearrange(profile, 'b c l -> b (l c)')
        #     self.metrics.update(source=sample, target=target)

    def on_validation_epoch_end(self):
        if self.metrics is None:
            return
        # print(self.local_rank, 'validation epoch end')
        metric_results = self.metrics.compute() # non-zero process results are invalid and not used
        if self.trainer.is_global_zero:
            for k, v in metric_results.items():
                self.log(f'Validation/{k}', v, on_step=False, on_epoch=True, sync_dist=False) # sync_dist = True is theoretically correct? but it freezes the process.
        self.metrics.reset()
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        # print(self.local_rank, 'validation epoch end done')

    # test step
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Validation step is not implemented yet.")

    # configure optimizer
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.diffusion_model.parameters(), lr=self.lr
        )  # just online model
        return optimizer

class Trainer1D():
    def __init__(
        self, 
        diffusion_model: GaussianDiffusion1D|RectifiedFlow,
        spaced_diffusion_model: SpacedDiffusion1D,
        dataset: Dataset,
        dpm_solver_sample: bool = False,
        train_batch_size: int = 16,
        gradient_accumulate_every: int = 1,
        train_lr: float = 1e-4,
        num_train_step: int = 100000,
        ema_update_every: int = 10,
        ema_decay: float = 0.995,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        save_and_sample_every: int = 1000,
        val_every: int|None = None,
        max_val_batch: int = 10,
        num_sample: int = 25, # num_sample at milestone
        result_folder: str = './results',
        amp = False, # accelerator mixed precision
        mixed_precision_type: str = 'fp16',
        split_batches = True,
        num_dataloader_workers = cpu_count(),
        val_dataset: Dataset|None = None,
        sample_model_kwargs: dict[str, Any] = {},
        post_transforms: Iterable[Callable] = [],
        val_batch_size: int|None = None,
        pre_eval_fn: Callable|None = None,
        dict_eval_fn: dict[str, Callable] = {},
        log_wandb: bool = False,
        log_id: str = 'train-diffusion',
        distribute_ema: bool = False, 
    ):
        super().__init__()
        
        # accelerator
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no',
            cpu=not torch.cuda.is_available()
        )
        self.distribute_ema = distribute_ema
        
        # model
        self.model = diffusion_model
        self.num_in_channel = diffusion_model.num_in_channel
        self.seq_length = diffusion_model.seq_length
        
        # sampling and training hyperparameters
        # assert math.sqrt(num_sample ** 2) == num_sample, 'num_sample must be a square number' # No need at all. 
        self.num_sample = num_sample
        self.save_and_sample_every = save_and_sample_every
        self.val_every = default(val_every, save_and_sample_every)
        self.max_val_batch = max_val_batch
        
        self.batch_size = train_batch_size
        self.val_batch_size = default(val_batch_size, train_batch_size)
        self.gradient_accumulate_every = gradient_accumulate_every
        self.num_train_step = num_train_step
        
        # dataset and dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_dataloader_workers,
            persistent_workers=True,
        )
        dataloader = self.accelerator.prepare(dataloader)
        self.dataloader = cycle(dataloader)
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.val_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_dataloader_workers,
            persistent_workers=True,
        )
        # self.val_loader = self.accelerator.prepare(self.val_loader) # do not prepare val_loader so that the validation is easier. 
        self.sample_model_kwargs = _dict_to(sample_model_kwargs, self.device)
        self.post_transforms = post_transforms
        if post_transforms:
            assert all([callable(transform) for transform in post_transforms])
        
        self.pre_eval_fn = pre_eval_fn
        self.dict_eval_fn = dict_eval_fn
        
        # optimizer
        self.optimizer = AdamW(
            diffusion_model.parameters(),
            lr = train_lr,
            betas = adam_betas
        )
        
        if isinstance(self.model, RectifiedFlow):
            dpm_solver_sample = False # not applicable for RectifiedFlow
        self.dpm_solver_sample = dpm_solver_sample
        # logging periodically
        if self.accelerator.is_main_process or self.distribute_ema:
            if not self.dpm_solver_sample:
                self.ema = EMA(spaced_diffusion_model, beta=ema_decay, update_every=ema_update_every, ignore_names={'post_transforms', 'pre_transforms'})
            else:
                self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every, ignore_names={'post_transforms', 'pre_transforms'})
                # spaced_diffusion_model is in fact not used. 
            self.ema.to(self.device)
            
            # make sample function (only needed for main process, or if distribute_ema is True)
            if not self.dpm_solver_sample:
                self.sample_fn = partial(self.ema.ema_model.sample,
                                    clip_denoised=True,
                                    model_kwargs=self.sample_model_kwargs,
                                    post_transforms=self.post_transforms,
                ) # missing: batch_size
            else:
                # NOTE make sure use full steps. 
                from energydiff.diffusion.dpm_solver import DPMSolverSampler # lazy import, avoid circular import (shabiyiyang python)
                self.dpm_solver_sampler = DPMSolverSampler(self.ema.ema_model)
                _fn = partial(self.dpm_solver_sampler.sample,
                                S = spaced_diffusion_model.num_timestep,
                                shape = (self.num_in_channel, self.seq_length),
                                cfg_scale = self.sample_model_kwargs.get('cfg_scale', 1.),
                                conditioning = self.sample_model_kwargs.get('c', None),
                                ) # missing: batch_size
                _sample_fn = lambda batch_size: _fn(batch_size=batch_size)[0]
                clip_denoised = True
                if clip_denoised:
                    self.sample_fn = lambda batch_size: torch.clamp(_sample_fn(batch_size), min=-1., max=1.)
                else:
                    self.sample_fn = _sample_fn
               
        self.result_folder = Path(result_folder)
        self.result_folder.mkdir(exist_ok=True)
        
        self.log_wandb = log_wandb
        self.log_id = log_id
        
        # step counter
        self.step = 0
        
        # prepare model and optimizer
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        
    @classmethod
    def from_config(
        cls,
        config,
        diffusion_model,
        spaced_diffusion_model,
        dataset,
        val_dataset,
        log_id,
        **kwargs,
    ):
        "config: TrainConfig"
        return cls(
            diffusion_model = diffusion_model,
            spaced_diffusion_model = spaced_diffusion_model,
            dataset = dataset,
            dpm_solver_sample = config.val_sample_config.dpm_solver_sample,
            train_batch_size = config.batch_size,
            gradient_accumulate_every = config.gradient_accumulate_every,
            train_lr = config.lr,
            num_train_step = config.num_train_step,
            ema_update_every = config.ema_update_every,
            ema_decay = config.ema_decay,
            adam_betas = config.adam_betas,
            save_and_sample_every = config.save_and_sample_every,
            val_every = config.val_every,
            num_sample = config.val_sample_config.num_sample,
            amp = config.amp,
            mixed_precision_type = config.mixed_precision_type,
            split_batches = config.split_batches,
            val_dataset = val_dataset,
            val_batch_size = config.val_batch_size,
            log_id = log_id,
            **kwargs,
        )
        
        
    @property
    def device(self):
        return self.accelerator.device
    
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        to_save = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'optimizer': self.optimizer.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() \
                if self.accelerator.scaler is not None else None, # scaler will be None if accelerator does not have a scaler
            'version': __version__
        }
        
        torch.save(to_save, os.path.join(self.result_folder, self.log_id+'-'+f'model-{milestone}.pt'))
    
    def load_model(self, milestone, directory = None, log_id = None, ignore_init_final = False):
        "load from a milestone, but only load the model & ema model"
        accelerator = self.accelerator
        device = accelerator.device
        
        if directory is None or log_id is None:
            path = os.path.join(self.result_folder, self.log_id+'-'+f'model-{milestone}.pt')
        else:
            path = os.path.join(directory, log_id+'-'+f'model-{milestone}.pt')
            
        loaded = torch.load(
            path,
            map_location=device
        )
        
        def _remove_init_final_params(model_state_dict):
            _filtered_dict = {}
            for key, value in model_state_dict.items():
                if isinstance(value, type(model_state_dict)):
                    return _remove_init_final_params(value)
                else:
                    if 'init' not in key and 'final' not in key:
                        _filtered_dict[key] = value
                    else:
                        pass
            
            return _filtered_dict
        
        if ignore_init_final:
            loaded['model'] = _remove_init_final_params(loaded['model'])
            loaded['ema'] = _remove_init_final_params(loaded['ema'])
        
        model = self.accelerator.unwrap_model(self.model) # TODO: what is?
        model.load_state_dict(loaded['model'], strict=False)
        
        # self.step = loaded['step']
        # self.optimizer.load_state_dict(loaded['optimizer'])
        if self.accelerator.is_main_process or self.distribute_ema: # Normally we only keep ema model in main process
            self.ema.load_state_dict(loaded['ema'], strict=False)
        if self.accelerator.scaler is not None and loaded['scaler'] is not None:
            self.accelerator.scaler.load_state_dict(loaded['scaler'])
        
        if 'version' in loaded:
            print(f"loading from version {loaded['version']}")
    
    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device
        
        path = os.path.join(self.result_folder, self.log_id+'-'+f'model-{milestone}.pt')
            
        loaded = torch.load(
            path,
            map_location=device
        )
        
        model = self.accelerator.unwrap_model(self.model) # TODO: what is?
        model.load_state_dict(loaded['model'])
        
        self.step = loaded['step']
        self.optimizer.load_state_dict(loaded['optimizer'])
        if self.accelerator.is_main_process or self.distribute_ema: # Normally we only keep ema model in main process
            self.ema.load_state_dict(loaded['ema'])
        if self.accelerator.scaler is not None and loaded['scaler'] is not None:
            self.accelerator.scaler.load_state_dict(loaded['scaler'])
        
        if 'version' in loaded:
            print(f"loading from version {loaded['version']}")
            
    @staticmethod
    def validate(generated_sample: Data1D, val_loader: DataLoader, dict_eval_fn: dict[str, Callable], pre_eval_fn: Callable|None = None, max_val_batch: int = 10):
        """ compare generated_sample with batches of val_loader and average the results 
        too many batches -> long evaluation time
        """
        if len(dict_eval_fn) == 0:
            return {}

        result = {fn_name: 0. for fn_name in dict_eval_fn}
        evaludated_batch = 0
        count_sample = 0
        if pre_eval_fn is not None:
            generated_sample = pre_eval_fn(generated_sample)
        for val_target, _ in val_loader:
            if evaludated_batch >= max_val_batch:
                break
            val_target = val_target.to(generated_sample.device)
            count_sample += val_target.shape[0]
            if pre_eval_fn is not None:
                val_target = pre_eval_fn(val_target)
            
            for fn_name, eval_fn in dict_eval_fn.items():
                result[fn_name] += eval_fn(generated_sample, val_target) * val_target.shape[0]
            evaludated_batch += 1
        
        for fn_name in dict_eval_fn:
            result[fn_name] = result[fn_name] / count_sample
        return result
            
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        
        with tqdm(
            initial = self.step,
            total = self.num_train_step,
            disable = not accelerator.is_main_process,
        ) as pbar:
            while self.step < self.num_train_step:
                total_loss = 0.
                total_mse = 0.
                total_vb = 0.
                
                for _ in range(self.gradient_accumulate_every):
                    data, cond = next(self.dataloader)
                    data, cond = data.to(device), cond.to(device)
                    
                    model_kwargs = {
                        'c': cond,
                        'cfg_scale': 1.,
                    }
                    with self.accelerator.autocast():
                        loss_terms = self.model(data, model_kwargs=model_kwargs)
                        loss = loss_terms['loss']
                        mse = loss_terms['mse'].item() if 'mse' in loss_terms else 0.
                        vb = loss_terms['vb'].item() if 'vb' in loss_terms else 0.
                        
                        loss = loss/self.gradient_accumulate_every # -> mean loss
                        vb, mse = vb/self.gradient_accumulate_every, mse/self.gradient_accumulate_every
                        
                        total_loss += loss.item()
                        total_mse += mse
                        total_vb += vb
                    
                    self.accelerator.backward(loss) 
                    
                accelerator.clip_grad_norm_(self.model.parameters(), 1.)
                pbar.set_description(f'loss: {total_loss:.4f}')
                
                if self.log_wandb and self.accelerator.is_main_process:
                    try:
                        wandb.log({'Train': {
                                'Train Loss': total_loss,
                                'Train MSE': total_mse,
                                'Train VB': total_vb,
                                'Train Step': self.step,
                        }}, step=self.step)
                    except:
                        self.log_wandb = False
                        print('wandb not available, logging disabled')
                
                accelerator.wait_for_everyone()
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                accelerator.wait_for_everyone()
                
                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()
                    
                    if self.step != 0 and (self.step % self.val_every == 0 or self.step % self.save_and_sample_every == 0):
                        self.ema.ema_model.eval()
                        
                        with torch.no_grad():
                            # with accelerator.autocast():
                            list_batch_size = num_to_group(self.num_sample, self.val_batch_size)
                            list_all_sample = list(map(
                                lambda n: self.sample_fn(batch_size=n),
                                list_batch_size
                            ))
                            
                        all_sample = torch.cat(list_all_sample, dim=0)
                        val_result = self.validate(
                            all_sample,
                            val_loader=self.val_loader,
                            dict_eval_fn=self.dict_eval_fn,
                            pre_eval_fn=self.pre_eval_fn,
                            max_val_batch=self.max_val_batch,
                        )
                        if self.log_wandb:
                            wandb.log({'Validation': val_result}, step=self.step)
                            # plt.plot(all_sample.squeeze(1).cpu().numpy())
                            wandb.log({'Samples': all_sample.squeeze(1)}, step=self.step)
                            mid_channel = all_sample.shape[1] // 2
                            wandb.log({'Source Histogram': wandb.Histogram(all_sample[:,mid_channel,:].cpu().flatten(), num_bins=200)}, step=self.step)
                            _all_target = self.val_loader.dataset.tensor # not a very good way to get target
                            wandb.log({'Target Histogram': wandb.Histogram(_all_target[:,mid_channel,:].cpu().flatten(), num_bins=200)}, step=self.step)
                        
                        if self.step != 0 and self.step % self.save_and_sample_every == 0:
                            milestone = self.step // self.save_and_sample_every
                            # save sample
                            torch.save(all_sample,
                                    os.path.join(self.result_folder, self.log_id+'-'+f'sample-{milestone}.pt'))
                            # save states
                            self.save(milestone)
                        
                pbar.update(1)
                
        accelerator.print('training complete')


def collect_init_arguments(class_type):
    """
    Collects the names of arguments needed by the __init__ method of a given class.

    Parameters:
        class_type (type): The class type for which to collect the argument names.

    Returns:
        set: A set containing the names of arguments.
    """
    init_arguments = set()
    if hasattr(class_type, '__init__'):
        init_signature = inspect.signature(class_type.__init__)
        for param_name, param in init_signature.parameters.items():
            if param_name != 'self':  # Exclude the 'self' parameter
                init_arguments.add(param_name)

    return init_arguments

class GenericConfig:
    name_model = ''
    keys = set()
    dict_renaming = {}
    def __init__(self, **kwargs):
        self.config_dict = {}
        for key, value in kwargs.items():
            if key in self.keys:
                self.config_dict[key] = value
            
    def __repr__(self):
        return str(self.config_dict)
    
    def items(self):
        return self.config_dict.items()

class UNet1DConfig(GenericConfig):
    keys = collect_init_arguments(Unet1D)
    
class Transformer1DConfig(GenericConfig):
    keys = collect_init_arguments(Transformer1D) 

class MLP1DConfig(GenericConfig):
    keys = collect_init_arguments(DenoisingMLP1D)

def create_backbone(model_config: ModelConfig, **kwargs):
    # Step 1: Setup backbone
    model_class = model_config.model_class
    init_kwargs = vars(model_config).copy()
    init_kwargs.update(kwargs)
    if model_class == 'unet':
        Model = Unet1D
        config = UNet1DConfig(**init_kwargs)
    # transformer (pytorch implementation) is deprecated
    # elif model_class == 'transformer':
    #     Model = Transformer1D
    #     config = Transformer1DConfig(**init_kwargs, type_transformer='transformer')
    elif model_class == 'gpt2':
        Model = Transformer1D
        # config = Transformer1DConfig(**init_kwargs, type_transformer='gpt2')
        config = Transformer1DConfig(**init_kwargs)
    elif model_class == 'mlp':
        Model = DenoisingMLP1D
        config = MLP1DConfig(**init_kwargs)
    else:
        raise ValueError(f"Unsupported model class {model_class}.")
    
    backbone_model = Model(**config.config_dict)
    # try:
    #     backbone_model = Model(**config.config_dict)
    # except:
    #     raise ValueError(f"Error when initializing {model_class} with config {config}.")
    
    return backbone_model

def create_full_diffusion(
    base_model: torch.nn.Module,
    seq_length: int,
    ddpm_config: DDPMConfig,
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
    :param num_timestep: the number of diffusion steps (=num sampling step)
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
        import warnings
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
        
    num_sampling_timestep = ddpm_config.num_diffusion_step
    return SpacedDiffusion1D(
        use_timesteps=space_timesteps(ddpm_config.num_diffusion_step, [num_sampling_timestep]),
        base_model=base_model,
        seq_length=seq_length,
        num_timestep=ddpm_config.num_diffusion_step,
        num_sampling_timestep=ddpm_config.num_diffusion_step,
        model_mean_type=prediction_type,
        model_variance_type=model_var_type,
        loss_type=loss_type,
        beta_schedule_type=beta_schedule,
        ddim_sampling_eta=ddim_sampling_eta,
        auto_normalize=auto_normalize,
        loss_only_central_channel=loss_only_central_channel,
        fft_loss_weight=fft_loss_weight,
    )