"""
Implementation according to Rectified Flow paper + Stable Diffuion 3 paper
"""
import enum
import math
from typing import Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce, einsum
from tqdm import tqdm

from .version import __version__
from .utils import default, identify
from .typing import Float, Int

class RFPredictionType(enum.Enum):
    """
    Enum for the type of prediction to use in the rectified flow model
    """
    VELOCITY = enum.auto()
    NOISE = enum.auto()
    
class RFScheduleType(enum.Enum):
    LOGIT_NORMAL = enum.auto()
    COSMAP = enum.auto()
    UNIFORM = enum.auto()

def get_logit_normal_sampler(loc: float, scale: float) -> callable:
    def sampler(batch_size, device=None) -> Float[Tensor, 'batch_size, ']:
        dist = torch.distributions.Normal(loc=loc, scale=scale)
        samples = dist.sample((batch_size,))
        samples = torch.sigmoid(samples)
        if device is not None:
            samples = samples.to(device)
        return samples
    
    return sampler

def get_uniform_sampler(l) -> callable:
    def sampler(batch_size, device=None) -> Float[Tensor, 'batch_size, ']:
        samples = torch.rand(batch_size) # [0,1)
        if device is not None:
            samples = samples.to(device)
        return samples
    
    return sampler

def get_cosmap_schedule():
    raise NotImplementedError

class RectifiedFlow(nn.Module):
    r"""
    **RF model**
    t continous from 0 to 1
    forward process:
        - x_t = (1-t) * x_0 + t * epsilon
    reverse process:
        - x_0 = int_{0}^{1} -v(x_t, t)dt
        
    Args:
        - base_model (nn.Module): 
            forward(x_t, t, c, ...) -> epsilon_{t}
        - rescale_t (bool): rescale t from [0,1] to [0,1000]
    """
    def __init__(
        self,
        base_model: nn.Module,
        seq_length: int,
        num_discretization_step: int = 1000,
        prediction_type: RFPredictionType = RFPredictionType.VELOCITY,
        schedule_type: RFScheduleType = RFScheduleType.LOGIT_NORMAL,
        rescale_t: bool = False,
    ):
        super().__init__()
        self.model = base_model
        self.num_in_channel = base_model.num_in_channel
        self.conditioning = base_model.conditioning
        
        self.num_sampling_timestep = num_discretization_step
        
        self.seq_length = seq_length
        self.prediction_type = prediction_type
        self.schedule_type = schedule_type
        assert prediction_type in RFPredictionType, "Invalid prediction type"
        assert schedule_type in RFScheduleType, "Invalid schedule type"

        self.rescale_t = rescale_t
        if schedule_type == RFScheduleType.LOGIT_NORMAL:
            self.t_sampler = get_logit_normal_sampler(loc=0., scale=1.)
        elif schedule_type == RFScheduleType.UNIFORM:
            self.t_sampler = get_uniform_sampler(1.)
        else:
            self.t_sampler = get_logit_normal_sampler(loc=0., scale=1.)
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    def model_fn_cfg(self, x_t, t, **model_kwargs):
        # rescale t from [0,1] to [0,1000]
        if self.rescale_t:
            t = t * 1000.
        return self.model.forward_with_cfg(x_t, t, **model_kwargs)
    
    def model_fn(self, x_t, t, **model_kwargs):
        return self.model.forward(x_t, t, **model_kwargs)
        
    def standard_loss_weight(self, t: Int[Tensor, 'batch_size,']) -> Float[Tensor, 'batch_size,']:
        t = t.float()
        epsilon = 1e-7
        # loss_weight = -(t-1)/((1-t)**2 * t + epsilon)
        loss_weight = 1/((1-t)**2+epsilon)
        
        if self.prediction_type == RFPredictionType.NOISE:
            return loss_weight
        elif self.prediction_type == RFPredictionType.VELOCITY:
            return 1.0
        else:
            raise NotImplementedError
        
    def calc_v_from_pred_noise(self, epsilon_t, z_t, t):
        """
        calculate the velocity from the predicted noise
        """
        t = t.clamp(0.0001, 0.9999)
        t = rearrange(t, 'b -> b () ()')
        return (epsilon_t - z_t) / (1 - t)
        
    @torch.no_grad()
    def reverse_sample_progressive(
        self,
        batch_size: int,
        noise: Float[Tensor, 'batch_size, num_in_channel, seq']|None=None,
        model_kwargs: dict|None = None,
        method: str = 'naive',
    ):
        """
        reverse sampling process, x0 <- x1
        t_i: from 0.999 to 0.001
        z_{t_{i+1}} = z_{t_{i+1}} - v * delta_t
        
        `method`:
            - 'naive': use the naive method to integrate the velocity
            - 'RK45': call scipy.integrate.solve_ivp to solve the ODE, using RK45 method
        """
        method = method.upper()
        assert method in ['NAIVE', 'EULER', 'RK45', 'ABM', 'AB'], "Invalid method"
        x_t = default(noise, torch.randn(batch_size, self.num_in_channel, self.seq_length)).to(self.device)
        model_kwargs = default(model_kwargs, {})
        all_t = torch.linspace(0.999, 0.001, self.num_sampling_timestep+1)
        # step_div = [
        #     # math.floor(self.num_sampling_timestep*0.4),
        #     # math.floor(self.num_sampling_timestep*0.2),
        # ]
        # step_div.append(self.num_sampling_timestep - sum(step_div))
        # start_t = torch.linspace(0.999, 0.7, step_div[0]+1) # incl. start
        # mid_t = torch.linspace(0.7, 0.3, step_div[1]+1) # incl. start
        # end_t = torch.linspace(0.3, 0.001, step_div[2]+1) # incl. start
        # all_t = torch.cat([start_t, mid_t[1:], end_t[1:]]) # T+1
        all_t = all_t[1:] # remove t=0.999
        dt = (all_t[-2] - all_t[-1]).item()
        
        if self.num_sampling_timestep <= 2:
            method = 'EULER'
        if method in ['NAIVE', 'EULER']:
            for idx, t_i in enumerate(all_t):
                # reshape t
                if idx >= 1:
                    dt = all_t[idx-1] - all_t[idx]
                else:
                    dt = all_t[0] - all_t[1]
                t = t_i * torch.ones(batch_size, device=x_t.device)
                # get prediction of velocity
                model_pred = self.model_fn_cfg(x_t, t, **model_kwargs)
                if self.prediction_type == RFPredictionType.NOISE:
                    v = self.calc_v_from_pred_noise(model_pred, x_t, t)
                else:
                    v = model_pred
                # integrate
                x_t = x_t - v * dt
                yield x_t
        elif method in ['ABM', 'AB']:
            # initial steps
            t_init_1 = all_t[0] * torch.ones(batch_size, device=x_t.device)
            x_init_1 = x_t
            model_pred_init_1 = self.model_fn_cfg(x_init_1, t_init_1, **model_kwargs)
            if self.prediction_type == RFPredictionType.NOISE:
                v_init_1 = -self.calc_v_from_pred_noise(model_pred_init_1, x_init_1, t_init_1)
            else:
                v_init_1 = -model_pred_init_1
            x_init_2 = x_init_1 + v_init_1 * dt
            yield x_init_2
            x_curr = x_init_2
            v_prev = v_init_1
            for t_i in all_t[1:]:
                t_curr = t_i * torch.ones(batch_size, device=x_t.device)
                model_pred_curr = self.model_fn_cfg(x_curr, t_curr, **model_kwargs)
                if self.prediction_type == RFPredictionType.NOISE:
                    v_curr = -self.calc_v_from_pred_noise(model_pred_curr, x_curr, t_curr)
                else:
                    v_curr = -model_pred_curr
                x_pred = abm_pred(x_curr, v_curr, v_prev, dt)
                if method == 'ABM':
                    model_pred_pred = self.model_fn_cfg(x_pred, t_curr, **model_kwargs)
                    if self.prediction_type == RFPredictionType.NOISE:
                        v_pred = -self.calc_v_from_pred_noise(model_pred_pred, x_pred, t_curr+dt)
                    else:
                        v_pred = -model_pred_pred
                    x_next = abm_correct(x_curr, v_curr, v_pred, dt)
                else:
                    x_next = x_pred
                yield x_next
                v_prev = v_curr
                x_curr = x_next
        else:
            from scipy.integrate import solve_ivp
            def f(t: Int, x: Float[np.array, 'dim']) -> Float[np.array, 'dim']:
                device = self.device
                print(f'Solving for t = {t:.4f}')
                x = torch.from_numpy(x).reshape(batch_size, self.num_in_channel, self.seq_length).to(device).float()
                t = t * torch.ones(batch_size, device=x.device)
                model_pred = self.model_fn_cfg(x, t, **model_kwargs)
                    # shape: (batch, num_in_channel, seq)
                if self.prediction_type == RFPredictionType.NOISE:
                    v = self.calc_v_from_pred_noise(model_pred, x, t)
                else:
                    v = model_pred
                return v.reshape(-1).cpu().numpy().astype(np.float64)
            
            res = solve_ivp(f, t_span=(0.9999, 0.0001), y0=x_t.reshape(-1).cpu().numpy(), 
                            method=method, 
                            t_eval=all_t.cpu().numpy(),
                            atol=1e-3, rtol=1e-1) # t_eval=all_t.cpu().numpy()
            t, x = res.t, res.y
            x = torch.from_numpy(x).reshape(batch_size, self.num_in_channel, self.seq_length, -1).to(self.device)
            for idx in range(x.shape[-1]):
                yield x[..., idx]
        
    @torch.no_grad()
    def reverse_sample_loop(
        self,
        batch_size: int,
        noise: Float[Tensor, 'batch_size, num_in_channel, seq']|None = None,
        model_kwargs: dict|None = None,
        method: str = 'naive',
    ):
        """
        reverse sampling process, x0 <- x1
        """
        
        sample_prev = None
        sample_init = None
        dx_mag_cum = 0
        dx = []
        vt_mag = []
        noise = default(noise, torch.randn(batch_size, self.num_in_channel, self.seq_length)).to(self.device)
        sample_prev = noise
        sample_init = noise
        all_sample = [sample_init]
        pbar = tqdm(self.reverse_sample_progressive(batch_size, noise, model_kwargs, method=method), desc='sampling loop time step', total=self.num_sampling_timestep)
        for sample in pbar:
            if sample_prev is not None:
                # vector
                step_dx = (sample - sample_prev) # (batch, num_in_channel, seq)
                dx.append(step_dx)
                # magnitude
                vt_mag.append(step_dx.flatten(start_dim=1).norm(dim=1).mean().item()*self.num_sampling_timestep)
                dx_mag_cum += vt_mag[-1]/self.num_sampling_timestep
                pbar.set_description(f"Step velocity: {vt_mag[-1]:.4f}")
            elif sample_init is None:
                sample_init = sample
            else:
                pass
            sample_prev = sample # (batch, num_in_channel, seq)
            all_sample.append(sample)
            
        import matplotlib.pyplot as plt
        # plot step velocity magnitude 
        plt.plot(vt_mag)
        plt.savefig('step_velocity.png')
        plt.close()
        
        # calculate velocity matching
        dx = torch.stack(dx, dim=0).flatten(2) # (T, batch, num_in_channel*seq)
        vt = dx*self.num_sampling_timestep
        displacement = (sample - sample_init).flatten(1).unsqueeze(0) # (1, batch, num_in_channel*seq)
        velocity_matching = einsum(vt, displacement, 't1 b d, t2 b d -> b t1 t2')
        velocity_matching = rearrange(velocity_matching, 'b t 1 -> t b')

        vt_norm = vt.norm(dim=-1) # (T, batch)
        displacement_norm = displacement.norm(dim=-1) # (1, batch)
        velocity_matching = velocity_matching / (1e-7 + vt_norm * displacement_norm)
        velocity_matching = velocity_matching.mean(dim=1)
            # shape: (T, )
        
        plt.plot(velocity_matching.cpu().numpy())
        plt.savefig('velocity_matching.png')
        plt.close()
        
        # calculate distance with ideal trajectory
        all_sample = torch.stack(all_sample, dim=0) # (T, batch, num_in_channel, seq)
        all_sample = all_sample.flatten(2) # (T, batch, num_in_channel*seq)
        _t = torch.linspace(0.999, 0.001, all_sample.shape[0]).to(all_sample.device)
        _t = rearrange(_t, 't -> t () ()')
        dist_with_dest = (all_sample - _t*all_sample[0] - (1-_t)*all_sample[-1]).norm(dim=-1).mean(dim=1) # (T, )
        plt.plot(dist_with_dest.cpu().numpy())
        plt.savefig('dist_with_ideal_traj.png')
        plt.close()
        
        direct_distance = displacement_norm.mean().item()
        print(f'Path length: {dx_mag_cum:.4f}, distance: {direct_distance:.4f}')
        
        return sample
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        clip_denoised: bool = False,
        model_kwargs: dict|None = None,
        post_transforms: Any|None = None,
        method: str = 'euler',
    ):
        """
        :param clip_denoised: bool, NOT USED
        :param model_kwargs: dict, additional kwargs for the model
        :param post_transform: callable, NOT USED
        """
        return self.reverse_sample_loop(batch_size, model_kwargs=model_kwargs, method=method)
        
    def forward_sample(
        self, 
        x_0: Float[Tensor, 'batch_size, num_in_channel, seq'],
        t: Int[Tensor, 'batch_size,'],
        noise: Float[Tensor, 'batch_size, num_in_channel, seq']|None = None,
    ) -> tuple[Float[Tensor, 'batch_size, num_in_channel, seq']]:
        """
        Forward sampling process
        x_t = (1-t) * x_0 + t * epsilon
        
        Return
            - x_t: sampled x_t
            - noise: noise added to x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0).to(x_0.device)
        t = rearrange(t, 'b -> b () ()')
        x_t = (1-t) * x_0 + t * noise
            # t should be broadcasted to the shape of x_0
        return x_t, noise
    
    def train_losses(self, x_0, t, noise=None, model_kwargs=None)\
        -> dict:
        """
        given
            - x_0: training sample
            - t: sampled time step (0,1)
            - [optional] noise: sampled noise (default: None)
            - [optional] model_kwargs: additional kwargs for the model (default: None)
            
        calculate the loss terms, return
            - loss_terms: dict of loss terms
        """
        model_kwargs = default(model_kwargs, {})
        if noise is not None:
            if x_0.shape != noise.shape:
                noise = None
        x_t, noise = self.forward_sample(x_0, t, noise)
        loss_weight = self.standard_loss_weight(t) # (batch_size,)
        
        model_pred = self.model_fn_cfg(x_t, t, **model_kwargs)
        
        loss_terms = {}        
        # mse loss
        if self.prediction_type == RFPredictionType.NOISE:
            _mse = reduce(
                F.mse_loss(noise, model_pred, reduction='none'), 
                'b c l -> b', 'mean'
            )
            # maybe here we can log the distribution of mse over t
            loss_terms['raw_mse'] = _mse.detach()
            loss_terms['mse'] = (_mse * loss_weight)
        else:
            _mse = reduce(
                F.mse_loss(noise-x_0, model_pred, reduction='none'),
                'b c l -> b', 'mean'
            )
            loss_terms['raw_mse'] = _mse.detach()
            loss_terms['mse'] = (_mse * loss_weight)
                # loss_weight = 1.0 for velocity prediction
                    
        loss_terms['loss'] = loss_terms['mse']
        
        # NOTE the loss terms' shape is NOT reduced to scalar
        return loss_terms
    
    def forward(self, x_0, noise=None, model_kwargs=None):
        """
        given
            - x_0: training sample
            - [optional] noise: sampled noise (default: None)
            - [optional] model_kwargs: additional kwargs for the model (default: None)
            
        calculate the loss terms, return
            - loss_terms: dict of loss terms
        """
        batch, channel, seq = x_0.shape
        device = x_0.device
        assert seq == self.seq_length, "Invalid sequence length"
        assert channel == self.num_in_channel, "Invalid number of input channel"
        t = self.t_sampler(x_0.shape[0], device=device)
        
        loss_terms = self.train_losses(x_0, t, noise, model_kwargs)
            # NOTE shape not reduced to scalar
        
        for k, v in loss_terms.items():
            loss_terms[k] = v.mean()
        
        return loss_terms
            
def abm_pred(x_t, f_t, f_t_1, dt):
    r"(x_t, f_t, f_{t-1}) -> x_{t+1}_pred"
    return x_t + 1.5*dt*f_t - 0.5*dt*f_t_1

def abm_correct(x_t, f_t, f_pred, dt):
    r"(x_t, f_t, f_pred) -> x_{t+1}"
    return x_t + 0.5*dt*(f_t + f_pred)

        