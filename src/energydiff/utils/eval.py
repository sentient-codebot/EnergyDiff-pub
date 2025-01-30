import os
import pickle
from abc import ABC
from typing import Annotated as Float, Any
from typing import Annotated as Long
from typing import Annotated as Int
from typing import Callable, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.signal import windows
from torch import Tensor
from numpy import ndarray
from einops import rearrange, einsum, reduce
from scipy import linalg
from scipy.stats import wasserstein_distance, ks_2samp
import matplotlib.pyplot as plt
import torchmetrics

from .configuration import DataConfig

def dist_daily_consumption(data, hist_bins):
    """calculate the distribution of daily total consumption
    
    Args:
        data: torch.tensor, shape (N, T)
    
    Return:
        dist: torch.tensor, shape (N, )
    """
    dist = torch.sum(data, dim=1)
    values, bins  = torch.histogram(dist, bins=hist_bins)
    values = values
    return values

def source_mean(source, target):
    return source.mean()

def target_mean(source, target):
    return target.mean()

def source_std(source, target):
    return source.std()

def target_std(source, target):
    return target.std()

def eval_fn(original_data: torch.Tensor, generated_data: torch.Tensor) -> dict:
    """evaluate model 
    
    Calculate multiple metrics at a time. Metrics:
        - hist of daily total consumption
        - autocorrelation function (ACF). torch.corrcoef
        - correlation between two consecutive points, e.g. 17:00 and 17:15
        - marginal hist of each hour
    """
    metrics_dict = {
        'common': {},
        'original': {},
        'generated': {}
    }
    
    total_min = torch.min(generated_data.min(), original_data.min())
    total_max = torch.max(generated_data.max(), original_data.max())*1.1
    hist_bins = torch.linspace(total_min, total_max, 100)
    metrics_dict['common']['hist_bins'] = hist_bins
    
    ori_hist_count_daily_consumption = dist_daily_consumption(original_data, hist_bins)
    gen_hist_count_daily_consumption = dist_daily_consumption(generated_data, hist_bins)
    metrics_dict['original']['hist_count_daily_consumption'] = ori_hist_count_daily_consumption
    metrics_dict['generated']['hist_count_daily_consumption'] = gen_hist_count_daily_consumption
    ...
    
    return metrics_dict

def plot_hist(ax, hist_bins, hist_counts) -> plt.Axes:
    x = (hist_bins - hist_bins[0]) / (hist_bins[-1] - hist_bins[0])
    ax.bar(x[:-1], hist_counts, tick_label=hist_bins[:-1])
            
    return ax

def estimate_autocorrelation(x: Float[np.ndarray, "batch, sequence"]) -> Float[np.ndarray, "sequence, sequence"]:
    """estimate autocorrelation, we deduct the mean before calculation. """
    assert isinstance(x, np.ndarray)
    x = x - np.mean(x, axis=1, keepdims=True)
    x = x.reshape(x.shape[0], x.shape[1], 1) # shape: [batch, sequence, 1], batched column vectors
    x_T = np.transpose(x, axes=(0, 2, 1)) # shape: [batch, 1, sequence], batched row vectors
    autocorrelation = np.matmul(x, x_T) # shape: [batch, sequence, sequence]
    averaged_autocorrelation = np.mean(autocorrelation, axis=0) # shape: [sequence, sequence]
    
    return averaged_autocorrelation

class _MkMMD_old(nn.Module):
    """ Calculate the multi-maximum mean discrepancy (MK-MMD) 
    
    For future: add linear coefficients to each kernel. 
    Args:
        ...
        coefficient: str, 'ones' or 'auto'. If 'ones', the coefficient of each kernel is 1. 
                            If 'auto', the coefficient of each kernel is from Hamming window.
    
    Forward: 
        source: batch of vectors. torch.tensor, shape (N, T)
        target: batch of vectors. torch.tensor, shape (N, T)
    """
    def __init__(self, kernel_type:str='rbf', kernel_mul:float=2.0, num_kernel:int=5, fix_sigma:float|None=None,
                 coefficient:str='ones'):
        super().__init__()
        assert coefficient in {'auto', 'ones'}
        self.coefficient = coefficient
        self.num_kernel = num_kernel
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type # TODO: currently not used
        
    def gaussian_kernel(self, source, target, kernel_mul=2.0, num_kernel=5, fix_sigma=None, coefficient='ones'):
        assert coefficient in {'auto', 'ones'}
        num_sample_s, num_sample_t = int(source.shape[0]), int(target.shape[0]) # b1, b2
        n_samples = num_sample_s + num_sample_t
        total = torch.cat([source, target], dim=0)

        # Compute pairwise L2 distance between all samples
        total0 = rearrange(total, 'b d -> () b d') # b = b1 + b2
        total1 = rearrange(total, 'b d -> b () d') # b = b1 + b2
        
        L2_squared = ((total0 - total1) ** 2).sum(dim=2) # should zero on the diagonal. calculate l2 squared. shape: [n_samples, n_samples]

        # Compute bandwidth
        if fix_sigma is not None:
            if not isinstance(fix_sigma, float):
                fix_sigma = torch.tensor(fix_sigma, device=L2_squared.device)
            bandwidth = fix_sigma
        else:
            # target_l2_squared = L2_squared[num_sample_s:, num_sample_s:] # shape: [b2, b2]
            # source_l2_squared = L2_squared[:num_sample_s, :num_sample_s] # shape: [b1, b1]
            # bandwidth = torch.sum(target_l2_squared.data) / (num_sample_t ** 2 - num_sample_t)
            bandwidth = torch.sum(L2_squared.data) / (n_samples ** 2 - n_samples)
        
        base_bandwidth = bandwidth / (kernel_mul ** (num_kernel // 2)) # base bandwidth
        bandwidth_list = [base_bandwidth * (kernel_mul ** i) for i in range(num_kernel)] # this way the original bandwidth is in the middle of this list
        
        self.register_buffer('bandwidth_list', torch.tensor(bandwidth_list)) # shape: (num_kernel,)
        self.register_buffer('bandwidth', bandwidth) # shape: scalar

        # Compute kernel values
        kernel_val = [torch.exp(-L2_squared / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        kernel_val = torch.stack(kernel_val, dim=0) # shape: [num_kernel, n_samples, n_samples]
        if coefficient == 'ones':
            coef_val = torch.ones(num_kernel, device=kernel_val.device) # shape: [num_kernel]
            coef_val = rearrange(coef_val, 'k -> k () ()') # shape: [num_kernel, 1, 1]
        elif coefficient == 'auto':
            coef_val = windows.hamming(num_kernel, device=kernel_val.device) # shape: [num_kernel]
            coef_val = coef_val / coef_val.sum() * num_kernel # normalize, sum == num_kernel
            coef_val = rearrange(coef_val, 'k -> k () ()') # shape: [num_kernel, 1, 1]
        else:
            raise ValueError(f'coefficient {coefficient} not supported')
        
        weighted_kernel_val = kernel_val * coef_val # shape: [num_kernel, n_samples, n_samples]

        # Sum kernel values
        return weighted_kernel_val.sum(dim=0) # summing over all kernels, shape: [n_samples, n_samples]
    
    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X - f_of_Y
        loss = torch.mean(delta*delta)
        return loss
    
    @staticmethod
    def flatten_to_2d(x):
        if x.ndim == 2:
            return x
        elif x.ndim == 3:
            x = rearrange(x, 'b d l -> b (d l)')
            return x
        else:
            raise ValueError(f'x should be 2d or 3d, but got {x.ndim}')
    
    def forward(self, source, target):
        " source/target: shape (batch, D, sequence). last 2d will be flattened. "
        source, target = map(self.flatten_to_2d, (source, target))
        # compute the mkmmd of two samples of shape [batch, sequence]
        bs_source, bs_target = int(source.shape[0]), int(target.shape[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, num_kernel=self.num_kernel, fix_sigma=self.fix_sigma, 
                                       coefficient=self.coefficient) # shape: [bs_source+bs_target, bs_source+bs_target]
        XX = torch.mean(kernels[:bs_source, :bs_source]) # upper left block, scalar, mean of shape (bs_source, bs_source)
        YY = torch.mean(kernels[bs_source:, bs_source:]) # lower right block, scalar, mean of shape (bs_target, bs_target)
        XY = torch.mean(kernels[:bs_source, bs_source:]) # upper right block, scalar, mean of shape (bs_source, bs_target)
        YX = torch.mean(kernels[bs_source:, :bs_source]) # lower left block, scalar, mean of shape (bs_target, bs_source)
        loss = torch.mean(XX + YY - XY - YX) # could be mean or sum
        return loss
    
class MkMMD(nn.Module):
    """ Calculate the multi-maximum mean discrepancy (MK-MMD) 
    
    For future: add linear coefficients to each kernel. 
    Args:
        ...
        coefficient: str, 'ones' or 'auto'. If 'ones', the coefficient of each kernel is 1. 
                            If 'auto', the coefficient of each kernel is from Hamming window.
    
    Forward: 
        source: batch of vectors. torch.tensor, shape (N, T)
        target: batch of vectors. torch.tensor, shape (N, T)
    """
    def __init__(self, kernel_type:str='rbf', kernel_mul:float=2.0, num_kernel:int=5, fix_sigma:float|None=None,
                 coefficient:str='ones'):
        super().__init__()
        assert coefficient in {'auto', 'ones'}
        self.coefficient = coefficient
        self.num_kernel = num_kernel
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type # TODO: currently not used
        
    def gaussian_kernel(self, source, target, kernel_mul=2.0, num_kernel=5, fix_sigma=None, coefficient='ones'):
        """
        L2_square = | A     | B     |
                    | B^T   | C     |
        """
        assert coefficient in {'auto', 'ones'}
        num_sample_s, num_sample_t = int(source.shape[0]), int(target.shape[0]) # b1, b2
        n_samples = num_sample_s + num_sample_t
        
        # source to source
        L2_squared_A = torch.cdist(source, source, p=2) ** 2 # shape: [b1, b1]
        L2_squared_A = L2_squared_A - torch.diag(torch.diagonal(L2_squared_A)) # remove diagonal
        # target to target
        L2_squared_C = torch.cdist(target, target, p=2) ** 2 # shape: [b2, b2]
        L2_squared_C = L2_squared_C - torch.diag(torch.diagonal(L2_squared_C)) # remove diagonal
        # source to target
        L2_squared_B = torch.cdist(source, target, p=2) ** 2 # shape: [b1, b2]

        # Compute bandwidth
        if fix_sigma is not None:
            if isinstance(fix_sigma, float):
                fix_sigma = torch.tensor(fix_sigma, device=L2_squared_A.device)
            bandwidth = fix_sigma
        else:
            # bandwidth = torch.sum(L2_squared.data) / (n_samples ** 2 - n_samples)
            bandwidth = torch.sum(L2_squared_C) / (num_sample_t ** 2 - num_sample_t)

        base_bandwidth = bandwidth / (kernel_mul ** (num_kernel // 2)) # base bandwidth
        bandwidth_list = [base_bandwidth * (kernel_mul ** i) for i in range(num_kernel)] # this way the original bandwidth is in the middle of this list
        
        self.register_buffer('bandwidth_list', torch.tensor(bandwidth_list)) # shape: (num_kernel,)
        self.register_buffer('bandwidth', bandwidth) # shape: scalar

        # Compute kernel values
        if coefficient == 'ones':
            coef_val = torch.ones(num_kernel, device=L2_squared_A.device) # shape: [num_kernel]
            coef_val = rearrange(coef_val, 'k -> k () ()') # shape: [num_kernel, 1, 1]
        elif coefficient == 'auto':
            coef_val = windows.hamming(num_kernel, device=L2_squared_A.device) # shape: [num_kernel]
            coef_val = coef_val / coef_val.sum() * num_kernel # normalize, sum == num_kernel
            coef_val = rearrange(coef_val, 'k -> k () ()') # shape: [num_kernel, 1, 1]
        else:
            raise ValueError(f'coefficient {coefficient} not supported')
        
        def _calc_kernel_val(L2_squared):
            kernel_val = 0.
            for kernel_idx, bandwidth_temp in enumerate(bandwidth_list):
                kernel_val += torch.exp(-L2_squared / bandwidth_temp).sum() * coef_val[kernel_idx].item() # shape: scalar
            
            return kernel_val
            
        XX = _calc_kernel_val(L2_squared_A) # upper left block, scalar, mean of shape (bs_source, bs_source)
        YY = _calc_kernel_val(L2_squared_C) # lower right block, scalar, mean of shape (bs_target, bs_target)
        XY = _calc_kernel_val(L2_squared_B) # upper right block, scalar, mean of shape (bs_source, bs_target)
        
        # Sum kernel values
        return XX, YY, XY
    
    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X - f_of_Y
        loss = torch.mean(delta*delta)
        return loss
    
    @staticmethod
    def flatten_to_2d(x):
        if x.ndim == 2:
            return x
        elif x.ndim == 3:
            x = rearrange(x, 'b d l -> b (d l)')
            return x
        else:
            raise ValueError(f'x should be 2d or 3d, but got {x.ndim}')
    
    def forward(self, source, target):
        " source/target: shape (batch, D, sequence). last 2d will be flattened. "
        source, target = map(self.flatten_to_2d, (source, target))
        # compute the mkmmd of two samples of shape [batch, sequence]
        bs_source, bs_target = int(source.shape[0]), int(target.shape[0])
        XX, YY, XY = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, num_kernel=self.num_kernel, fix_sigma=self.fix_sigma, 
                                       coefficient=self.coefficient) # shape: [bs_source+bs_target, bs_source+bs_target]
        XX *= 1/(bs_source*max((bs_source-1), 1.)) # upper left block, scalar, mean of shape (bs_source, bs_source)
        YY *= 1/(bs_target*max((bs_target-1), 1.)) # lower right block, scalar, mean of shape (bs_target, bs_target)
        XY *= 2/(bs_source*bs_target)              # upper right block, scalar, mean of shape (bs_source, bs_target)
        loss = XX + YY - XY
        return loss

def only_central_dim(func, dim):
    """decorator for functions that only use the central dimension of a tensor"""
    def wrapper(source, target, *args, **kwargs):
        if source.ndim == 2:
            return func(source, target, *args, **kwargs)
        channel = lambda x: torch.tensor((x.shape[dim]-1)//2, dtype=torch.long, device=x.device)
        _source = source.select(dim, channel(source)).unsqueeze(dim)
        _target = target.select(dim, channel(target)).unsqueeze(dim)
        return func(_source, _target, *args, **kwargs)
    return wrapper

def only_central_dim_decorator(dim):
    def decorator(func):
        return only_central_dim(func, dim)
    return decorator
        

def reshape_3d_2d_pre_fn(func):
    def wrapped(source, target, *args, **kwargs):
        if source.ndim == 3:
            _source = source.squeeze(1)
        if target.ndim == 3:
            _target = target.squeeze(1)
        return func(_source, _target, *args, **kwargs)
    return wrapped
        

# def return_central_dim(dim):
#     " produce a hookable function "
#     def _return_central_dim(module, *args): # module here is the nn.Module instance
#         " a nn.Module.forward_pre_hook "
#         channel = lambda x: torch.tensor((x.shape[dim]-1)//2, dtype=torch.long, device=x.device)
#         input = args[0]
#         modified_input = map(
#             lambda x: x.index_select(dim, channel(x)),
#             input
#         )
#         modified_args = tuple(modified_input) + args[1:]
#         return modified_args
#     return _return_central_dim

@torch.no_grad()
def kl_divergence(source: Float[Tensor, 'batch1 channel sequence'], target: Float[Tensor, 'batch2 channel sequence']) -> Float[Tensor, '']:
    """ source/target: shape (batch, D, sequence). last 2d will be flattened. 
    mathematical definition: 
        target distribution: p(x)
        source distribution: q(x). 
    assume support(q) and support(p) are close enough.
        KL divergence: KL(p||q) = \int p(x) log(p(x)/q(x)) dx
        
    For future: use two infinite s
    """
    source, target = source.float().cpu(), target.float().cpu()
    source, target = map(torch.flatten, (source, target)) # (batch_1,) (batch_2,)
    # find the optimal bins
    min = torch.min(source.min(), target.min())
    max = torch.max(source.max(), target.max())
    bins = torch.linspace(min, max, 200) 
    bin_width = (bins[1] - bins[0]).item()
    # start_source = (source.min() - min) // bin_width # the start bin index of source
    # end_source = (source.max() - min) // bin_width + 1
    # start_target = (target.min() - min) // bin_width
    # end_target = (target.max() - min) // bin_width + 1
    # start_source, end_source, start_target, end_target = map(lambda x: x.item(), (start_source, end_source, start_target, end_target))
    
    # compute histogram of source and target
    p_source, bin_source = torch.histogram(
        source, 
        bins=bins,
    )
    p_target, bin_target = torch.histogram(
        target, 
        bins=bins
    )
    #   NOTE: this way the bins are supposed to be aligned. 
    
    # compute the prob
    # p_source = torch.zeros(len(bins), device=source.device).index_add_(
    #     0, torch.arange(start_source, end_source+1, step=1), p_source) # shape: [len(bins)]
    # p_target = torch.zeros(len(bins), device=target.device).index_add_(
    #     0, torch.arange(start_target, end_target+1, step=1), p_target) # shape: [len(bins)]
    p_source = p_source * torch.logical_and(p_source > 0, p_target > 0).float() # remove zero bins
    p_target = p_target * torch.logical_and(p_source > 0, p_target > 0).float() # remove zero bins
    p_source = p_source / p_source.sum() # re-normalize
    p_target = p_target / p_target.sum() # re-normalize
    
    # compute the kl divergence
    # NOTE: how to deal with when p_source == 0 but p_target != 0?
    kl_div = 0.
    for bin, p, q in zip(bins, p_source, p_target):
        if p > 0 and q > 0:
            kl_div += p * torch.log(p / q) # shape: scalar
    
    return kl_div
    
@torch.no_grad()
def ws_distance(
    source: Float[Tensor, 'batch1 channel sequence'],
    target: Float[Tensor, 'batch2 channel sequence'],
) -> Float[Tensor, '']:
    """ Calculates the Wasserstein distance between samples of two 1D distributions. 
    """
    source = source.flatten().float().cpu().numpy()
    target = target.flatten().float().cpu().numpy()
    _ws_dist = wasserstein_distance(source, target)
    
    return _ws_dist
    
@torch.no_grad()
def ks_2samp_test(
    source: Float[Tensor, 'batch1 channel sequence'],
    target: Float[Tensor, 'batch2 channel sequence'],
):
    """ Calculates the Kolmogorov-Smirnov 2-sample statistic. 
    
    returns:
        KstestResult(D statistic: float, pvalue: float)
        
            - D statistic: lower=better, the absolute max distance between the CDFs of the two samples.
            - pvalue: higher=better, a significance level of the test. 
        
                if pvalue < (1-confidence_threshold), then reject the null hypothesis that the distributions are the same.
                otherwise, if pvalue >= (1-confidence_threshold), then accept the null hypothesis that the distributions are the same.
                
    """
    source = source.flatten().float().cpu().numpy()
    target = target.flatten().float().cpu().numpy()
    _ks_2samp = ks_2samp(source, target)
    
    return _ks_2samp.statistic, _ks_2samp.pvalue

def ks_test_d(*args, **kwargs):
    """wrapper for ks_2samp_test"""
    return ks_2samp_test(*args, **kwargs)[0]

def ks_test_p(*args, **kwargs):
    """wrapper for ks_2samp_test"""
    return ks_2samp_test(*args, **kwargs)[1]

@torch.no_grad()
def estimate_psd(
    x: Float[Tensor, 'batch sequence'],
    window_size: Int,
) -> Float[Tensor, 'sequence window_size']:
    assert window_size % 2 == 1
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    padded = F.pad(x, (window_size//2, window_size//2), mode='circular') # shape: [batch, sequence+window_size-1]
    unfolded = padded.unfold(dimension=1, size=window_size, step=1) # shape: [batch, sequence, window_size]
    fouriered = torch.fft.fft(unfolded, dim=2, norm='ortho') # shape: [batch, sequence, window_size]
    fourier_mag = torch.abs(fouriered) # shape: [batch, sequence, window_size]
    fourier_mag = torch.mean(fourier_mag, dim=0) # shape: [sequence, window_size]
    
    pool_size = window_size // 2 + 1
    pool_stride = window_size // 2
    padded_mag = F.pad(fourier_mag, (window_size//2, window_size//2), mode='circular') # shape: [batch, sequence+window_size-1]
    avg_mag = F.avg_pool1d(rearrange(padded_mag, 'seq win -> 1 win seq'), kernel_size=pool_size, stride=pool_stride)
    
    return rearrange(avg_mag, '1 win seq -> seq win').numpy()

def fit_umap_mapper(data: Float[Tensor|ndarray, 'batch sequence'], **umap_kwargs):
    """
    Fit a UMAP mapper on the provided data.
    Data should be a numpy array of shape (batch, features).
    """
    from umap import UMAP
    if isinstance(data, Tensor):
        data = data.cpu().numpy()
    umap_mapper = UMAP(**umap_kwargs)
    print(f'fitting umap mapper with {umap_kwargs}')
    umap_mapper.fit(data)
    print(f'fitting umap mapper done')
    return umap_mapper

def apply_mapper_and_calculate_frechet(
    source: Float[Tensor|ndarray, 'batch sequence'],
    target: Float[Tensor|ndarray, 'batch sequence'],
    mapper: Callable,
) -> Float[Tensor, '']:
    """
    Apply a mapper (e.g. umap) to source data and calculate Fréchet Distance
    between source (generated data) and target (real data).
    Both source and target should be numpy arrays of shape (batch, features).
    """
    # Adjust data type
    if isinstance(source, Tensor):
        source = source.cpu().numpy()
    if isinstance(target, Tensor):
        target = target.cpu().numpy()
    
    # Apply dimension reduction
    reduced_source = mapper.transform(source)
    reduced_target = mapper.transform(target)
    
    # Calculating Frechet Distance
    frechet_dist = calculate_frechet(reduced_source, reduced_target)
    
    return frechet_dist

@torch.no_grad()
def calculate_frechet(
    source: Float[Tensor|ndarray, 'batch sequence'],
    target: Float[Tensor|ndarray, 'batch sequence'],
) -> Float[ndarray, '']:
    # Adjust data type
    if isinstance(source, Tensor):
        source = source.float().cpu().numpy()
    if isinstance(target, Tensor):
        target = target.float().cpu().numpy()
        
    if source.ndim == 3:
        mid_dim = source.shape[1] // 2
        source = source[:, mid_dim, :]
    if target.ndim == 3:
        mid_dim = target.shape[1] // 2
        target = target[:, mid_dim, :]
    
    # cov. numerical stability enhancement with eps 
    eps = 1e-6
    
    # Calculating Frechet Distance
    mean_source = np.mean(source, axis=0)
    cov_source = np.cov(source, rowvar=False) + np.eye(source.shape[1]) * eps
    
    mean_target = np.mean(target, axis=0)
    cov_target = np.cov(target, rowvar=False) + np.eye(target.shape[1]) * eps
    
    if cov_source.ndim < 2:
        cov_source = cov_source.reshape(1, 1)
        cov_target = cov_target.reshape(1, 1)
    
    diff = mean_source - mean_target
    covmean, _ = linalg.sqrtm(cov_source.dot(cov_target), disp=False)
    
    # Numerical error might give a complex component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    frechet_dist = np.dot(diff, diff) + np.trace(cov_source) + np.trace(cov_target) - 2 * np.trace(covmean)
    
    return frechet_dist

class FrechetUMAPDistance():
    """ Frechet UMAP distance. """
    def __init__(
        self,
        mapper_train_data: None|Float[Tensor|ndarray, 'batch sequence'] = None,
        label_mapper: str = 'unknown_dataset_unknown_resolution',
        mapper: None|Callable = None,
        **mapper_kwargs: dict,
    ):
        self.label = label_mapper
        if mapper is None:
            self.mapper = fit_umap_mapper(mapper_train_data, **mapper_kwargs)
        else:
            self.mapper = mapper
        
    @classmethod
    def from_pickle(cls, pickle_dir: str, label_mapper: str, n_components: int = 1):
        pickle_path = os.path.join(pickle_dir, f'umap_{label_mapper}_{n_components}.pkl')
        with open(pickle_path, 'rb') as f:
            mapper = pickle.load(f)
        return cls(mapper=mapper)
        
    def save_to_pickle(self, pickle_dir: str):
        pickle_path = os.path.join(pickle_dir, f'umap_{self.label}_{self.mapper.n_components}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.mapper, f)
            
    def __call__(
        self,
        source: Float[Tensor|ndarray, 'batch sequence'],
        target: Float[Tensor|ndarray, 'batch sequence'],
    ):
        if source.ndim == 3:
            source = source.squeeze(1)
        if target.ndim == 3:
            target = target.squeeze(1)
        result = apply_mapper_and_calculate_frechet(source, target, self.mapper)
        if isinstance(source, Tensor):
            result = torch.tensor(result)
            
        return result
    
class UMAPMkMMD():
    def __init__(
        self, 
        mapper_train_data: None|Float[Tensor|ndarray, 'batch sequence'] = None,
        label_mapper: str = 'unknown_dataset_unknown_resolution',
        mapper: None|Callable = None,
        **mapper_kwargs: dict,
    ):
        self.mkmmd = MkMMD()
        self.label = label_mapper
        if mapper is None:
            self.mapper = fit_umap_mapper(mapper_train_data, **mapper_kwargs)
        else:
            self.mapper = mapper
            
    @classmethod
    def from_pickle(cls, pickle_dir: str, label_mapper: str, n_components: int = 1):
        pickle_path = os.path.join(pickle_dir, f'umap_{label_mapper}_{n_components}.pkl')
        with open(pickle_path, 'rb') as f:
            mapper = pickle.load(f)
        return cls(mapper=mapper)
    
    def save_to_pickle(self, pickle_dir: str):
        pickle_path = os.path.join(pickle_dir, f'umap_{self.label}_{self.mapper.n_components}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.mapper, f)
            
    def __call__(self, source, target):
        if isinstance(source, Tensor):
            source = source.cpu().numpy()
        if isinstance(target, Tensor):
            target = target.cpu().numpy()
        reduced_source = self.mapper.transform(source)
        reduced_target = self.mapper.transform(target)
        return self.mkmmd(torch.from_numpy(reduced_source), torch.from_numpy(reduced_target))
        

def get_mapper_label(data_config: DataConfig) -> str:
    full_dataset_name = data_config.dataset
    if data_config.dataset == 'cossmic':
        cat_cossmic_dataset_names = '_'.join(data_config.subdataset_names)
    else:
        cat_cossmic_dataset_names = ''
    if data_config.dataset == 'cossmic':
        if 'pv' in cat_cossmic_dataset_names:
            full_dataset_name += '_pv'
        if 'grid' in cat_cossmic_dataset_names:
            full_dataset_name += '_grid_import'

        if 'industrial' in cat_cossmic_dataset_names:
            full_dataset_name += '_industrial'
        if 'residential' in cat_cossmic_dataset_names:
            full_dataset_name += '_residential'
        if 'public' in cat_cossmic_dataset_names:
            full_dataset_name += '_public'
    full_dataset_name += f'_{data_config.train_season}'
    full_dataset_name += f'_{data_config.resolution}'

    return full_dataset_name

class UMAPEvalCollection():
    def __init__(self, full_dataset_name: str):
        self.full_dataset_name = full_dataset_name
        
        try:
            with open(f'results/umap_{full_dataset_name}_24.pkl', 'rb') as f:
                self.mapper = pickle.load(f)
                self.mapper.random_state = 42
            with open(f'results/umap_{full_dataset_name}_1.pkl', 'rb') as f:
                self.mapper_extreme = pickle.load(f)
                self.mapper_extreme.random_state = 42
        except:
            self.mapper = None
            self.mapper_extreme = None
        
        if self.mapper is not None: 
            self.fud = FrechetUMAPDistance(mapper=self.mapper)
            self.umap_mkmmd = UMAPMkMMD(mapper=self.mapper)
        else:
            self.fud = lambda x,y : -1.
            self.umap_mkmmd = lambda x,y : -1.
        
        if self.mapper_extreme is not None:
            self.fud_extreme = FrechetUMAPDistance(mapper=self.mapper_extreme)
            self.umap_mkmmd_extreme = UMAPMkMMD(mapper=self.mapper_extreme)
        else:
            self.fud_extreme = lambda x,y : -1.
            self.umap_mkmmd_extreme = lambda x,y : -1.
    
    @staticmethod
    def prepare_data_pre_fn(func):
        _func = only_central_dim(func, dim=1)
        _func = reshape_3d_2d_pre_fn(_func)
        return _func
    
    # @prepare_data_pre_fn
    def reduced_kl_divergence(self, source, target):
        if self.mapper_extreme is None:
            return -1.
        reduced_source = self.mapper_extreme.transform(source)
        reduced_target = self.mapper_extreme.transform(target)
        
        return kl_divergence(torch.tensor(reduced_source), torch.tensor(reduced_target))
    
    # @prepare_data_pre_fn
    def reduced_ws_distance(self, source, target):
        # source = self._reshape_3d_2d(source)
        # target = self._reshape_3d_2d(target)
        if self.mapper_extreme is None:
            return -1.
        reduced_source = self.mapper_extreme.transform(source)
        reduced_target = self.mapper_extreme.transform(target)
        
        return ws_distance(torch.tensor(reduced_source), torch.tensor(reduced_target))
    
    # @prepare_data_pre_fn
    def reduced_ks_test_d(self, source, target):
        # source = self._reshape_3d_2d(source)
        # target = self._reshape_3d_2d(target)
        if self.mapper_extreme is None:
            return -1.
        reduced_source = self.mapper_extreme.transform(source)
        reduced_target = self.mapper_extreme.transform(target)
        
        return ks_test_d(torch.tensor(reduced_source), torch.tensor(reduced_target))
    
    def generate_eval_sequence(self):
        """ generate a list of functions to evaluate, including all the metrics. 
        - input: required to be tensor, allow shape (batch, channel, sequence) or (batch, sequence)
        - handles tensor -> array -> tensor automatically, and also handles 3d -> 2d automatically.
        - output: tensor
        - the first metric have to executed first. 
        """
        if self.mapper is None or self.mapper_extreme is None:
            return {}
        def _transform_data(source, target):
            source = source.detach()
            target = target.detach()
            if source.ndim == 3:
                channel = lambda x: torch.tensor((x.shape[1]-1)//2, dtype=torch.long, device=x.device)
                source = source.select(1, channel(source)) # shape: [batch, sequence]
                target = target.select(1, channel(target)) # shape: [batch, sequence]
            reduced_source = torch.tensor(self.mapper.transform(source.cpu().numpy()))
            reduced_target = torch.tensor(self.mapper.transform(target.cpu().numpy()))
            reduced_source_extreme = torch.tensor(self.mapper_extreme.transform(source.cpu().numpy()))
            reduced_target_extreme = torch.tensor(self.mapper_extreme.transform(target.cpu().numpy()))
            
            return (reduced_source, reduced_target), (reduced_source_extreme, reduced_target_extreme)
        
        def _wrap_first_fn(func, extreme = False):
            def _wrapped(source, target):
                "transform data and save to buffer"
                transformed_data = _transform_data(source, target)
                self.buffered_data = transformed_data
                result = func(*transformed_data[0 if not extreme else 1])
                if not isinstance(result, Tensor):
                    result = torch.tensor(result)
                return result
            return _wrapped
        
        def _wrap_other_fn(func, extreme = False):
            def _wrapped(source, target):
                "use buffered, ignore input"
                transformed_data = self.buffered_data
                result = func(*transformed_data[0 if not extreme else 1])
                if not isinstance(result, Tensor):
                    result = torch.tensor(result)
                return result
            return _wrapped
        
        self.buffered_data = None # for transformed data
        enabled_metrics = [
            ('fud', _wrap_first_fn(calculate_frechet)),
            ('fud_extreme', _wrap_other_fn(calculate_frechet, extreme=True)),
            ('umap_mkmmd', _wrap_other_fn(self.umap_mkmmd.mkmmd)),
            ('umap_mkmmd_extreme', _wrap_other_fn(self.umap_mkmmd_extreme.mkmmd, extreme=True)),
            ('reduced_kl_divergence', _wrap_other_fn(kl_divergence, extreme=True)),
            ('reduced_ws_distance', _wrap_other_fn(ws_distance, extreme=True)),
            ('reduced_ks_test_d', _wrap_other_fn(ks_test_d, extreme=True)),
        ]
        
        return enabled_metrics
    
class PairedUMAP():
    """train a pair of umap that is aligned between dataset (A U X) and (A U Y). """
    def __init__(self, 
                 common: Float[ndarray, 'batch channel'], 
                 source: Float[ndarray, 'batch channel'],
                 target: Float[ndarray, 'batch channel'],
                 n_neighbors: int = 15,
                 n_components: int = 2,
                 set_op_mix_ratio: float = 1.,
                 random_state: int = 42,
                 **kwargs):
        from umap import AlignedUMAP
        self.union_source = np.concatenate([common, source], axis=0)
        self.union_target = np.concatenate([common, target], axis=0)
        self.relation_dict = {idx: idx for idx in range(common.shape[0])}
        self.aligned_umap = AlignedUMAP(
            n_neighbors,
            n_components,
            set_op_mix_ratio = set_op_mix_ratio,
            random_state = random_state,
            **kwargs
        )
        print('Fitting aligned UMAP')
        self.embedding_source, self.embedding_target = self.aligned_umap.fit_transform(
            [self.union_source, self.union_target], 
            relations=[self.relation_dict]
        ) # would stuck at calculating the normalized laplacian's eigenvectors
        print('Fitting finished.')
        self.umap_source, self.umap_target = self.aligned_umap.mappers_

class FrechetMetric(torchmetrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add states for collecting source and target features
        self.add_state("source_features", 
                      default=[], 
                      dist_reduce_fx=None)  # Will handle reduction in compute
        self.add_state("target_features", 
                      default=[], 
                      dist_reduce_fx=None)

    def update(self, source: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate features for Frechet distance calculation"""
        # Ensure inputs are tensors and on correct device
        source, target = self._input_format(source, target)
        
        # Append to lists
        self.source_features.append(source)
        self.target_features.append(target)

    def compute(self) -> torch.Tensor:
        """Calculate Frechet distance using accumulated features"""
        # Concatenate all accumulated features
        source_all = torch.cat(self.source_features, dim=0)
        target_all = torch.cat(self.target_features, dim=0)

        # Calculate Frechet distance
        return calculate_frechet(source_all, target_all)
    
class FeatureStatisticsMetric(torchmetrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add states for collecting source and target features
        self.add_state("source_features", 
                      default=[], 
                      dist_reduce_fx=None)
        self.add_state("target_features", 
                      default=[], 
                      dist_reduce_fx=None)

    def update(self, source: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate features for statistics calculation"""
        source, target = self._input_format(source, target)
        
        self.source_features.append(source)
        self.target_features.append(target)

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate mean and std for both source and target features"""
        # Concatenate all accumulated features
        source_all = torch.cat(self.source_features, dim=0)
        target_all = torch.cat(self.target_features, dim=0)

        # Calculate statistics
        source_mean = torch.mean(source_all, dim=0)
        source_std = torch.std(source_all, dim=0)
        target_mean = torch.mean(target_all, dim=0)
        target_std = torch.std(target_all, dim=0)

        return source_mean, source_std, target_mean, target_std
    
class StatisticalDistancesMetric(torchmetrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("source_features", 
                      default=[], 
                      dist_reduce_fx=None)
        self.add_state("target_features", 
                      default=[], 
                      dist_reduce_fx=None)

    def update(self, source: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate features"""
        source, target = self._input_format(source, target)
        self.source_features.append(source)
        self.target_features.append(target)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Calculate all statistical distances"""
        # Concatenate all accumulated features
        source_all = torch.cat(self.source_features, dim=0)
        target_all = torch.cat(self.target_features, dim=0)

        # Calculate all distances
        kl_div = kl_divergence(source_all, target_all)
        ws_dist = wasserstein_distance(source_all, target_all)
        ks_d = ks_test_d(source_all, target_all)

        return {
            'kl_divergence': kl_div,
            'wasserstein_distance': ws_dist,
            'ks_test_d': ks_d
        }
        
class _MultiMetric(torchmetrics.Metric):
    def __init__(self, metric_fns: Dict[str, Callable], **kwargs):
        """
        Args:
            metric_fns: Dictionary mapping metric names to their computation functions.
                       Each function should take (source_features, target_features) as input.
        """
        # Since we handle data gathering in validation_step, we don't need 
        # torchmetrics' distributed features
        kwargs.update({
            'dist_sync_on_step': False,  # No need to sync since we gather in validation_step
        })
        super().__init__(**kwargs)
        self._printed = False
        
        # Store metric names and functions
        self.metric_names = list(metric_fns.keys())
        self._register_metric_fns(metric_fns)
        
        # Simple state storage, no DDP sync needed since we handle gathering elsewhere
        self.add_state("source_features", default=[])
        self.add_state("target_features", default=[])

    def _register_metric_fns(self, metric_fns):
        """Register metric functions as class methods to avoid pickle issues"""
        for name, fn in metric_fns.items():
            setattr(MultiMetric, f"compute_{name}", staticmethod(fn))

    def update(self, source: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update state with pre-gathered data 
        Note: This should only be called from the main process with gathered data
        """
        self.source_features.append(source.detach())
        self.target_features.append(target.detach())

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Compute metrics on accumulated data.
        Called by PyTorch Lightning at the end of validation epoch.
        """
        if len(self.source_features) == 0:
            return {}
            
        # Concatenate all accumulated features
        source_all = torch.cat(self.source_features, dim=0)
        target_all = torch.cat(self.target_features, dim=0)
        
        if not self._printed:
            print(f"source_all: {source_all.shape}, target_all: {target_all.shape}")
            self._printed = True

        # Compute all metrics
        results = {}
        for name in self.metric_names:
            metric_fn = getattr(self, f"compute_{name}")
            try:
                result = metric_fn(source_all, target_all)
                # Ensure result is a tensor
                if not isinstance(result, torch.Tensor):
                    result = torch.tensor(result, device=self.device)
                results[name] = result
            except Exception as e:
                print(f"Error computing metric {name}: {str(e)}")
                results[name] = torch.tensor(float('nan'), device=self.device)

        return results
    
class MultiMetric(object):
    def __init__(self, metric_fns: Dict[str, Callable], compute_on_cpu:bool=False, **kwargs):
        """
        non torchmetrics based implementation. it is so bad when it comes to ddp sync
        Args:
            metric_fns: Dictionary mapping metric names to their computation functions.
                       Each function should take (source_features, target_features) as input.
        """
        super().__init__()
        self._printed = False
        
        # Store metric names and functions
        self.metric_names = list(metric_fns.keys())
        self._device = torch.device('cpu') # following 
        self._register_metric_fns(metric_fns)
        self.compute_on_cpu = compute_on_cpu
        
        # Simple state storage, no DDP sync needed since we handle gathering elsewhere
        self.source_features = []
        self.target_features = []
        
    @property
    def device(self):
        "the device on which the metric is computed"
        return self._device
    
    def to(self, device):
        self._device = device
        return self

    def _register_metric_fns(self, metric_fns):
        """Register metric functions as class methods to avoid pickle issues"""
        for name, fn in metric_fns.items():
            setattr(MultiMetric, f"compute_{name}", staticmethod(fn))

    def update(self, source: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update state with pre-gathered data 
        Note: This should only be called from the main process with gathered data
        """
        self.source_features.append(source.detach())
        self.target_features.append(target.detach())

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Compute metrics on accumulated data.
        Called by PyTorch Lightning at the end of validation epoch.
        """
        if len(self.source_features) == 0:
            return {}
            
        # Concatenate all accumulated features
        source_all = torch.cat(self.source_features, dim=0)
        target_all = torch.cat(self.target_features, dim=0)
        
        if self.compute_on_cpu:
            source_all = source_all.cpu()
            target_all = target_all.cpu()
        
        source_all = source_all.float()
        target_all = target_all.float()
        
        if not self._printed:
            print(f"source_all: {source_all.shape}, target_all: {target_all.shape}")
            self._printed = True
            

        # Compute all metrics
        results = {}
        for name in self.metric_names:
            metric_fn = getattr(self, f"compute_{name}")
            try:
                result = metric_fn(source_all, target_all)
                # Ensure result is a tensor
                if not isinstance(result, torch.Tensor):
                    result = torch.tensor(result, device=self.device)
                results[name] = result
            except Exception as e:
                print(f"Error computing metric {name}: {str(e)}")
                results[name] = torch.tensor(float('nan'), device=self.device)

        return results
    
    def reset(self):
        self.source_features = []
        self.target_features = []
        
    def __call__(self, source, target):
        self.update(source, target)
        _result = self.compute()
        self.reset()
        return _result
    
    def forward(self, source, target):
        self.update(source, target)
        _result = self.compute()
        self.reset()
        return _result
    
def _tensor_stats(tensor: torch.Tensor) -> str:
    return f"mean:{tensor.mean().item():.4f}|std:{tensor.std().item():.4f}|max:{tensor.max().item():.4f}|min:{tensor.min().item():.4f}|median:{tensor.median().item():.4f}"

def main():
    ...
    
if __name__ == '__main__':
    main()
