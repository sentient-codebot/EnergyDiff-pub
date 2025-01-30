from typing import Optional, Any, Optional, Callable, Sequence

import torch
import numpy as np


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    
    :param mean1: tensor, the mean of the first Gaussian distribution
    :param logvar1: tensor, the log variance of the first Gaussian distribution
    :param mean2: tensor, the mean of the second Gaussian distribution
    :param logvar2: tensor, the log variance of the second Gaussian distribution
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def continuous_gaussian_log_likelihood(x, mean, log_scales):
    """
    Ref: https://github.com/facebookresearch/DiT
    
    Compute the log-likelihood of a Gaussian distribution with diagonal covariance.
    :param x: the targets
    :param mean: the mean of the Gaussian distribution (predicte)
    :param log_scales: the log of the standard deviation of the Gaussian distribution (predicted)
    :return: the log-likelihood of the Gaussian distribution
    """
    centered_x = x - mean
    inv_stdv = torch.exp(-log_scales)
    normalized_x = centered_x * inv_stdv
    log_probs = torch.distributions.Normal(torch.zeros_like(x), torch.ones_like(x)).log_prob(normalized_x)
    return log_probs

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal (erf function).
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, means, log_scales, n_bits=8):
    """
    Ref: https://github.com/facebookresearch/DiT
    
    :param x: the targets
    
    Not used now, just in case. 
    """
    assert x.shape == means.shape == log_scales.shape
    level = 2 ** n_bits - 1.
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / level)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / level)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

# helper functions
def default(value: Optional[Any], default_value: Any) -> Any:
    if value is not None:
        return value
    else:
        # more efficient in that it does not evaluate default_value unless it is needed
        if callable(default_value):
            return default_value()
        else:
            return default_value

def identify(t: Any, *args, **kwargs) -> Any:
    return t
