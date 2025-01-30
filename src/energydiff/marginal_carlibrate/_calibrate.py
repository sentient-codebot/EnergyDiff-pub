import torch
from energydiff.diffusion.typing import Tensor, Float
from energydiff.dataset.utils import MultiDimECDF

def calibrate(
    target: Float[Tensor, 'batch, channel, length'],
    source: Float[Tensor, 'batch, channel, length'],
)-> Float[Tensor, 'batch, channel, length']:
    """
    calibrate the marginals of source data to target data. return calibrated source data.
    """
    ecdf_source = MultiDimECDF(source)
    ecdf_target = MultiDimECDF(target)
    calibrated_source = ecdf_target.inverse_transform(
        ecdf_source.transform(source)
    )
    
    return calibrated_source