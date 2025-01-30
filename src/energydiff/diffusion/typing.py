import enum
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from torch import Tensor

from typing import (
    Annotated as BFloat16,  # noqa: F401
    Annotated as Bool,  # noqa: F401
    Annotated as Complex,  # noqa: F401
    Annotated as Complex64,  # noqa: F401
    Annotated as Complex128,  # noqa: F401
    Annotated as Float,  # noqa: F401
    Annotated as Float16,  # noqa: F401
    Annotated as Float32,  # noqa: F401
    Annotated as Float64,  # noqa: F401
    Annotated as Inexact,  # noqa: F401
    Annotated as Int,  # noqa: F401
    Annotated as Int8,  # noqa: F401
    Annotated as Int16,  # noqa: F401
    Annotated as Int32,  # noqa: F401
    Annotated as Int64,  # noqa: F401
    Annotated as Integer,  # noqa: F401
    Annotated as Key,  # noqa: F401
    Annotated as Num,  # noqa: F401
    Annotated as Shaped,  # noqa: F401
    Annotated as UInt,  # noqa: F401
    Annotated as UInt8,  # noqa: F401
    Annotated as UInt16,  # noqa: F401
    Annotated as UInt32,  # noqa: F401
    Annotated as UInt64,  # noqa: F401
)

Data1D = Float[Tensor, "batch, channel, sequence"]


ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start', 'pred_var_factor'])

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts
    """
    X_START = enum.auto()
    NOISE = enum.auto()
    V = enum.auto()
    
class ModelVarianceType(enum.Enum):
    """
    Either:
        - FIXED_SMALL
        - FIXED_LARGE
        - LEARNED_RANGE: learned, but in the range of [FIXED_SMALL, FIXED_LARGE]
    """
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()
    
class LossType(enum.Enum):
    """
    Which type of loss to use for learning the model mean;
    Loss for model var is always learned through VB term. 
    final_loss = loss_mean + loss_var if learned_variances else loss_mean
    
    Either:
        - MSE: L_{simple}
        - RESCALED_MSE: L_{simple} + VB term scaled DOWN if applicable
        - KL: L_{VB} for both mean and var
        - RESCALED_KL: L_{VB} for both mean and var, mean term scaled UP
    """
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL
    
class BetaScheduleType(enum.Enum):
    LINEAR = enum.auto()
    COSINE = enum.auto()