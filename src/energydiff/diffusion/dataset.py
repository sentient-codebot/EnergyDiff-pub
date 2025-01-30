import math

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from einops import rearrange

from typing import Annotated, Callable, Sequence, Iterable
from .typing import Float, Data1D
from .models_1d import SinusoidalPosEmb

class Dataset1D(Dataset):
    def __init__(self, tensor: Annotated[Tensor, "batch, channel, sequence"], transforms: Iterable[Callable] = [],):
        super().__init__()
        self.tensor = tensor.clone()
        self.transforms = transforms
        if transforms:
            assert all([callable(t) for t in transforms]), "transforms must be callable"

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        tensor = self.tensor[idx].clone()
        for t in self.transforms:
            tensor = t(tensor)
        return tensor, None
    
    @property
    def num_channel(self):
        return self.tensor.shape[1]
    
    @property
    def sequence_length(self):
        return self.tensor.shape[2]
    
    @property
    def sample_shape(self):
        return (self.num_channel, self.sequence_length)
    
    def __repr__(self):
        return f"Dataset1D(tensor={self.tensor.shape})"
    
    
# NOTE maybe I should use nn.Embedding instead? 
class DateOrSeasonEmb(nn.Module):
    """ Use sinusoidal positional embedding to encode date or season information
    
    Args:
        dim: dimension to be embedded
        scale_factor: scale factor to be multiplied to the input integers (default: 2*pi)
                        - pos_emb period \in (2*pi, 2*pi*1e8)
                        - season period is intrinsically 4
                        - date period is intrinsically 1~365
        
    Input:
        x: condition, Float[Tensor, 'batch 1 sequence']
        
    Output:
        encoded_x: encoded condition, Float[Tensor, 'batch dim sequence']
    """
    def __init__(self, dim: int, scale_factor: float = 2.0*math.pi):
        super().__init__()
        self.dim = dim
        self.scale_factor = scale_factor
        self.pos_emb = SinusoidalPosEmb(dim)
        
    def forward(self, x: Float[Tensor, 'batch 1 sequence']) -> Float[Tensor, 'batch dim sequence']:
        assert x.shape[1] == 1, "input shape must be 'batch 1 sequence'"
        b, c, l = x.shape
        x = x * self.scale_factor
        reshaped_x = rearrange(x, 'batch 1 sequence -> (batch sequence)')  
        encoded_x = self.pos_emb(reshaped_x) # shape: (batch*sequence, dim)
        encoded_x = rearrange(encoded_x, '(batch sequence) dim -> batch dim sequence', batch=b, sequence=l)
        return encoded_x
    
class ConditionalDataset1D(Dataset1D):
    def __init__(
        self, 
        tensor: Float[Tensor, 'batch channel sequence'],
        condition: None|dict[str, Float[Tensor, 'batch, *']],
        transforms: Iterable[Callable] = [],
    ):
        if condition is not None:
            assert tensor.shape[0] == list(condition.values())[0].shape[0], "batch size of tensor and condition must be same"
        super().__init__(tensor, transforms=transforms)
        self.raw_condition = condition
        
        self.condition, self.list_dim_cond = self.process_condition(self.raw_condition)
        
    def __getitem__(self, idx):
        _tensor, _ = super().__getitem__(idx)
        if self.condition is None:
            _condition = torch.empty(0) # the second value NEED to be a tensor for dataloader
        else:
            _condition = self.condition[idx].clone()
        return _tensor, _condition
    
    @property
    def num_condition_channel(self):
        if self.condition is None:
            return 0
        return self.condition.shape[1]
    
    @property
    def condition_shape(self):
        if self.condition is None:
            return (0, 0)
        return (self.num_condition_channel, self.condition.shape[2])
    
    def __repr__(self):
        return f"ConditionalDataset1D(tensor={self.tensor.shape}, condition={self.condition.shape if self.condition is not None else None})"
    
    # define a staticmethod so it can be used elsewhere
    @staticmethod
    def process_condition(
        condition: None|dict[str, Float[Tensor, 'batch,']|Float[Tensor, 'batch channel']],
    ):
        """ process the conditions using the provided processing function. 
            the conditions will be concatenated in channel dimension as output. 
            
        !CURRENT SETUP: just pass cond_process_fn = {} or None. 
            this will just return the concatenated raw condition.    
        """
        if condition is None:
            return None, None
        
        list_processed_condition = []
        list_dim = []
        for key, value in condition.items():
            if value.ndim == 1:
                value = rearrange(value, 'batch -> batch 1 1')
            elif value.ndim == 2:
                value = rearrange(value, 'batch channel -> batch channel 1')
            elif value.ndim == 3 and value.shape[2] == 1:
                pass # already in the right shape (batch, channel, 1)
            else:
                raise ValueError(f"condition {key} has invalid shape {value.shape}")
            
            list_processed_condition.append(value.float()) # shape: (batch, c, 1)
            list_dim.append(value.shape[1])
        
        processed_cond = torch.cat(list_processed_condition, dim=1) # shape: (batch, channel, 1)
        
        return processed_cond, list_dim