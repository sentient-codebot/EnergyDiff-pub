"""
This file provides utility functions such as ECDF estimation, PIT, etc.

Author: Nan Lin
Date: 2023-12-04

"""
from typing import Tuple, Annotated, Sequence, Set
from typing import Annotated as Float
from typing import Annotated as Int
from abc import ABC, abstractmethod
import os
from collections import namedtuple
import enum
from functools import partial

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from scipy.interpolate import interp1d, splrep, splev
from scipy.stats import lognorm

from energydiff.diffusion.typing import Float

DatasetWithMetadata = namedtuple(
    'DatasetWithMetadata', 
    ['profile', 'label', 'scaling_factor', 'pit']
)

class DataShapeType(enum.Enum):
    """
    Enum class for data shape
    """
    BATCH_CHANNEL_SEQUENCE = enum.auto()
    BATCH_SEQUENCE = enum.auto()
    CHANNEL_SEQUENCE = enum.auto()
    SEQUENCE = enum.auto()

NAME_SEASONS = [
    'winter',
    'spring',
    'summer',
    'autumn'
]

def months_of_season(season: str) -> list[int]:
    season = season.lower()
    if season == 'winter':
        months = [12, 1, 2]
    elif season == 'spring':
        months = [3, 4, 5]
    elif season == 'summer':
        months = [6, 7, 8]
    elif season in ['autumn', 'fall']:
        months = [9, 10, 11]
    else:
        raise ValueError('Invalid')
    return months

def season_of_month(month: int) -> str:
    assert month in range(1, 13)
    if month in [12, 1, 2]:
        season = 'winter'
    elif month in [3, 4, 5]:
        season = 'spring'
    elif month in [6, 7, 8]:
        season = 'summer'
    elif month in [9, 10, 11]:
        season = 'autumn'
    else:
        raise ValueError('Invalid')
    return season

class TimeSeriesDataset():
    """
    Base class for time series dataset.
    
    Contains common methods:
        - hash_option
        - normalize_fn
        - denormalize_fn
        - (pit not needed because it's imported from another class)
        - shuffle_dataset
        - vectorize_transform
    """
    original_dict_cond_dim = {} # to be overwritten by child class
    def __init__(self):
        self.all_dict_cond_dim = self.original_dict_cond_dim.copy()
    
    @property
    def dict_cond_dim(self,):
        "The actually used condition dimension"
        if hasattr(self, 'process_option'):
            if 'target_labels' in self.process_option:
                return {
                    key: value for key, value in self.all_dict_cond_dim.items() \
                        if key in self.process_option['target_labels']
                }
        return self.original_dict_cond_dim
    
    @abstractmethod
    def load_dataset(self) -> DatasetWithMetadata:
        raise NotImplementedError
    
    @abstractmethod
    def create_dataset(self) -> DatasetWithMetadata:
        raise NotImplementedError
    
    @abstractmethod
    def save_dataset(self, dataset: DatasetWithMetadata) -> None:
        raise NotImplementedError
    
    @staticmethod
    def hash_option(option: dict) -> str:
        str_option = ','.join([f'{key}:{value}' for key, value in option.items()])
        hashed_option = sum(ord(char) for char in str_option)
        hashed_option = f"{hashed_option}".replace('-', '1')
        
        return hashed_option
    
    @staticmethod
    def hash_set_string(set_string: Set[str]) -> str:
        " hash a set of string "
        str_concat = ','.join(sorted(set_string))
        hashed_set = sum(ord(char) for char in str_concat)
        hashed_set = f"{hashed_set}".replace('-', '1') # replace negative sign
        
        return hashed_set
    
    def vectorize_fn(self, data: Annotated[Tensor, 'batch 1 seq'], style: str = 'chronological', window_size: int = 3) -> Annotated[Tensor, 'batch window_size ...']:
        """Transform (N, 96) OR (N, 1, 96) to (N, D, 96) by concatenating temporally adjacent data or by STFT"""
        assert data.ndim in [2, 3]
        if data.ndim == 3:
            assert data.shape[1] == 1
        else:
            data = rearrange(data, 'n l -> n 1 l')
        
        if style in ['chronological', 'chrono']:
            data = self.chrono_vectorize(data, window_size)
        elif style == 'stft':
            data = self.stft_vectorize(data, window_size)
        elif style == 'patchify':
            data = self.patchify_vectorize(data, window_size)
        else:
            raise ValueError('Invalid style')
        return data
    
    def inverse_vectorize_fn(self, data: Annotated[Tensor, 'batch window_size seq'], style: str|None = None) -> Annotated[Tensor, 'batch 1 seq']:
        if data.ndim == 2:
            return data
        if data.shape[1] == 1:
            return data
        if style is None:
            style = self.process_option['style_vectorize']
        
        if style in ['chronological', 'chrono']:
            data = self.inverse_chrono_vectorize(data)
        elif style == 'stft':
            data = self.inverse_stft_vectorize(data)
        elif style == 'patchify':
            data = self.inverse_patchify(data)
        else:
            raise ValueError('Invalid style')
        return data
    
    @staticmethod
    def chrono_vectorize(data: Annotated[Tensor, "batch, 1, seq_length"], window_size: int = 3) -> Annotated[Tensor, "batch, window_size, seq_length"]:
        """..."""
        seq_length = data.shape[-1] # shape: [batch, 1, seq_length]
        data_padded = torch.nn.functional.pad(
                    data,
                    ((window_size-1)//2,)*2,
                    mode='circular'
        ) # shape: [batch, 1, seq_length+window_size-1]
        data_unfolded = data_padded.unfold(2, window_size, 1) # shape: [batch, 1, seq_length, window_size]
        data_vectorized = data_unfolded[:, :, :seq_length, :] # shape: [batch, 1, seq_length, window_size], *actually unnecessary*
        data_vectorized = rearrange(data_vectorized, 'b 1 s w -> b w s') # shape: [batch, window_size, seq_length]
        
        return data_vectorized.clone() 
    
    @staticmethod
    def inverse_chrono_vectorize(data: Annotated[Tensor, "batch, window_size, seq_length"]) -> Annotated[Tensor, "batch, 1, seq_length"]:
        """..."""
        mid_channel = data.shape[1] // 2
        data = data[:, mid_channel:mid_channel+1, :]
        return data
    
    @staticmethod
    def patchify_vectorize(data: Annotated[Tensor, "batch, 1, seq_length"], window_size: int = 3) -> Annotated[Tensor, "batch, window_size, seq_length//window_size"]:
        """Segment a long sequence into patches of window_size.
        1440, 1min:
            - 5, 5min: 288
            - 10, 10min: 144
            - 15, 15min: 96
        """
        assert data.shape[2] % window_size == 0
        return rearrange(data, 'b 1 (l c) -> b c l', c=window_size) # shape: [batch, window_size, seq_length//window_size]
        
    @staticmethod
    def inverse_patchify(data: Annotated[Tensor, "batch, window_size, seq_length//window_size"]) -> Annotated[Tensor, "batch, 1, seq_length"]:
        """Segment a long sequence into patches of window_size"""
        return rearrange(data, 'b c l -> b 1 (l c)') # shape: [batch, 1, seq_length]
        
    
    @staticmethod
    def stft_vectorize(data: Tensor, window_size: int = 5) -> Tensor:
        raise NotImplementedError
    
    @staticmethod
    def inverse_stft_vectorize(data: Tensor) -> Tensor:
        raise NotImplementedError
    
    def _denormalize_fn(self, data: torch.Tensor, max_value: torch.Tensor|int, min_value: torch.Tensor|int) -> torch.Tensor:
        """
        data shape: [sample, channel, seq_length] OR [sample, seq_length]

        """
        # assert max_value.ndim in [0, 1, 2, 3]
        assert data.ndim in [2, 3]
        _to_tensor = lambda x: torch.tensor(x, device=data.device) if not isinstance(x, torch.Tensor) else x
        max_value, min_value = map(_to_tensor, (max_value, min_value))
        if max_value.ndim == 0:
            max_value = max_value.unsqueeze(0)
        if min_value.ndim == 0:
            min_value = min_value.unsqueeze(0)
        assert max_value.shape == min_value.shape
        if max_value.ndim == 1:
            max_value = rearrange(max_value, 'l -> 1 1 l')
            min_value = rearrange(min_value, 'l -> 1 1 l')
        elif max_value.ndim == 2:
            max_value = rearrange(max_value, 'c l -> 1 c l')
            min_value = rearrange(min_value, 'c l -> 1 c l')
        
        if data.ndim == 2: # could only be channel==1 and channel dim is skipped
            dim_data_in = 2
            data = rearrange(data, 's l -> s 1 l')
        else:
            dim_data_in = 3
                
        max_value, min_value = map(
            lambda x: x.to(data.device),
            (max_value, min_value)
        )
        
        data = self.inverse_minmax_normalize(data, max_value, min_value)
        if dim_data_in == 2:
            return rearrange(data, 's 1 l -> s l')
        else:
            return data
        
    def denormalize_fn(self, 
                       data: Annotated[torch.Tensor, 'b (c) l'], 
                       *scaling_factor, **kwargs) -> torch.Tensor:
        if len(scaling_factor) == 0:
            scaling_factor = self.scaling_factor
        return self._denormalize_fn(data, *scaling_factor)
        
    def normalize_fn(self, data: torch.Tensor, min_value = None, max_value = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """ Normalize the data in to range [-1, 1]
        data shape: [day, channel, seq_length] OR [day, seq_length] 
        """
        max_value = 1.1 * torch.max(data) if max_value is None else max_value # shape: ()
        min_value = min(torch.tensor(0., dtype=data.dtype, device=data.device), torch.min(data)) if min_value is None else min_value # shape: ()
        data = self.minmax_normalize(data, max_value, min_value) # shape: [day, channel, seq_length] OR [day, seq_length]
        return data, (max_value, min_value)
    
    @staticmethod
    def minmax_normalize(data, max_value, min_value):
        """ Normalize the data in to range [0, 1]
        data shape: [day, channel, seq_length] OR [day, seq_length] 
        """
        data = (data - min_value) / (max_value - min_value) * 2. - 1. # shape: [day, channel, seq_length] OR [day, seq_length]
        return data
    
    @staticmethod
    def inverse_minmax_normalize(data, max_value, min_value):
        """ Inverse the minmax normalization
        data shape: [day, channel, seq_length] OR [day, seq_length] 
        """
        data = (data + 1) / 2 * (max_value - min_value) + min_value
        
        return data

    @staticmethod
    def clean_dataset(dataset: Annotated[Tensor, 'sample channel seq_length']) -> Annotated[Tensor, 'sample channel seq_length']:
        """Remove nan and inf. dataset shape [day, channel, seq_length]"""
        is_dim3 = dataset.ndim == 3
        if not is_dim3:
            dataset = rearrange(dataset, 'sample seq_length -> sample 1 seq_length')
        notnan = ~torch.isnan(dataset).any(dim=-1).any(dim=-1) # ~ is logical not
        notinf = ~torch.isinf(dataset).any(dim=-1).any(dim=-1)
        dataset = dataset[notnan & notinf, :, :] 
        if not is_dim3:
            dataset = rearrange(dataset, 'sample 1 seq_length -> sample seq_length')
        return dataset, notnan & notinf                
    
    @staticmethod
    def shuffle_fn(dataset: torch.Tensor) -> torch.Tensor:
        """Shuffle the dataset"""
        indices = torch.randperm(dataset.shape[0])
        return dataset[indices], indices # validated. smart copilot!
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')
    
    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')
    
    @property
    def num_channel(self):
        return self.dict_season_tensor[NAME_SEASONS[0]].shape[1]
    
    @property
    def sequence_length(self):
        return self.dict_season_tensor[NAME_SEASONS[0]].shape[2]
    
    @property
    def sample_shape(self):
        return (1, self.num_channel, self.sequence_length)
    
    def inverse_pit(self, x, season):
        if self.process_option['pit_transform']:
            if self.process_option['vectorize']:
                bs = x.shape[0]
                x = rearrange(x, 'bs channel sequence -> (bs channel) sequence')
            x = self.pit_season[season].inverse_transform(x)
            if self.process_option['vectorize']:
                x = rearrange(x, '(bs channel) sequence -> bs channel sequence', bs=bs)
        return x
    
class MultiDimECDFV2():
    """
    Batched ECDF estimation.
    
    ref: https://github.com/MauricioSalazare/multi-copula/
    
    """
    def __init__(self, x, interp_mode='spline'):
        """
        Args:
            x: input data, shape (n_samples, dim1, dim2) or (n_samples, dim)
        """
        # super().__init__()
        if len(x.shape) == 2:
            self.dim = x.shape[1]
            x = x
        else:
            self.dims = (x.shape[1], x.shape[2])
            self.dim = x.shape[1] * x.shape[2]
            x = rearrange(x, 'n d1 d2 -> n (d1 d2)')
        self.x = x # (n, cl)
        self.x_sorted, _ = torch.sort(x, dim=0) # expected: DataShapeType.BATCH_SEQUENCE
        self.num_sample = x.shape[0]
        self.init_cdf(interp_mode=interp_mode)
        
    def init_cdf(self, interp_mode='spline') -> None:
        " returns x, cdf "
        collect_cdf = []
        collect_icdf = []
        bounds_y = []
        for channel_idx in range(self.x.shape[1]):
            x, counts = torch.unique(self.x_sorted.contiguous()[:,channel_idx], sorted=True, return_counts=True) # shape (n_unique,)
            events = torch.cumsum(counts, dim=0) # shape (n_unique,)
            normalized_events = events.float()/self.num_sample
            x = x.cpu().numpy()
            normalized_events = normalized_events.cpu().numpy()
            if interp_mode == 'linear':
                _x_bnd = np.r_[-np.inf, x, np.inf] # see ref. 
                _y_bnd = np.r_[0.0, normalized_events, 1.0] # see ref
                cdf = interp1d(_x_bnd, _y_bnd)
                icdf = interp1d(_y_bnd, _x_bnd)
            else:
                cdf_rep = splrep(x, normalized_events, s=0.01)
                icdf_rep = splrep(normalized_events, x, s=0.01)
                cdf = partial(splev, tck=cdf_rep)
                icdf = partial(splev, tck=icdf_rep)
                y_min, y_max = normalized_events.min(), normalized_events.max()
            collect_cdf.append(cdf)
            collect_icdf.append(icdf)
            bounds_y.append((y_min, y_max))
            
        def apply_cdf(_x):
            "_x: tensor/array (n, dim)"
            converter = self.ArrayConverter(_x)
            _x = converter.convert(_x) # (n, dim)
            _y = []
            for channel_idx in range(_x.shape[1]):
                __x = _x[:,channel_idx] # (n,)
                __y = collect_cdf[channel_idx](__x)
                _y.append(__y)
            _y = np.stack(_y, axis=1)
            _y = converter.revert(_y)
            return _y
        
        def apply_icdf(_y):
            "_y: tensor/array (n, dim) in [0,1]"
            converter = self.ArrayConverter(_y)
            _y = converter.convert(_y)
            _x = []
            for channel_idx in range(_y.shape[1]):
                __y = _y[:,channel_idx] # (n,)
                __y = np.where(
                    __y>=bounds_y[channel_idx][0],
                    __y,
                    bounds_y[channel_idx][0]
                )
                __y = np.where(
                    __y<=bounds_y[channel_idx][1],
                    __y,
                    bounds_y[channel_idx][1]
                )
                __x = collect_icdf[channel_idx](__y)
                _x.append(__x)
            _x = np.stack(_x, axis=1)
            _x = converter.revert(_x)
            return _x
            
        self.cdf = apply_cdf
        self.icdf = apply_icdf
    
    class ArrayConverter():
        def __init__(self, maybe_tensor):
            "tensor -> array"
            self.is_input_tensor = isinstance(maybe_tensor, torch.Tensor)
            if self.is_input_tensor:
                self.device = maybe_tensor.device 
            else:
                self.device = None
        
        def convert(self, input):
            if self.is_input_tensor:
                return self._array(input)
            else:
                return self._do_nothing(input)
            
        def revert(self, input):
            if self.is_input_tensor:
                return self._tensor(input)
            else:
                return self._do_nothing(input)
        
        def _do_nothing(self, input):
            return input
        
        def _array(self, tensor):
            return tensor.cpu().numpy()

        def _tensor(self, array):
            return torch.from_numpy(array).to(self.device)
        
    def to(self, device):
        self.x = self.x.to(device)
        self.x_sorted = self.x_sorted.to(device)
        return self

    def transform(self, x):
        """
        Args:
            x: input data, shape (b, c, l) or (b, c*l) or (c, l) or (c*l)
        (x_sorted shape) (b, c*l)
        Returns:
            y: ECDF transformed data, shape (batch_size, dim1, dim2) or (batch_size, dim)
        """
        x_sorted = self.x_sorted.to(x.device)
        xndim = len(x.shape)
        input_shape_type = DataShapeType.BATCH_SEQUENCE
        if xndim == 3:
            input_shape_type = DataShapeType.BATCH_CHANNEL_SEQUENCE
            c, l = x.shape[1], x.shape[2]
            x = rearrange(x, 'b c l -> b (c l)')
        elif xndim == 2:
            if x.shape[0]*x.shape[1] == self.x_sorted.shape[1]:
                input_shape_type = DataShapeType.CHANNEL_SEQUENCE
                c, l = x.shape[0], x.shape[1]
                x = rearrange(x, 'c l -> 1 (c l)')
            else:
                input_shape_type = DataShapeType.BATCH_SEQUENCE
                pass
        elif xndim == 1:
            input_shape_type = DataShapeType.SEQUENCE
            x = rearrange(x, 'cl -> 1 cl')
        else:
            raise ValueError('Invalid input shape')
            
        # x (n, cl)
        y = self.cdf(x) # (n, cl)
        y = torch.clamp(y, 0., 1.)
        
        if input_shape_type == DataShapeType.BATCH_CHANNEL_SEQUENCE:
            y = rearrange(y, 'b (c l) -> b c l', c=c)
        elif input_shape_type == DataShapeType.CHANNEL_SEQUENCE:
            y = rearrange(y, '1 (c l) -> c l', c=c)
        elif input_shape_type == DataShapeType.SEQUENCE:
            y = rearrange(y, '1 cl -> cl')
        elif input_shape_type == DataShapeType.BATCH_SEQUENCE:
            pass
        else:
            pass # never reach here
        
        return y
    
    def inverse_transform(self, y):
        """
        Args:
            y: ECDF transformed data, shape (batch_size, dim1, dim2) or (batch_size, dim)
        Returns:
            x: input data, shape (batch_size, dim)
        """
        yndim = len(y.shape)
        if yndim == 3:
            input_shape_type = DataShapeType.BATCH_CHANNEL_SEQUENCE
            c, l = y.shape[1], y.shape[2]
            y = rearrange(y, 'b c l -> b (c l)')
        elif yndim == 2:
            if y.shape[0]*y.shape[1] == self.x_sorted.shape[1]:
                input_shape_type = DataShapeType.CHANNEL_SEQUENCE
                c, l = y.shape[0], y.shape[1]
                y = rearrange(y, 'c l -> 1 (c l)')
            else:
                input_shape_type = DataShapeType.BATCH_SEQUENCE
                pass
        elif yndim == 1:
            input_shape_type = DataShapeType.SEQUENCE
            y = rearrange(y, 'cl -> 1 cl')
        else:
            raise ValueError('Invalid input shape')
        
        x = self.icdf(y)
        
        if input_shape_type == DataShapeType.BATCH_CHANNEL_SEQUENCE:
            x = rearrange(x, 'b (c l) -> b c l', c=c)
        elif input_shape_type == DataShapeType.CHANNEL_SEQUENCE:
            x = rearrange(x, '1 (c l) -> c l', c=c)
        elif input_shape_type == DataShapeType.SEQUENCE:
            x = rearrange(x, '1 cl -> cl')
        elif input_shape_type == DataShapeType.BATCH_SEQUENCE:
            pass
        return x
    
    def re_estimate(self, x):
        """
        Args:
            x: input data, shape (n_samples, dim)
        """
        self.x = self.x.to(x.device)
        if len(x.shape) == 2:
            self.x = x
            self.dim = x.shape[1]
        else:
            self.x = rearrange(x, 'n d1 d2 -> n (d1 d2)')
            self.dims = (x.shape[1], x.shape[2])
            self.dim = x.shape[1] * x.shape[2]
        self.num_sample = x.shape[0]
        self.x_sorted, _ = torch.sort(x, dim=0)
        self.init_cdf()
        
    def continue_estimate(self, x):
        """
        Args:
            x: input data, shape (n_samples, dim)
        """
        self.x = self.x.to(x.device)
        if len(x.shape) == 3:
            x = rearrange(x, 'n d1 d2 -> n (d1 d2)')
        self.x = torch.cat([self.x, x], dim=0)
        self.num_sample = self.x.shape[0]
        self.x_sorted, _ = torch.sort(self.x, dim=0)
        self.init_cdf()
        
    def __call__(self, x):
        """ get ecdf of x """
        x = self.transform(x)
        
        return x

class MultiDimECDFV1():
    """
    Batched ECDF estimation.
    """
    def __init__(self, x):
        """
        Args:
            x: input data, shape (n_samples, dim1, dim2) or (n_samples, dim)
        """
        # super().__init__()
        if len(x.shape) == 2:
            self.dim = x.shape[1]
            x = x
        else:
            self.dims = (x.shape[1], x.shape[2])
            self.dim = x.shape[1] * x.shape[2]
            x = rearrange(x, 'n d1 d2 -> n (d1 d2)')
        self.x = x
        self.num_sample = x.shape[0]
        self.x_sorted, _ = torch.sort(x, dim=0) # expected: DataShapeType.BATCH_SEQUENCE
        # self.x_sorted = self.stable(self.x_sorted)
        
    @property
    def cdf(self) -> Tuple[torch.Tensor, torch.Tensor]:
        " returns x, cdf "
        x, counts = torch.unique(self.x_sorted.contiguous(), dim=0, sorted=True, return_counts=True) # shape (n_unique, dim)
        events = torch.cumsum(counts, dim=0) # shape (n_unique, dim)
        cdf = events.float() / self.num_sample # shape (n_unique, dim)
        return x, cdf
        
    def to(self, device):
        self.x = self.x.to(device)
        self.x_sorted = self.x_sorted.to(device)
        return self

    def transform(self, x):
        """
        Args:
            x: input data, shape (b, c, l) or (b, c*l) or (c, l) or (c*l)
        (x_sorted shape) (b, c*l)
        Returns:
            y: ECDF transformed data, shape (batch_size, dim1, dim2) or (batch_size, dim)
        """
        x_sorted = self.x_sorted.to(x.device)
        xndim = len(x.shape)
        input_shape_type = DataShapeType.BATCH_SEQUENCE
        if xndim == 3:
            input_shape_type = DataShapeType.BATCH_CHANNEL_SEQUENCE
            c, l = x.shape[1], x.shape[2]
            x = rearrange(x, 'b c l -> b (c l)')
        elif xndim == 2:
            if x.shape[0]*x.shape[1] == self.x_sorted.shape[1]:
                input_shape_type = DataShapeType.CHANNEL_SEQUENCE
                c, l = x.shape[0], x.shape[1]
                x = rearrange(x, 'c l -> 1 (c l)')
            else:
                input_shape_type = DataShapeType.BATCH_SEQUENCE
                pass
        elif xndim == 1:
            input_shape_type = DataShapeType.SEQUENCE
            x = rearrange(x, 'cl -> 1 cl')
        else:
            raise ValueError('Invalid input shape')
            
        indices = torch.searchsorted(x_sorted.transpose(0,1).contiguous(), x.transpose(0,1).contiguous(), side='right').transpose(0,1) # [b, c*l]
        # --- old ---
        # indices = torch.clamp(indices, 1, self.num_sample-1) # if x >= x_sorted[0], then indices >= 1 naturally with right=True. 
            # if x >= x_sorted[-1], then indices == self.num_sample. 
        # x1 = torch.gather(x_sorted, 0, indices - 1)
        # x2 = torch.gather(x_sorted, 0, indices)
        # y1 = (indices - 1).float() / self.num_sample
        # y2 = indices.float() / self.num_sample
        # y = y1 + (x - x1 + 1e-6) * (y2 - y1) / (x2 - x1 + 1e-6) # Guarantee: if x == x2 == x1, then y == y2
        # --- /old ---
        cdf_values = indices.float() / self.num_sample

        # y = torch.where(x<=self.x_sorted.min(dim=0, keepdim=True)[0], torch.zeros_like(y), y) # The order of these two lines is important.
        # y = torch.where(x>=self.x_sorted.max(dim=0, keepdim=True)[0], torch.ones_like(y), y) # if x_sorted.min() == x_sorted.max(), then y will be 1. 
        y = torch.clamp(cdf_values, 0., 1.)
        
        if input_shape_type == DataShapeType.BATCH_CHANNEL_SEQUENCE:
            y = rearrange(y, 'b (c l) -> b c l', c=c)
        elif input_shape_type == DataShapeType.CHANNEL_SEQUENCE:
            y = rearrange(y, '1 (c l) -> c l', c=c)
        elif input_shape_type == DataShapeType.SEQUENCE:
            y = rearrange(y, '1 cl -> cl')
        elif input_shape_type == DataShapeType.BATCH_SEQUENCE:
            pass
        else:
            pass # never reach here
        
        return y
    
    def inverse_transform(self, y):
        """
        Args:
            y: ECDF transformed data, shape (batch_size, dim1, dim2) or (batch_size, dim)
        Returns:
            x: input data, shape (batch_size, dim)
        """
        x_sorted = self.x_sorted.to(y.device)
        yndim = len(y.shape)
        if yndim == 3:
            input_shape_type = DataShapeType.BATCH_CHANNEL_SEQUENCE
            c, l = y.shape[1], y.shape[2]
            y = rearrange(y, 'b c l -> b (c l)')
        elif yndim == 2:
            if y.shape[0]*y.shape[1] == x_sorted.shape[1]:
                input_shape_type = DataShapeType.CHANNEL_SEQUENCE
                c, l = y.shape[0], y.shape[1]
                y = rearrange(y, 'c l -> 1 (c l)')
            else:
                input_shape_type = DataShapeType.BATCH_SEQUENCE
                pass
        elif yndim == 1:
            input_shape_type = DataShapeType.SEQUENCE
            y = rearrange(y, 'cl -> 1 cl')
        else:
            raise ValueError('Invalid input shape')
        
        y = torch.clamp(y, 0., 1.)
        y_scaled = y * self.num_sample
        indices_lower = torch.floor(y_scaled).long() # range (0, num_sample)
        indices_upper = torch.ceil(y_scaled).long() # range (0, num_sample)
        xlower = torch.gather(x_sorted, 0, indices_lower.clamp(min=1,max=self.num_sample)-1) # range bounded by x_sorted
        xupper = torch.gather(x_sorted, 0, indices_upper.clamp(min=1,max=self.num_sample)-1)
            # theoretically indices should never be <= 0, probability == 0
        x_interp = xlower + (y_scaled - indices_lower.float()) * (xupper - xlower)
        # x = x_interp
        x = xlower
        
        if input_shape_type == DataShapeType.BATCH_CHANNEL_SEQUENCE:
            x = rearrange(x, 'b (c l) -> b c l', c=c)
        elif input_shape_type == DataShapeType.CHANNEL_SEQUENCE:
            x = rearrange(x, '1 (c l) -> c l', c=c)
        elif input_shape_type == DataShapeType.SEQUENCE:
            x = rearrange(x, '1 cl -> cl')
        elif input_shape_type == DataShapeType.BATCH_SEQUENCE:
            pass
        return x
    
    def re_estimate(self, x):
        """
        Args:
            x: input data, shape (n_samples, dim)
        """
        self.x = self.x.to(x.device)
        if len(x.shape) == 2:
            self.x = x
            self.dim = x.shape[1]
        else:
            self.x = rearrange(x, 'n d1 d2 -> n (d1 d2)')
            self.dims = (x.shape[1], x.shape[2])
            self.dim = x.shape[1] * x.shape[2]
        self.num_sample = x.shape[0]
        self.x_sorted, _ = torch.sort(x, dim=0)
        
    def continue_estimate(self, x):
        """
        Args:
            x: input data, shape (n_samples, dim)
        """
        self.x = self.x.to(x.device)
        if len(x.shape) == 3:
            x = rearrange(x, 'n d1 d2 -> n (d1 d2)')
        self.x = torch.cat([self.x, x], dim=0)
        self.num_sample = self.x.shape[0]
        self.x_sorted, _ = torch.sort(self.x, dim=0)
        
    def __call__(self, x):
        """ get ecdf of x """
        x = self.transform(x)
        
        return x
    
class MultiDimECDF(MultiDimECDFV1):...

class LogNorm(MultiDimECDF):
    """
    !not fully debugged yet. 
    Batched LogNorm estimation 
    !ASSUME x is in (-1,1)
    """
    def __init__(self, x, *args, **kwargs):
        assert x.min() >= -1.
        x = self.to_zero_one(x)
        super().__init__(x)
        # self.x # (n, cl)
        
    def to_zero_one(self, x):
        "unnecessary anymore because the loc parameter is fitted"
        # return (x+1.)/2. # (-1,1) -> (0,1)
        return x
    
    def to_minus_one_one(self, x):
        # return x*2.-1
        return x
        
    def init_cdf(self, *args, **kwargs) -> None:
        # self.x, shape (n, cl)
        collect_cdf = []
        collect_icdf = []
        for channel_idx in range(self.x.shape[1]):
            x = self.x[:,channel_idx].cpu().numpy()
            shape, loc, scale = lognorm.fit(x)
            cdf = partial(lognorm.cdf, s=shape, loc=loc, scale=scale)
            icdf = partial(lognorm.ppf, s=shape, loc=loc, scale=scale)
            collect_cdf.append(cdf)
            collect_icdf.append(icdf)
        
        def apply_cdf(_x):
            "_x: tensor/array, (-1,1)"
            converter = self.ArrayConverter(_x)
            _x = converter.convert(_x)
            _x = self.to_zero_one(_x)
            _y = []
            for channel_idx in range(_x.shape[1]):
                __x = _x[:,channel_idx] # (n,)
                __y = collect_cdf[channel_idx](__x)
                _y.append(__y)
            _y = np.stack(_y, axis=1)
            _y = converter.revert(_y)
            return _y
        
        def apply_icdf(_y):
            "_y: tensor/array (n, dim) in [0,1]"
            converter = self.ArrayConverter(_y)
            _y = converter.convert(_y)
            _y = np.clip(_y, 0., 1.)
            _x = []
            for channel_idx in range(_y.shape[1]):
                __y = _y[:,channel_idx] # (n,)
                __x = collect_icdf[channel_idx](__y)
                _x.append(__x)
            _x = np.stack(_x, axis=1)
            _x = self.to_minus_one_one(_x)
            _x = converter.revert(_x)
            return _x
        
        self.cdf = apply_cdf
        self.icdf = apply_icdf
    
    
class PIT():
    """ apply PIT to samples per channel. """ # is this class necessary?
    def __init__(self, samples = None, perturb = False):
        self.ecdf = None
        if samples is not None:
            if perturb:
                samples = samples + torch.randn_like(samples) * 1e-6
            self.ecdf = MultiDimECDF(samples)
            self.transform = self.ecdf.transform
            self.inverse_transform = self.ecdf.inverse_transform
        
    def __call__(self, x):
        """ apply PIT to samples per channel. """
        if self.ecdf is not None:
            x = self.ecdf(x)
        else:
            raise ValueError("PIT is not initialized.")
        return x
    
    def to(self, device):
        self.ecdf.to(device)
        return self
    
    def fit(self, x) -> None:
        """ fit """
        self.ecdf = MultiDimECDF(x)
        self.transform = self.ecdf.transform
        self.inverse_transform = self.ecdf.inverse_transform
    
    def fit_transform(self, x):
        """ fit and transform """
        self.ecdf = MultiDimECDF(x)
        self.transform = self.ecdf.transform
        self.inverse_transform = self.ecdf.inverse_transform
        return self.ecdf.transform(x)

def standard_normal_cdf(x: Float[Tensor, 'any']):
    return 0.5 * (1 + torch.erf(x / np.sqrt(2)))

def standard_normal_icdf(y: Float[Tensor, 'any']):
    y = y.clamp(min=0., max=0.999999)
    return np.sqrt(2) * torch.erfinv(2 * y - 1)