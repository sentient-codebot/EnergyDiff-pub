import warnings
import math
from multiprocessing import cpu_count
from random import random
from functools import partial
from typing import Optional, Any, Optional, Callable, Sequence
from .typing import Float, Data1D, Int

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import Adam

from einops import rearrange, reduce, einsum
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

from .version import __version__
from .utils import default

class QuantizeCondition(nn.Module):
    def __init__(self, num_level: int = 400, max_val: float = 20000., min_val: float = 0.0):
        super().__init__()
        self.num_level = num_level
        self.min_val = min_val
        self.max_val = max_val
        
        if (max_val - min_val) < 1e-6:
            raise ValueError(f"max_val {max_val} must be (at least 1e-6) larger than min_val {min_val}")
        
        self.level_width = (max_val - min_val) / self.num_level
        
    def forward(self, x: Float[Tensor, 'batch, ']):
        x = torch.clamp(x, self.min_val, self.max_val) 
        x = torch.floor((x - self.min_val) / self.level_width)
        return x
    
    def inverse(self, x: Float[Tensor, 'batch, ']):
        x = x * self.level_width + self.min_val
        return x

class IntegerEmbedder(nn.Module):
    """
    Embed integer suchs as day index or season index into vector representations. Also handles label dropout (for classifier-free) guidance? 
        (See https://github.com/facebookresearch/DiT/blob/main/models.py#L27)
        
        for unconditional generation:
            - use dropout > 0. to enable condition dropout (= unconditional generation)
            - set `train=True|None` or use `force_drop_ids` to force drop certain ids
        
    Arguments:
        - num_embedding: number of embeddings (e.g. 365 for day index)
        - dim_embedding: dimension of embedding
        - dropout: dropout rate for discarding condition (for classifier-free guidance) (default: 0.1)
        
    Input:
        - cond: (batch, )
        - train: bool, whether to use dropout (default: None, use self.training)
        - force_drop_ids: (batch, ), set to 1. to force drop certain ids (default: None)
    """
    def __init__(self, num_embedding: int, dim_embedding: int, dropout: float = 0.1, quantize: bool = False,
                 quantize_max_val: float = 20000., quantize_min_val: float = 0.0):
        super().__init__()
        use_null_embedding = dropout > 0 # drop condition = unconditional generation
        self.num_embedding = num_embedding
        self.dim_embedding = dim_embedding
        self.dropout = dropout
        self.quantize = quantize
        self.quantize_level = num_embedding
        self.quantize_max_val = quantize_max_val
        self.quantize_min_val = quantize_min_val
        
        if self.quantize:
            self.quantizer = QuantizeCondition(
                num_level=self.quantize_level,
                max_val=self.quantize_max_val,
                min_val=self.quantize_min_val
            )
        self.embedding_table = nn.Embedding(num_embedding + use_null_embedding, dim_embedding)
        
    def token_drop(self, cond, force_drop_ids=None):
        """
        Drop condition to enable classifier-free guidance. 
        
        `cond`: condition, shape: (batch, ), range: [0, num_embedding)
        
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(cond.shape[0], device=cond.device) < self.dropout
        else:
            drop_ids = force_drop_ids == 1
        cond = torch.where(drop_ids, self.num_embedding, cond)
        return cond
    
    def forward(self, 
                cond: Float[Tensor, '*, '],
                train: bool|None = None,
                force_drop_ids: Optional[Float[Tensor, '*, ']] = None
               ) -> Float[Tensor, '*, dim_embedding']:
        if self.quantize:
            cond = self.quantizer(cond)
        
        cond = cond.long() # floor
        
        train = default(train, self.training)
        use_dropout = self.dropout > 0.
        #   if train and use_
        if (train and use_dropout) or (force_drop_ids is not None):
            cond = self.token_drop(cond, force_drop_ids)
        
        embeddings = self.embedding_table(cond.long()) # shape: (batch, dim_embedding)
        return embeddings
    
class Zeros(nn.Module):
    def __init__(self, num_embedding: int, dim_embedding: int, *args, **kwargs):
        super().__init__()
        self.num_embedding = num_embedding
        self.dim_embedding = dim_embedding
        
    def forward(self, cond, *args, **kwargs):
        batch_size = cond.shape[0]
        return torch.zeros(batch_size, self.dim_embedding, device=cond.device, dtype=cond.dtype)

class EmbedderWrapper(nn.Module):
    """
    Automatically splits input into multiple parts and SUMS each part together. 
        Concatenates the embeddings and returns. 
    
    Arguments:
        - embedder_list: list of embedders
            - either (batch, ) -> (batch, dim_embedding)
            - or (batch, channel) -> (batch, dim_embedding)
        - list_dim: list of dimensions for each condition (num_condiiton_i_channel)
        
    Input:
        - c: (batch, num_condiiton_channel, 1)
        - *args: additional arguments for embedders, such as `train` or `force_drop_ids`
        
    Return:
        - encoded_c: (batch, dim_base, 1)
    
    """
    def __init__(
        self,
        list_embedder: Sequence[IntegerEmbedder],
        list_dim_cond: Sequence[int],
    ):
        super().__init__()
        assert len(list_embedder) == len(list_dim_cond), 'embedder_list and dim_list must have same length'
        self.list_embedder = nn.ModuleList(list_embedder)  if not isinstance(list_embedder, nn.ModuleList) \
            else list_embedder
        self.list_dim_cond = list_dim_cond
        
    def forward(self, c: Float[Tensor, 'batch, num_condition_channel, 1'], *args, **kwargs) -> Float[Tensor, 'batch dim_base, 1']:
        list_c = c.split(self.list_dim_cond, dim=1) # list of (batch, dim, 1)
        
        # list_encoded_c = []
        sum_encoded_c = 0.
        for cond, embedder in zip(list_c, self.list_embedder):
            if cond.shape[1] == 1: # channel == 1, shape (batch, 1, 1)
                cond = rearrange(cond, 'batch 1 1 -> batch')
                encoded_c = embedder(cond, *args, **kwargs) # shape: (batch, dim_embedding)
            else: # channel > 1, shape (batch, channel, 1)
                cond = rearrange(cond, 'batch channel 1 -> batch channel')
                encoded_c = embedder(cond, *args, **kwargs) # shape: (batch, channel) -> (batch, dim_embedding)
            # list_encoded_c.append(encoded_c)
            sum_encoded_c += encoded_c # shape: (batch, dim_embedding)
            
        # concat_encoded_c = torch.cat(list_encoded_c, dim=1) # shape: (batch, dim_all)
        
        return rearrange(sum_encoded_c, 'batch channel -> batch channel 1')

# small modules
class Residual(nn.Module):
    def __init__(self, func: Callable[[Tensor], Tensor]):
        super().__init__()
        self.func = func
    
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.func(x, *args, **kwargs) + x
    
class Upsample(nn.Module):
    def __init__(self, dim_in: int, dim_out: int = 3, scale_factor: int = 2, mode: str = 'nearest'):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.scale_factor = scale_factor
        self.mode = mode
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.conv = nn.Conv1d(dim_in, dim_out, 3, padding=1)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        return x
    
class Downsample(nn.Module):
    def __init__(self, 
                 dim_in: int, 
                 dim_out: Optional[int] = 3):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.downsample_conv = nn.Conv1d(dim_in, dim_out, 4, stride=2, padding=1)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample_conv(x)
        return x
    
class RMSNorm(nn.Module):
    "do normalization over channel dimension and multiply with a learnable parameter g"
    def __init__(self, dim: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        
    def forward(self, x: Data1D) -> Data1D:
        return F.normalize(x, dim=1) * self.g * math.sqrt(x.shape[1])
    
class PreNorm(nn.Module):
    "add pre-normalization to given callable"
    def __init__(self, dim: int, func: Callable):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.func = func
        
    def forward(self, x: Data1D) -> Data1D:
        return self.func(self.norm(x))
    
class SinusoidalPosEmb(nn.Module):
    """ for position t, dimension i of a d-dim vector, the embedding is
        1/(10000**(i/(d/2-1)))
        
        dim must be even. 
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, 'dimension must be even'
        self.dim = dim
        
    def forward(self, pos: Float[Tensor, 'batch, ']) -> Float[Tensor, 'batch, dim']:
        device = pos.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb) # shape: (dim/2,)
        emb = pos.unsqueeze(-1) * emb.unsqueeze(0) # shape: (batch, 1) * (1, dim/2) -> (batch, dim/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # shape: (batch, dim)
        return emb
    
class RandomSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, 'dimension must be even'
        half_dim = dim // 2 - 1
        self.weights = nn.Parameter(torch.randn(1, half_dim), requires_grad=False) # random
        
    def forward(self, pos: Float[Tensor, "batch, "]) -> Float[Tensor, "batch, dim"]:
        pos = rearrange(pos, 'b -> b 1') 
        freqs = pos * self.weights * 2 * math.pi    # shape: (batch, dim/2-1)
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1) # shape: (batch, dim-2)
        fouriered = torch.cat((pos, fouriered), dim=-1) # shape: (batch, dim-1)
        fouriered = torch.cat((-pos, fouriered), dim=-1) # shape: (batch, dim)
        return fouriered
    
class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, 'dimension must be even'
        half_dim = dim // 2 - 1
        self.weights = nn.Parameter(torch.randn(1, half_dim), requires_grad=True) # leranable
        
    def forward(self, pos: Float[Tensor, "batch, "]) -> Float[Tensor, "batch, dim"]:
        pos = rearrange(pos, 'b -> b 1') 
        freqs = pos * self.weights * 2 * math.pi    # shape: (batch, dim/2-1)
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1) # shape: (batch, dim-2)
        fouriered = torch.cat((pos, fouriered), dim=-1) # shape: (batch, dim-1)
        fouriered = torch.cat((-pos, fouriered), dim=-1) # shape: (batch, dim)
        return fouriered
    
# building blocks
class ResnetSubBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_group: Optional[int] = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim_in, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(num_group, dim_out) 
        self.activation = nn.SiLU()
        
    def forward(self, 
                x: Data1D, 
                scale_shift: Optional[tuple[Data1D, Data1D]] = None
                ) -> Data1D:
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (1+scale) + shift
        x = self.activation(x)
        return x
    
class ResnetBlock(nn.Module):
    """
    Residual block for 1D convolutional neural networks.
    
    Args:
        dim_in (int): Number of input channels.
        dim_out (int): Number of output channels.
        time_emb_dim (Optional[int]): Dimension of time embedding. Defaults to None.
            will be used to compute scale and shift using MLP. so dimension is can be different from dim_out.
        num_group (int): Number of groups for group normalization. Defaults to 8.
        
    Forward:
        Args:
            x (Data1D): Input tensor of shape (batch, dim_in, sequence).
            time_emb (Optional[Float[Tensor, 'batch time_emb_dim']]): Time embedding tensor of shape (batch, time_emb_dim).
        Returns:
            Data1D: Output tensor of shape (batch, dim_out, sequence).
        Process:
            1. If time_emb is given, use it to compute scale and shift.
            2. Apply two ResnetSubBlock sequentially.
            3. Add residual connection.
        
    """
    
    def __init__(self, dim_in: int, dim_out: int, dim_time_emb: Optional[int] = None, num_group = 8, dropout = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_time_emb, dim_out * 2)
        ) if dim_time_emb is not None else None
        
        self.block1 = ResnetSubBlock(dim_in, dim_out, num_group)
        self.block2 = ResnetSubBlock(dim_out, dim_out, num_group)
        # residual_conv: if dim_in == dim_out, use identity, else use 1x1 conv (=linear transformation)
        self.residual_conv = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        self.dropout =dropout
        
    def forward(self, x: Data1D, time_emb: Optional[Float[Tensor, 'batch dim_time_emb']]
                ) -> Data1D:
        # if time_emb is given, use it to compute scale and shift
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb) # shape: (batch, dim_out * 2)
            time_emb = rearrange(time_emb, 'b c -> b c 1') # shape: (batch, dim_out * 2, 1)
            scale_shift = time_emb.chunk(2, dim=1) # shape: (batch, dim_out, 1), (batch, dim_out, 1) 
        
        h = self.block1(x, scale_shift) # shape: (batch, dim_out, sequence)
        h = self.block2(h)              # shape: (batch, dim_out, sequence)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h + self.residual_conv(x)  # shape: (batch, dim_out, sequence)
    
class LinearSelfAttention(nn.Module):
    r"""(batch, dim, sequence) -> (batch, dim, sequence)
    
    See: https://openaccess.thecvf.com/content/WACV2021/html/Shen_Efficient_Attention_Attention_With_Linear_Complexities_WACV_2021_paper.html
    """
    def __init__(self, dim: int, num_head: int=4, dim_head: int|None=None):
        super().__init__()
        self.dim_inout = dim
        self.scale = dim_head ** -0.5
        self.num_head = num_head
        self.hidden_dim = dim_head * num_head
        self.to_qkv = nn.Conv1d(self.dim_inout, self.hidden_dim*3, 1, bias=True)
        
        self.to_out = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.dim_inout, 1),
            RMSNorm(self.dim_inout)
        )
        self.rms_norm_q = nn.Sequential(
            Rearrange('b num_head d l -> b (num_head d) l'),
            RMSNorm(self.hidden_dim), # in dim 1
            Rearrange('b (num_head d) l -> b num_head d l', num_head = self.num_head)
        )
        self.rms_norm_k = nn.Sequential(
            Rearrange('b num_head d l -> b (num_head d) l'),
            RMSNorm(self.hidden_dim), # in dim 1
            Rearrange('b (num_head d) l -> b num_head d l', num_head = self.num_head)
        )
        
    def forward(self, x: Data1D) -> Data1D:
        qkv = self.to_qkv(x).chunk(3, dim = 1) # 3 * (batch, hidden_dim, sequence)
        q, k, v = map(
            lambda t: rearrange(t, 'b (num_head d) l -> b num_head d l', num_head = self.num_head),
            qkv
        ) # shape: 3 * (batch, num_head, dim_head, sequence)
        
        # such normalization: one independent attention for each head
        q = self.rms_norm_q(q)
        k = self.rms_norm_k(k)
        # q = q.softmax(dim = -2) # over dim_head
        # k = k.softmax(dim = -1) # over sequence
        q = q * self.scale # scale
        
        sim = einsum(
            q, k,
            'b h dq l, b h dq n -> b h l n'
        )
        sim = sim.softmax(dim = -1) # over sequence
        # out = einsum(context, q, 'b h dq dv, b h dq l -> b h dv l')
        out = einsum(sim, v, 'b h l n, b h dv n -> b h dv l') # shape: (batch, num_head, dim_head, sequence)
        out = rearrange(out, 'b h dv l -> b (h dv) l') # shape: (batch, hidden_dim, sequence)
        return self.to_out(out) # shape: (batch, dim, sequence)
    
class WrappedLinearSelfAttention(nn.Module):
    " (b l c) -> (b l c) "
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.attention = LinearSelfAttention(*args, **kwargs)
        
    def forward(self, x: Float[Tensor, 'b l c']) -> Float[Tensor, 'b l c']:
        return rearrange(
            self.attention(
                rearrange(x, 'b l c -> b c l')
            ), # out shape: (b, c, l)
            'b c l -> b l c'
        )
    
class Attention(nn.Module):
    """(batch, dim, sequence) -> (batch, dim, sequence) 
    difference with LinearSelfAttention:
        1. LinearSelfAttention has a RMSNorm after the output, while Attention does not.
        2. LinearSelfAttention normalizes over heads and sequence, while Attention normalizes over sequence only.
    """
    def __init__(self, dim = int, num_head: int=4, dim_head: int=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.num_head = num_head
        hidden_dim = dim_head * num_head
        
        self.to_qkv = nn.Conv1d(dim, hidden_dim*3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)
        
    def forward(self, x: Data1D) -> Data1D:
        b, c, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (num_head d) l -> b num_head d l', num_head=self.num_head),
            qkv
        )
        
        q = q * self.scale
        similarity = einsum(q, k, 'b h d lq, b h d lk -> b h lq lk')
        similarity = similarity.softmax(dim = -1) # over key/value
        out = einsum(similarity, v, 'b h lq lk, b h d lk -> b h lq d')
        out = rearrange(out, 'b h lq d -> b (h d) lq') 
        
        return self.to_out(out)
    
class SelfAttention(nn.Module):
    """(batch, sequence, dim) -> (batch, sequence, dim) 
    difference with LinearSelfAttention:
        1. LinearSelfAttention has a RMSNorm after the output, while Attention does not.
        2. LinearSelfAttention normalizes over heads and sequence, while Attention normalizes over sequence only.
    """
    def __init__(self, dim: int, num_head: int, dim_head: int):
        super().__init__()
        self.dim_inout = dim
        self.scale = dim_head ** -0.5
        self.num_head = num_head
        hidden_dim = dim_head * num_head
        
        self.to_qkv = nn.Linear(dim, hidden_dim*3, bias=True)
        self.to_out = nn.Linear(hidden_dim, dim)
        
    def forward(self, x: Float[Tensor, 'b l c']) -> Float[Tensor, 'b l c']:
        b, l, c = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 2)
        q, k, v = map(
            lambda t: rearrange(t, 'b l (num_head d) -> b num_head l d', num_head=self.num_head),
            qkv
        )
        
        q = q * self.scale
        similarity = einsum(q, k, 'b h lq d, b h lk d -> b h lq lk')
        similarity = similarity.softmax(dim = -1) # over key/value
        out = einsum(similarity, v, 'b h lq lk, b h lk d -> b h lq d')
        # out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h lq d -> b lq (h d)') 
        
        return self.to_out(out)
    
# backbone modules
class Unet1D(nn.Module):
    """
    UNet1D
    
    processing condition: add the embedded condition to time embedding. 
    """
    def __init__(
        self,
        dim_base: int, # base dim for conv, not dim of input
        dim_out: int|None = None, # output dim for conv, final output
        dim_mult: Sequence[int] = (1, 2, 4, 8), # multiplier for each resolution
        num_in_channel: int = 3,
        self_condition: bool = False,
        num_resnet_block_group: int = 8,
        learn_variance: bool = False,
        type_pos_emb: str = 'sinusoidal',
        dim_learned_pos_emb: int = 16,
        # dim_attn_head: int = 32,
        num_attn_head: int = 4,
        dropout: float = 0.1,
        conditioning: bool = False,
        cond_embedder: Optional[EmbedderWrapper] = None,
    ):
        super().__init__()
        assert type_pos_emb in {'sinusoidal', 'learned', 'random'}, \
            'positional embedding type must be one of sinusoidal, learned, random'
        self.type_pos_emb = type_pos_emb
        self.dim_base = dim_base # dimensions
        self.conditioning = conditioning
        self.dropout = dropout
        
        # dimensions
        self.num_in_channel = num_in_channel
        self.self_condition = self_condition
        num_in_channel_init_conv = num_in_channel * (2 if self_condition else 1)
        
        self.init_conv = nn.Conv1d(
            num_in_channel_init_conv, dim_base, 7, padding=3
        ) # (batch, num_in_channel, sequence) -> (batch, dim_base, sequence)
        self.dim_out = default(
            dim_out,
            num_in_channel * (1 if not learn_variance else 2)
        )
        
        list_dim = [*map(lambda t: dim_base * t, dim_mult)] # base * multipliers
        list_dim_in_out = list(zip(list_dim[:-1], list_dim[1:])) # (dim_in, dim_out) for each resolution
        
        _ResnetBlock = partial(ResnetBlock, num_group = num_resnet_block_group, dropout = self.dropout)
        
        # time embedding
        #       time embedding is basically a vector f(t). 
        #       it will be used by ResnetBlocks to compute scale and shift.
        #       so the dimension does not have to be the same as model dimension.
        dim_time_emb = dim_base * 4 # default
        
        if self.type_pos_emb == 'learned':
            pos_emb = LearnedSinusoidalPosEmb(dim_learned_pos_emb)
            # dim_fourier: dimension of fourier embedding.
            dim_fourier = dim_learned_pos_emb # customized dimension
        elif self.type_pos_emb == 'random':
            pos_emb = RandomSinusoidalPosEmb(dim_learned_pos_emb)
            dim_fourier = dim_learned_pos_emb
        else:
            pos_emb = SinusoidalPosEmb(dim_base)
            dim_fourier = dim_base # same as model dimension
            
        """ computes a vector based on time t, used by ResnetBlocks to compute scale and shift
            time_mlp: convert a time t to a vector of dimension dim_time_emb.
                - input: time t, shape: (batch, )
                - output: vector f(t), shape: (batch, dim_time_emb)
                - steps:
                    - pos_emb, shape: (batch, dim_fourier)
                    - mlp, shape: (batch, dim_time_emb)
        """
        self.time_mlp = nn.Sequential(
            pos_emb,
            nn.Linear(dim_fourier, dim_time_emb),
            nn.GELU(),
            nn.Linear(dim_time_emb, dim_time_emb),
        )
        if self.conditioning:
            self.cond_embedder = cond_embedder
            self.post_cond_embedder = nn.Sequential(
                nn.SiLU(),
                nn.Conv1d(dim_base, dim_time_emb, 1),
                Rearrange('b c l -> b (c l)') # (batch, dim_time_emb, 1) -> (batch, dim_time_emb)
            )
        
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_layer = len(list_dim_in_out) # one layer = multiple sub-layer
        
        for idx, (d_in, d_out) in enumerate(list_dim_in_out):
            is_last = idx == (num_layer - 1)
            
            # downsample
            self.downs.append(
                nn.ModuleList([
                    _ResnetBlock(d_in, d_in, dim_time_emb=dim_time_emb), # it has residual connection
                    _ResnetBlock(d_in, d_in, dim_time_emb=dim_time_emb),
                    Residual(PreNorm(d_in, LinearSelfAttention(d_in, num_head=num_attn_head, dim_head=d_in//num_attn_head))),
                    # if not last, downsample, else use 3x3 conv (same dimension)
                    Downsample(d_in, d_out) if not is_last else nn.Conv1d(d_in, d_out, 3, padding=1)
                ])
            ) # (batch, d_in, sequence_in) -> (batch, d_out, sequence_out)
            
        mid_dim = list_dim[-1] # = last d_out
        self.mid_block1 = _ResnetBlock(mid_dim, mid_dim, dim_time_emb=dim_time_emb)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, num_head=num_attn_head, dim_head=mid_dim//num_attn_head)))
        self.mid_block2 = _ResnetBlock(mid_dim, mid_dim, dim_time_emb=dim_time_emb)
        
        # upsample
        for idx, (d_in, d_out) in enumerate(reversed(list_dim_in_out)):
            is_last = idx == (num_layer - 1)
            
            self.ups.append(
                nn.ModuleList([
                    _ResnetBlock(d_in+d_out, d_out, dim_time_emb=dim_time_emb),
                    _ResnetBlock(d_in+d_out, d_out, dim_time_emb=dim_time_emb),
                    Residual(PreNorm(d_out, LinearSelfAttention(d_out, num_head=num_attn_head, dim_head=d_out//num_attn_head))),
                    Upsample(d_out, d_in) if not is_last else nn.Conv1d(d_out, d_in, 3, padding=1)
                ])
            )
            
        # conditioning
        #   pre-final adaln modulation
        #   !not necessary. modulation will be done in the resnet block. 
            
        self.final_resnet_block = _ResnetBlock(
            dim_base*2, dim_base, dim_time_emb=dim_time_emb
        ) # connect to the input of init_conv
        self.final_conv = nn.Conv1d(dim_base, self.dim_out, 1) # 1x1 conv
        
    def forward(
        self,
        x: Data1D,
        time: Float[Tensor, 'batch, '],
        c: Optional[Float[Tensor, 'batch num_cond_channel 1']] = None,
        force_drop_ids: Optional[Float[Tensor, 'batch']] = None,
        x_self_cond: Optional[Data1D] = None,
    ) -> Data1D:
        # generate embedding for time and condition
        encoded_t = self.time_mlp(time) # shape: (batch, dim_time_emb)
        if self.conditioning and c is not None:
            encoded_c = self.cond_embedder(c, force_drop_ids=force_drop_ids)
            encoded_c = self.post_cond_embedder(encoded_c) # shape: (batch, dim_time_emb)
        elif self.conditioning and c is None:
            " if c is not given, then seen as force drop all conditions"
            c = torch.randn((x.shape[0], self.num_cond_channel, 1), device=x.device, dtype=x.dtype)
                # shape: (batch, num_cond_channel, 1)
            force_drop_ids = torch.ones((x.shape[0],), device=x.device,
                                        dtype=torch.long) # shape: (batch, )
            encoded_c = self.cond_embedder(c, force_drop_ids=force_drop_ids) # shape: (batch, dim_base, sequence)
        else:
            encoded_c = torch.zeros_like(encoded_t) # shape: (batch, dim_time_emb)
        encoded_tc = encoded_t + encoded_c # shape: (batch, dim_time_emb)
        
        if self.self_condition:
            if x_self_cond is None:
                warnings.warn('self_condition is True, but x_self_cond is None, using full zeros.')
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x, x_self_cond), dim=1) # shape: (batch, num_in_channel * 2, sequence)
        
        # convert from input channels into base channels (dim_base)
        x = self.init_conv(x) # shape: (batch, num_in_channel, sequence) -> (batch, dim_base, sequence)
        x_copy = x.clone()
        
        h = [] # list of feature maps
        
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, encoded_tc) # shape: (batch, dim_in, sequence)
            h.append(x)
            
            x = block2(x, encoded_tc) # shape: (batch, dim_in, sequence)
            x = attn(x)
            h.append(x)
            
            # last downsample does not "downsample"
            x = downsample(x) # shape: (batch, dim_out, sequence//2)
            
        x = self.mid_block1(x, encoded_tc)
        x = self.mid_attn(x)
        x = self.mid_block2(x, encoded_tc)
        
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1) # shape: (batch, dim_in + dim_out, sequence//2)
            x = block1(x, encoded_tc) # shape: (batch, dim_out, sequence//2)
            
            x = torch.cat((x, h.pop()), dim=1) # shape: (batch, dim_in + dim_out, sequence//2)
            x = block2(x, encoded_tc) # shape: (batch, dim_out, sequence//2)
            x = attn(x) # shape: (batch, dim_out, sequence//2)
            
            # last upsample does not "upsample"
            x = upsample(x) # shape: (batch, dim_in, sequence)
            
        # TODO: this concatenation is unexpected. 
        x = torch.cat((x, x_copy), dim=1) 
        x = self.final_resnet_block(x, encoded_tc)
        #   shape: (batch, dim_base + dim_base, sequence)
        #           -> (batch, dim_base, sequence)
        x = self.final_conv(x) # shape: (batch, dim_out, sequence)
        
        return x
    
    def forward_with_cfg(
        self,
        x: Float[Tensor, 'batch num_in_channel sequence'], 
        time: Float[Tensor, 'batch, '], 
        c: Optional[Float[Tensor, 'batch num_cond_channel 1']] = None, 
        x_self_cond: Optional[Data1D] = None,
        cfg_scale:float=1.
    ) -> Data1D: 
        """ forward with classfier-free guidance (cfg)
        
        Inputs:
            - x: (batch, num_in_channel, sequence)
            - time: (batch, )
            - c: None | (batch, num_cond_channel, 1)
            - x_self_cond: None | (batch, num_in_channel, sequence)
            - cfg_scale: float, range [1., +inf), default 1. = no guidance. 
                            this scale = 1 + w, the w is as in the paper.
        """
        if c is None or cfg_scale == 1.:
            return self.forward(x, time, c, x_self_cond)
        
        _x = torch.cat([x, x], dim=0) # shape: (batch*2, num_in_channel, sequence)
        _time = torch.cat([time, time], dim=0) # shape: (batch*2, )
        _c = torch.cat([c, c], dim=0) # shape: (batch*2, num_cond_channel, 1)
        if x_self_cond is not None:
            _x_self_cond = torch.cat([x_self_cond, x_self_cond], dim=0)
        else:
            _x_self_cond = None
        force_drop_ids_1 = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long) # no drop
        force_drop_ids_2 = torch.ones((x.shape[0],), device=x.device, dtype=torch.long) # drop all
        force_drop_ids = torch.cat([force_drop_ids_1, force_drop_ids_2], dim=0) # shape: (batch*2, )
        # shape: (batch*2, num_cond_channel, sequence), cond + uncond
        cond_x, uncond_x = self.forward(_x, _time, _c, force_drop_ids=force_drop_ids, x_self_cond=_x_self_cond).chunk(2, dim=0)
        scaled_x = uncond_x + cfg_scale * (cond_x - uncond_x)
        
        return scaled_x
    
class Transformer1D(nn.Module):
    """ transformer for 1D data
    Data1D -> Data1D
    
    Arguments: 
        - dim_base: d_model of transformer, recommended value = ?
        - ...
        - type_transformer: transformer or gpt2
        - conditioning: whether to use conditioning, the condition tensor 'c' should have same dimension as `dim_base`
        
    Forward:
        input:
            - x: (batch, num_in_channel, sequence)
            - time: (batch, )
            - x_self_cond: None|(batch, num_in_channel, sequence)
        
        return:
            - x: (batch, dim_out, sequence)
    
    """
    MAX_SEQ_RESOLUTION = 8640 # 24 * 60 * 6, 10 seconds per step. used to interpolate time embedding. 
                            # theoretically can also accept resolution > 8640. 
    def __init__(
        self,
        dim_base: int,
        num_in_channel: int = 3,
        dim_out: int|None = None,
        self_condition: bool = False,
        type_pos_emb: str = 'sinusoidal',
        # dim_learned_pos_emb: int = 16,
        num_attn_head: int = 4,
        num_encoder_layer = 6,
        num_decoder_layer = 6,
        dim_feedforward = 2048, # why so big by default?
        dropout = 0.1,
        learn_variance: bool = False,
        conditioning: bool = False,
        cond_embedder: Optional[EmbedderWrapper] = None,
    ):
        super().__init__()
        self.dim_base = dim_base
        self.num_in_channel = num_in_channel
        self.self_condition = self_condition
        self.type_pos_emb = type_pos_emb
        self.num_atten_head = num_attn_head
        self.num_encoder_layer = num_encoder_layer
        self.num_decoder_layer = num_decoder_layer
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.conditioning = conditioning
        self.cond_embedder = cond_embedder
            # (batch, num_condition_channel, sequence)
            #   -> (batch, dim_base, sequence)
        self.num_cond_channel = sum(self.cond_embedder.list_dim_cond) if conditioning else None
        
        self.dim_out = default(dim_out, num_in_channel * (1 if not learn_variance else 2))
        
        # self.type_transformer = type_transformer
        # assert self.type_transformer in {'gpt2'}, 'type_transformer must be one of "gpt2"'
        
        # pos_emb for time
        #   (batch, ) -> (batch, dim_base)
        if self.type_pos_emb == 'learned':
            pos_emb = LearnedSinusoidalPosEmb(self.dim_base)
        elif self.type_pos_emb == 'random':
            pos_emb = RandomSinusoidalPosEmb(self.dim_base)
        else:
            pos_emb = SinusoidalPosEmb(self.dim_base)
        
        self.time_mlp = nn.Sequential(
            pos_emb,
            nn.Linear(self.dim_base, self.dim_base),
            nn.GELU(),
            nn.Linear(self.dim_base, self.dim_base * 2),
        )
        
        # pos_emb for transformer
        #   use sinusoidal as the only option for now
        _transformer_pos_emb = SinusoidalPosEmb(self.dim_base)
        self.transformer_pos_emb = nn.Sequential(
            _transformer_pos_emb,
            nn.Linear(self.dim_base, self.dim_base),
            nn.GELU(),
            nn.Linear(self.dim_base, self.dim_base),
        )
        
        # init proj 
        #   (batch, dim_base) -> (batch, dim_base * 2)
        self.init_proj = nn.Conv1d(num_in_channel if not self.self_condition else num_in_channel * 2, 
                                   dim_base, 5, padding=2)
        #   (batch, num_in_channel, sequence) -> (batch, dim_base, sequence)
        
        # transformer
        # if self.type_transformer == 'transformer':
        #     self.transformer = nn.Transformer(
        #         d_model = self.dim_base,
        #         nhead = self.num_atten_head,
        #         num_encoder_layers = self.num_encoder_layer,
        #         num_decoder_layers = self.num_decoder_layer,
        #         dim_feedforward = self.dim_feedforward,
        #         dropout = self.dropout,
        #         batch_first= True, # (batch, sequence, channel)
        #     )
        # else:
        self.transformer = GPT2Model(
            dim_base = self.dim_base,
            num_attn_head = self.num_atten_head,
            num_layer = self.num_decoder_layer,
            dim_feedforward=self.dim_feedforward,
            dropout = self.dropout,
            conditioning = self.conditioning,
        )
        
        # conditioning 
        #   pre-final adaln modulation
        if self.conditioning:
            self.final_ln = nn.Sequential(
                Rearrange('batch channel sequence -> batch sequence channel'),
                nn.LayerNorm(self.dim_base, elementwise_affine=False, eps=1e-6),
                Rearrange('batch sequence channel -> batch channel sequence'),
            )
            self.final_adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Conv1d(self.dim_base, self.dim_base * 2, 1), # linear layer
            ) # operate on the second dimension, so using conv1d instead of linear
        
        # final conv
        self.final_conv = nn.Conv1d(self.dim_base * 2, self.dim_out, 1)
        
        # initialize weights
        self.initialize_weights()
        
    def freeze_layers(self):
        "freeze all layers except init and final conv. "
        self.time_mlp.requires_grad_(False)
        self.transformer_pos_emb.requires_grad_(False)
        self.init_proj.requires_grad_(True) # True
        self.transformer.requires_grad_(False)
        if self.conditioning:
            self.final_ln.requires_grad_(False)
            self.final_adaLN_modulation.requires_grad_(False)
        for name, module in self.named_modules():
            # if the module name contains 'adaLN_modulation', then unfreeze it
            if 'adaLN_modulation' in name:
                module.requires_grad_(True)
        self.final_conv.requires_grad_(True) # True
        
    def initialize_weights(self):
        # initialize all transformer layers
        def _basic_init(module):
            if getattr(module, 'weight', None) is not None and not isinstance(module, nn.LayerNorm):
                nn.init.xavier_uniform_(module.weight)
            if getattr(module, 'bias', None) is not None:
                nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # zero-out final conv layer
        nn.init.constant_(self.final_conv.weight, 0)
        nn.init.constant_(self.final_conv.bias, 0)
        
        if not self.conditioning:
            return # normal LayerNorm is already initialized as gamma=1, beta=0
        
        # zero-out adaLN modulation layers in GPT2 blocks
        for decoder in self.transformer.decoders:
            # this layer: c -> 2 * (scale, shift,gate)
            nn.init.constant_(decoder.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(decoder.adaLN_modulation[-1].bias, 0)
        
        # zero-out final adaLN modulation layer
        nn.init.constant_(self.final_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_adaLN_modulation[-1].bias, 0)
        
    def forward(
        self,
        x: Float[Tensor, 'batch num_in_channel sequence'], 
        time: Float[Tensor, 'batch, '], 
        c: Optional[Float[Tensor, 'batch num_cond_channel 1']] = None, 
        force_drop_ids: Optional[Float[Tensor, 'batch']] = None,
        x_self_cond: Optional[Data1D] = None
    ) -> Data1D:
        # generate embedding for time
        encoded_t = self.time_mlp(time) # shape: (batch, dim_base * 2)
        encoded_t = rearrange(encoded_t, 'b d -> b d 1') # shape: (batch, dim_base * 2, 1)
        scale, shift = encoded_t.chunk(2, dim=1) # shape: (batch, dim_base, 1), (batch, dim_base, 1)
        
        if self.conditioning and c is not None:
            encoded_c = self.cond_embedder(c, force_drop_ids=force_drop_ids) # shape: (batch, dim_base, sequence)
        elif self.conditioning and c is None:
            " if c is not given, then seen as force drop all conditions"
            c = torch.randn((x.shape[0], self.num_cond_channel, 1), device=x.device, dtype=x.dtype)
                # shape: (batch, num_cond_channel, 1)
            force_drop_ids = torch.ones((x.shape[0],), device=x.device,
                                        dtype=torch.long) # shape: (batch, )
            encoded_c = self.cond_embedder(c, force_drop_ids=force_drop_ids) # shape: (batch, dim_base, sequence)
        else:
            encoded_c = None
        
        # init conv
        if self.self_condition:
            if x_self_cond is None:
                warnings.warn('self_condition is True, but x_self_cond is None, using full zeros.')
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x, x_self_cond), dim=1) # shape: (batch, num_in_channel * 2, sequence)
        x = self.init_proj(x) # shape: (batch, dim_base, sequence)
        x_copy = x.clone()
        x = x * (1 + scale) + shift # shape: (batch, dim_base, sequence)
        x = F.silu(x) # shape: (batch, dim_base, sequence)
        
        # transformer
        x = rearrange(x, 'batch channel sequence -> batch sequence channel')
        encoded_c = rearrange(encoded_c, 'batch channel sequence -> batch sequence channel') if encoded_c is not None else None
        # pos_seq = torch.arange(x.shape[1], device=x.device, dtype=x.dtype) # shape: (sequence, )
        pos_seq = torch.arange(self.MAX_SEQ_RESOLUTION, device=x.device, dtype=x.dtype) # shape: (sequence, )
        if x.shape[1] < self.MAX_SEQ_RESOLUTION:
            pos_seq = F.interpolate(rearrange(pos_seq, 'seq -> 1 1 seq'), size=(x.shape[1],), mode='linear') # shape: (1, 1, sequence)
            pos_seq = rearrange(pos_seq, '1 1 seq -> seq') # shape: (sequence, )
        pos_emb_seq = self.transformer_pos_emb(pos_seq) # shape: (sequence, dim_base)
        pos_emb_seq = rearrange(pos_emb_seq, 'sequence channel -> 1 sequence channel') # add batch dim, shape: (1, sequence, dim_base)
        x = x + pos_emb_seq # sequence pos emb, shape: (batch, sequence, dim_base)
        
        # if self.type_transformer == 'transformer':
        #     tgt_causal_mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).to(x.dtype).to(x.device)
        #     x = self.transformer(x, x, 
        #                         tgt_mask=tgt_causal_mask,
        #                         tgt_is_causal=True
        #                      ) # shape: (batch, sequence, channel)
        # elif self.type_transformer == 'gpt2':
        #     x = self.transformer(x, encoded_c) # shape: (batch, sequence, channel)
        # else:
        #     raise RuntimeError('type_transformer must be one of transformer, gpt2')
        x = self.transformer(x, encoded_c) # shape: (batch, sequence, channel)
        
        x = rearrange(x, 'batch sequence channel -> batch channel sequence')
        encoded_c = rearrange(encoded_c, 'batch sequence channel -> batch channel sequence') if encoded_c is not None else None
        
        # final adaln modulation
        if self.conditioning and encoded_c is not None:
            scale_final, shift_final = self.final_adaLN_modulation(encoded_c).chunk(2, dim=1) 
                # shape: (batch, dim_base, sequence), (batch, dim_base, sequence)
            x = self.final_ln(x) # shape: (batch, dim_base, sequence)
            x = x * (1. + scale_final) + shift_final # shape: (batch, dim_base, sequence)
        
        # final linear/conv layer
        # skip connection: discard in exp_id 2.1.0; readopted after 2.3.0 / 1.2.0 
        x = torch.concat((x, x_copy), dim=1) # shape: (batch, dim_base*2, sequence)
        x = self.final_conv(x) # shape: (batch, dim_out, sequence)
        
        # NOTE: compared with UNet1D, there is no final resnet block
        
        return x
    
    def forward_with_cfg(
        self,
        x: Float[Tensor, 'batch num_in_channel sequence'], 
        time: Float[Tensor, 'batch, '], 
        c: Optional[Float[Tensor, 'batch num_cond_channel 1']] = None, 
        x_self_cond: Optional[Data1D] = None,
        cfg_scale:float=1.
    ) -> Data1D: 
        """ forward with classfier-free guidance (cfg)
        
        Inputs:
            - x: (batch, num_in_channel, sequence)
            - time: (batch, )
            - c: None | (batch, num_cond_channel, 1)
            - x_self_cond: None | (batch, num_in_channel, sequence)
            - cfg_scale: float, range [1., +inf), default 1. = no guidance. 
                            this scale = 1 + w, the w is as in the paper.
        """
        if c is None or cfg_scale == 1.:
            return self.forward(x, time, c, x_self_cond)
        
        _x = torch.cat([x, x], dim=0) # shape: (batch*2, num_in_channel, sequence)
        _time = torch.cat([time, time], dim=0) # shape: (batch*2, )
        _c = torch.cat([c, c], dim=0) # shape: (batch*2, num_cond_channel, 1)
        if x_self_cond is not None:
            _x_self_cond = torch.cat([x_self_cond, x_self_cond], dim=0)
        else:
            _x_self_cond = None
        force_drop_ids_1 = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long) # no drop
        force_drop_ids_2 = torch.ones((x.shape[0],), device=x.device, dtype=torch.long) # drop all
        force_drop_ids = torch.cat([force_drop_ids_1, force_drop_ids_2], dim=0) # shape: (batch*2, )
        # shape: (batch*2, num_cond_channel, sequence), cond + uncond
        cond_x, uncond_x = self.forward(_x, _time, _c, force_drop_ids=force_drop_ids, x_self_cond=_x_self_cond).chunk(2, dim=0)
        scaled_x = uncond_x + cfg_scale * (cond_x - uncond_x)
        
        return scaled_x
    
class GPT2Block(nn.Module):
    """ GPT2 block for 1D data 
    
    Forward:
        input:
            - x: (batch, sequence, channel)
        
        return:
            - x: (batch, sequence, channel)
            
    """
    def __init__(
        self,
        dim_base: int, # model dimension
        num_attn_head: int = 4,
        dim_head: None|int = None,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        conditioning: bool = False
    ):
        super().__init__()
        self.dim_base = dim_base
        self.num_attn_head = num_attn_head
        self.dim_head = default(dim_head, dim_base // num_attn_head)
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.conditioning = conditioning
        
        self.ln_1 = nn.LayerNorm(dim_base, elementwise_affine=not self.conditioning, eps=1e-6)
        self.attn = WrappedLinearSelfAttention(dim_base, num_head=self.num_attn_head, dim_head=dim_base//self.num_attn_head)
        self.ln_2 = nn.LayerNorm(dim_base, elementwise_affine=not self.conditioning, eps=1e-6)
        
        
        # no cross attention compared with the GPT2
        self.mlp = nn.Sequential(
            nn.Linear(dim_base, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, dim_base),
            nn.Dropout(self.dropout)
        )
        
        # conditioning
        if self.conditioning:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim_base, dim_base * 6),
            )
        
    def forward(
            self, 
            x: Float[Tensor, 'batch sequence channel'], 
            c: None|Float[Tensor, 'batch 1 channel'] = None
        ) -> Float[Tensor, 'batch sequence channel']:
        if c is not None and self.conditioning:
            cond_scale_shift_gate = self.adaLN_modulation(c) # shape: (batch, 1, channel * 6)
            scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp = cond_scale_shift_gate.chunk(6, dim=2)
        else:
            scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp = [0., 0., 1., 0., 0., 1.]
            
        # attn
        x_copy = x.clone()
        x = self.ln_1(x)
        x = x * (1. + scale_attn) + shift_attn # scale shift from adaLN
        attn_output = self.attn(x) # shape: (batch, sequence, channel)
        x = x_copy + attn_output * gate_attn  # residual connection
        
        # mlp / feedforward
        x_copy = x.clone()
        x = self.ln_2(x)
        x = x * (1. + scale_mlp) + shift_mlp # scale shift from adaLN
        x = self.mlp(x)
        x = x_copy + x * gate_mlp # residual connection
        
        return x # shape: (batch, sequence, channel)
        

class GPT2Model(nn.Module):
    """ base GPT2 model 
    
    Forward:
        input:
            - x: (batch, sequence, channel)
        
        return:
            - x: (batch, sequence, channel)
    """
    def __init__(
        self,
        dim_base: int,
        num_attn_head: int = 4,
        num_layer = 6,
        dim_feedforward = 2048, # why so big by default?
        dropout = 0.1,
        conditioning: bool = False
    ):
        super().__init__()
        self.dim_base = dim_base
        self.num_attn_head = num_attn_head
        self.num_layer = num_layer
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.conditioning = conditioning
        
        # define model layers here
        self.decoders = nn.ModuleList([])
        for idx in range(self.num_layer):
            self.decoders.append(
                GPT2Block(
                    dim_base = self.dim_base,
                    num_attn_head = self.num_attn_head,
                    dim_feedforward = self.dim_feedforward,
                    dropout = self.dropout,
                    conditioning = self.conditioning
                ) # shape: (batch, dim_base, channel) -> (batch, dim_base, channel)
            )
        
    def forward(
        self, 
        x: Float[Tensor, 'batch sequence channel'],
        c: None|Float[Tensor, 'batch sequence channel'] = None
    ) -> Float[Tensor, 'batch sequence channel']:
        for decoder in self.decoders:
            x = decoder(x, c)
        return x # shape: (batch, sequence, channel)
    
class MLPBlock(nn.Module):
    " layernorm -> linear -> silu -> linear (dropout) -> (gate) " 
    def __init__(
        self,
        dim_base: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        conditioning: bool = False,
    ):
        super().__init__()
        self.dim_base = dim_base
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.conditioning = conditioning
        
        self.ln_1 = nn.LayerNorm(dim_base, elementwise_affine=not self.conditioning, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim_base, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, dim_base),
            nn.Dropout(self.dropout)
        )
        
        # conditioning
        if self.conditioning:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim_base, dim_base * 3),
            )
            
    def forward(
        self,
        x: Float[Tensor, 'batch channel'],
        c: None|Float[Tensor, 'batch channel'] = None
    ) -> Float[Tensor, 'batch channel']:
        if c is not None and self.conditioning:
            cond_scale_shift_gate = self.adaLN_modulation(c) # shape: (batch, channel * 3)
            scale, shift, gate = cond_scale_shift_gate.chunk(3, dim=1)
        else:
            scale, shift, gate = [0., 0., 1.] # no conditioning
        
        x_copy = x.clone() # can also just assign, since we do not use in-place operation
        x = self.ln_1(x)
        x = x * (1. + scale) + shift # scale shift from adaLN
        x = self.mlp(x)
        x = x_copy + x * gate # residual connection
        
        return x
            
    
class DenoisingMLP1D(nn.Module):
    """ denoising MLP for 1D data

    Arguments:
        - dim_base: d_model of transformer, recommended value = ?
        - ...
        - conditioning: whether to use conditioning, the condition tensor 'c' should have same dimension as `dim_base`
    Forward:
        input:
            - x: (batch, num_in_channel, sequence)
            - time: (batch, )
            - c: None|(batch, num_cond_channel, 1)
            - x_self_cond: None|(batch, num_in_channel, sequence)
            - cfg_scale: float
        
        return:
            - x: (batch, dim_out, sequence)
    """
    def __init__(
        self, 
        dim_base: int,
        seq_length: int,
        num_in_channel: int = 3,
        self_condition: bool = False,
        dim_feedforward: int = 2048,
        num_block: int = 6,
        type_pos_emb: str = 'sinusoidal',
        dropout = 0.1,
        learn_variance: bool = False,
        conditioning: bool = False,
        cond_embedder: Optional[EmbedderWrapper] = None,
    ):
        super().__init__()
        self.dim_base = dim_base
        self.seq_length = seq_length
        self.num_in_channel = num_in_channel
        self.self_condition = self_condition
        self.dim_feedforward = dim_feedforward
        assert num_block >= 1, 'num_layer must be >= 1'
        self.num_block = num_block
        self.type_pos_emb = type_pos_emb
        self.dropout = dropout
        self.learn_variance = learn_variance
        self.conditioning = conditioning
        self.cond_embedder = cond_embedder
            # (batch, num_condition_channel, sequence)
            #   -> (batch, dim_base, sequence)
        self.dim_out = num_in_channel * (1 if not learn_variance else 2)
        
        Linear_Op = partial(nn.Conv1d, kernel_size=1, padding=0, stride=1, bias=True)
        
        if self.type_pos_emb == 'learned':
            pos_emb = LearnedSinusoidalPosEmb(self.dim_base)
        elif self.type_pos_emb == 'random':
            pos_emb = RandomSinusoidalPosEmb(self.dim_base)
        else:
            pos_emb = SinusoidalPosEmb(self.dim_base)
        self.time_mlp = nn.Sequential(
            pos_emb, # (batch, ) -> (batch, dim_base)
            nn.Linear(self.dim_base, self.dim_base),
            nn.GELU(),
            nn.Linear(self.dim_base, self.dim_base * 2),
        )
        
        self.init_proj = Linear_Op(seq_length * num_in_channel if not self.self_condition else num_in_channel * 2, dim_base)
        self.final_proj = Linear_Op(dim_base * 2, seq_length * self.dim_out)
        
        self.hidden_mlp = nn.ModuleList([])
        for idx in range(0, self.num_block):
            self.hidden_mlp.append(
                MLPBlock(
                    dim_base=self.dim_base,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout,
                    conditioning=self.conditioning
                )
            )
            
    def forward(
        self,
        x: Float[Tensor, 'batch num_in_channel sequence'],
        time: Float[Tensor, 'batch, '],
        c: Optional[Float[Tensor, 'batch num_cond_channel 1']] = None,
        force_drop_ids: Optional[Float[Tensor, 'batch']] = None,
        x_self_cond: Optional[Data1D] = None,
    ) -> Data1D:
        # reshape for MLP
        # num_in_channel = x.shape[1]
        # mid_dim = x.shape[1] // 2 # eliminate vectorize 
        # x = x[:, mid_dim:mid_dim+1, :] # shape: (batch, 1, sequence)
        x = rearrange(x, 'batch channel sequence -> batch (channel sequence) 1')
        
        # encode timestep
        encoded_t = self.time_mlp(time) # shape: (batch, dim_base * 2)
        encoded_t = rearrange(encoded_t, 'b d -> b d 1') # shape: (batch, dim_base * 2, 1)
        scale, shift = encoded_t.chunk(2, dim=1) # shape: (batch, dim_base, 1), (batch, dim_base, 1)
        
        # encode condition
        if self.conditioning and c is not None:
            encoded_c = self.cond_embedder(c, force_drop_ids=force_drop_ids)
        elif self.conditioning and c is None:
            " if c is not given, then seen as force drop all conditions"
            c = torch.randn((x.shape[0], self.num_cond_channel, 1), device=x.device, dtype=x.dtype)
                # shape: (batch, num_cond_channel, 1)
            force_drop_ids = torch.ones((x.shape[0],), device=x.device,
                                        dtype=torch.long) # shape: (batch, )
            encoded_c = self.cond_embedder(c, force_drop_ids=force_drop_ids) # shape: (batch, dim_base, sequence)
        else:
            encoded_c = None
            
        # init proj
        if self.self_condition:
            if x_self_cond is None:
                warnings.warn('self_condition is True, but x_self_cond is None, using full zeros.')
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x, x_self_cond), dim=1)
        x = self.init_proj(x) # shape: (batch, dim_base, 1)
        x_copy = x.clone() 
        x = x * (1 + scale) + shift
        x = F.silu(x) # shape: (batch, dim_base, 1)
        
        # hidden mlp layers
        x = rearrange(x, 'batch channel sequence -> batch (channel sequence)') # shape: (batch, dim_base)
        encoded_c = rearrange(encoded_c, 'batch channel sequence -> batch (channel sequence)') if encoded_c is not None else None
        for hidden_mlp in self.hidden_mlp:
            x = hidden_mlp(x, encoded_c) # shape: (batch, dim_base)
            
        x = rearrange(x, 'batch (channel sequence) -> batch channel sequence', channel=self.dim_base) # shape: (batch, dim_base, 1)
        encoded_c = rearrange(encoded_c, 'batch (channel sequence) -> batch channel sequence', channel=self.dim_base) if encoded_c is not None else None
        
        # skipping the final adaLN_modulation
        # final proj
        x = torch.concat([x, x_copy], dim=1) # shape: (batch, dim_base * 2, 1)
        x = self.final_proj(x) # shape: (batch, dim_out, 1)
        x = rearrange(x, 'batch (channel sequence) 1 -> batch channel sequence', channel=self.dim_out) # shape: (batch, dim_out, 1)
        
        return x # shape: (batch, dim_out, 1)
    
    def forward_with_cfg(
        self,
        x: Float[Tensor, 'batch num_in_channel sequence'],
        time: Float[Tensor, 'batch, '],
        c: Optional[Float[Tensor, 'batch num_cond_channel 1']] = None,
        x_self_cond: Optional[Data1D] = None,
        cfg_scale:float=1.
    ):
        if c is None or cfg_scale == 1.:
            return self.forward(x, time, c, x_self_cond)
        
        _x = torch.cat([x, x], dim=0)
        _time = torch.cat([time, time], dim=0)
        _c = torch.cat([c, c], dim=0)
        if x_self_cond is not None:
            _x_self_cond = torch.cat([x_self_cond, x_self_cond], dim=0)
        else:
            _x_self_cond = None
        force_drop_ids_1 = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long) # no drop
        force_drop_ids_2 = torch.ones((x.shape[0],), device=x.device, dtype=torch.long) # drop all
        force_drop_ids = torch.cat([force_drop_ids_1, force_drop_ids_2], dim=0) # shape: (batch*2, )
        cond_x, uncond_x = self.forward(
            _x, _time, _c, force_drop_ids=force_drop_ids, x_self_cond=_x_self_cond
        ).chunk(2, dim=0)
        scaled_x = uncond_x + cfg_scale * (cond_x - uncond_x)
        
        return scaled_x

# class DenoisingGPT2Model(nn.Module):
#     """ GPT2 model for denoising 1D data 
#     GPT model is a decoder-only transformer.
    
#     Data1D -> Data1D
    
#     """
#     def __init__(
#         self,
#         dim_base: int,
#         num_in_channel: int = 3,
#         dim_out: int|None = None,
#         self_condition: bool = False,
#         type_pos_emb: str = 'sinusoidal',
#         # dim_learned_pos_emb: int = 16,
#         num_attn_head: int = 4,
#         num_layer = 6,
#         dim_feedforward = 2048, # why so big by default?
#         dropout = 0.1,
#         learned_variance: bool = False,
#     ):
#         super().__init__()
#         self.dim_base = dim_base
#         self.num_in_channel = num_in_channel
#         self.dim_out = default(dim_out, num_in_channel * (1 if not learned_variance else 2))
#         self.self_condition = self_condition
#         self.type_pos_emb = type_pos_emb
#         self.num_attn_head = num_attn_head
#         self.num_layer = num_layer
#         self.dim_feedforward = dim_feedforward
#         self.dropout = dropout
        
#         # define model layers here
#         # pos_emb for time
#         #   (batch, ) -> (batch, dim_base)
#         if self.type_pos_emb == 'learned':
#             pos_emb = LearnedSinusoidalPosEmb(self.dim_base)
#         elif self.type_pos_emb == 'random':
#             pos_emb = RandomSinusoidalPosEmb(self.dim_base)
#         else:
#             pos_emb = SinusoidalPosEmb(self.dim_base)
        
#         self.time_mlp = nn.Sequential(
#             pos_emb,
#             nn.Linear(self.dim_base, self.dim_base),
#             nn.GELU(),
#             nn.Linear(self.dim_base, self.dim_base * 2),
#         )
        
#         # pos_emb for transformer
#         #   use sinusoidal as the only option for now
#         self.transformer_pos_emb = SinusoidalPosEmb(self.dim_base)
        
#         # init proj 
#         #   (batch, dim_base) -> (batch, dim_base * 2)
#         self.init_proj = nn.Conv1d(num_in_channel if not self.self_condition else num_in_channel * 2, 
#                                    dim_base, 5, padding=2)
#         #   (batch, num_in_channel, sequence) -> (batch, dim_base, sequence)
        
#         # transformer
#         self.
        
#         # final conv
#         self.final_conv = nn.Conv1d(self.dim_base * 2, self.dim_out, 1)
    
#     def forward(self, x: Data1D, time: Float[Tensor, 'batch, '], x_self_cond: Optional[Data1D] = None) -> Data1D:
#         pass

class KernelVelocity(nn.Module):
    """
    x0: data -> x1: noise
    
    m: number of kernels (~multiple of number of training samples)
    h: bandwidth of the kernel, literature suggests h = 1
    k: number of nearest neighbors to consider
    """
    def __init__(self, m: int, h: float):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.m = m
        self.h = h
        
    def batched_guassian_kernel(self, x: Float[Tensor, 'batch *'], c: Float[Tensor, 'm *']) -> Float[Tensor, 'batch m']:
        "x: batch of data, c: batch of centers"
        x = rearrange(x, 'batch ... -> batch 1 ...')
        c = rearrange(c, 'm ... -> 1 m ...')
        diff = x - c
        diff = diff.flatten(2).norm(dim=-1) # shape: (batch, m)
        return torch.exp(- diff ** 2 / (2 * self.h ** 2))
    
    def train(self, x_train: Float[Tensor, 'batch *']):
        "x_train: samples of x1"
        # is m < len(x_train)?
        # assign an x_0 for each x_1? 
        self.x_0 = x_train
        self.x_1 = torch.randn_like(x_train)
        
    def parameters(self):
        yield self.x_0
        yield self.x_1
        
    def to(self, device):
        try:
            self.x_0 = self.x_0.to(device)
            self.x_1 = self.x_1.to(device)
            self.dummy_param = self.dummy_param.to(device)
        except AttributeError:
            pass
        return self
        
    def predict(self, z_t: Float[Tensor, 'batch *'], t: float, **kwargs) -> Float[Tensor, 'batch *']:
        "x_t,t -> v_t"
        if isinstance(t, torch.Tensor):
            assert t.std() <= 1e-6, 't must be a constant'
            t = t[0].item()
        x_t = (1-t) * self.x_0.to(z_t.device) + t * self.x_1.to(z_t.device) # shape: (M, *)
        dist = self.batched_guassian_kernel(z_t, x_t) # shape: (batch, M)
        topk = torch.topk(dist, self.m, dim=1)
        topk_dist, topk_idx = topk.values, topk.indices # shape: (batch, m)
        topk_x_1 = self.x_1[topk_idx.flatten(), ...] # shape: (batch*m, *)
        topk_x_1 = rearrange(topk_x_1, '(batch m) ... -> batch m ...', m = self.m) # shape: (batch, m, *)
        topk_weight = topk_dist / (topk_dist.sum(dim=1, keepdim=True)+1e-7) # shape: (batch, m)
        z_t = rearrange(z_t, 'batch ... -> batch 1 ...') # shape: (batch, 1, *)
        for _ in range(z_t.ndim -2):
            topk_weight = topk_weight.unsqueeze(-1) 
            # shape: (batch, m, 1, 1, ..., 1)
        velocity = (topk_x_1 - z_t)/(1-t+1e-7)*topk_weight # shape: (batch, m, *)
        velocity = velocity.sum(dim=1) # shape: (batch, *)
        
        return velocity
    
    def forward_with_cfg(self, x, t, **kwargs):
        return self.predict(x, t, **kwargs)
    
    def forward(self, x, t, **kwargs):
        return self.predict(x, t, **kwargs)
        
    # def __call__(self, x, t, **kwargs):
    #     return self.predict(x, t, **kwargs)
    