from typing import Iterable, Callable
from functools import partial

from tqdm import tqdm
import torch
from energydiff.diffusion import EmbedderWrapper, Trainer1D, SpacedDiffusion1D
from energydiff.diffusion.dpm_solver import DPMSolverSampler

class ConditionCrafter():
    def __init__(
        self,
        dict_cond_num_emb: dict[str, int]|None=None,
    ):
        self.dict_cond_num_emb = dict_cond_num_emb
            # has name of all cond and their according num_embedding
        
    @classmethod
    def from_embedder(cls, dict_cond_embedder: dict[str, EmbedderWrapper]):
        dict_cond_num_emb = {
            cond_name: embedder.num_embedding
            for cond_name, embedder in dict_cond_embedder.items()
        }
        return cls(dict_cond_num_emb)
        
    def __call__(self, batch_size: int, dict_cond: dict[str, torch.Tensor|None] = {}) -> torch.Tensor:
        """ dict_cond is a dictionary of the condition tensor. 
        
        cond value shape: [] (scalar) / [channel]
        
        example: 
            {
                'day': None
                'season': torch.Tensor(0)
                'annual_consumption': torch.Tensor(10000)
            }
        """
        cond_list = []
        for cond_name in self.dict_cond_num_emb.keys():
            if cond_name in dict_cond.keys() and dict_cond[cond_name] is not None:
                cond_list.append(dict_cond[cond_name].reshape((1, -1, 1)))
                    # shape: [1, channel, 1]
            else:
                null_cond = torch.tensor(self.dict_cond_num_emb[cond_name])
                cond_list.append(null_cond.reshape((1, -1, 1)))
                    # shape: [1, channel, 1]
                
        _cond_tensor = torch.cat(cond_list, dim=1)
        batched = _cond_tensor.repeat((batch_size, 1, 1))
        
        return batched
    
def ancestral_sample(
    num_sample: int, 
    batch_size: int, 
    cond: torch.Tensor|None=None,
    cfg_scale: float = 1.,
    trainer: Trainer1D|None=None, 
    model: SpacedDiffusion1D|None=None,
    post_transforms: Iterable[Callable] = [],
) -> torch.Tensor:
    """ sample from either the EMA in trainer or directly the diffusion model. 
    
    trainer = Trainer1D and model = None: sample from EMA
    trainer = None and model = GaussianDiffusion1D: sample from model
    
    Arguments:
        - num_sample: int, number of samples to generate
        - batch_size: int, batch size for sampling
        - cond: shape [batch_size, channel, 1], condition for sampling
    """
    # process cond
    if cond is not None:
        if model is not None:
            _device = next(model.parameters()).device
            cond = cond.to(_device)
        else:
            cond = cond.to(trainer.device)
    
    # process batch size split
    if num_sample < batch_size:
        list_batch_size = [num_sample]
    else:
        list_batch_size = [batch_size] * (num_sample // batch_size)
        if num_sample % batch_size != 0:
            list_batch_size.append(num_sample % batch_size)
    
    # sample
    list_sample = []
    for idx, batch_size in enumerate(list_batch_size):
        print(f'sampling batch {idx+1}/{len(list_batch_size)}, batch size {batch_size}. ')
        cond = cond[:batch_size] if cond is not None else None
        model_kwargs = {
            'c': cond,
            'cfg_scale': cfg_scale,
        }
        if model is None:
            # trainer.ema.ema_model.half()
            # with trainer.accelerator.autocast():
            sample_batch = trainer.ema.ema_model.sample(batch_size=batch_size, clip_denoised=True, model_kwargs=model_kwargs).float()
        else:
            sample_batch = model.sample(batch_size=batch_size, clip_denoised=True, model_kwargs=model_kwargs)
        
        # post transforms
        for post_trans in post_transforms:
            sample_batch = post_trans(sample_batch)
        
        list_sample.append(sample_batch)
    all_sample = torch.cat(list_sample, dim=0)
    
    if model is None:
        gathered_all_sample = trainer.accelerator.gather(all_sample)
        return gathered_all_sample
    return all_sample

def dpm_solver_sample(
    sampler: DPMSolverSampler,
    total_num_sample: int,
    batch_size: int,
    step: int,
    shape: tuple[int, int],
    conditioning: torch.Tensor|None,
    cfg_scale: float,
    accelerator = None,
    clip_denoised: bool = True,
) -> torch.Tensor:
    if accelerator is not None:
        num_sample = total_num_sample // accelerator.num_processes # for this process
    else:
        num_sample = total_num_sample
    if num_sample < batch_size:
        list_batch_size = [num_sample]
    else:
        list_batch_size = [batch_size] * (num_sample // batch_size)
        if num_sample % batch_size != 0:
            list_batch_size.append(num_sample % batch_size)
    
    list_sample = []
    for idx, batch_size in enumerate(list_batch_size):
        print(f'sampling batch {idx+1}/{len(list_batch_size)}, batch size {batch_size}. ')
        sample_batch, _ = sampler.sample(
            S = step,
            batch_size = batch_size,
            shape = shape,
            conditioning = conditioning,
            cfg_scale = cfg_scale,
        )
        list_sample.append(sample_batch)
        
    all_sample = torch.cat(list_sample, dim=0)
    if accelerator is not None:
        gathered_all_sample = accelerator.gather(all_sample)
        return gathered_all_sample
    
    if clip_denoised:
        all_sample = torch.clamp(all_sample, -1, 1)
    return all_sample
    
def ancestral_sample_progressive(
    batch_size: int, 
    save_every: int = 100,
    save_steps: list[int]|None=None,
    cond: torch.Tensor|None=None,
    cfg_scale: float = 1.,
    trainer: Trainer1D|None=None, 
    model: SpacedDiffusion1D|None=None,
    post_transforms: Iterable[Callable] = [],
    clip_denoised: bool = True,
) -> torch.Tensor:
    "Save every (few) reverse step, but only sample one batch. "
    # process cond
    if cond is not None:
        if model is not None:
            _device = next(model.parameters()).device
            cond = cond.to(_device)
        else:
            cond = cond.to(trainer.device)
    
    # sample prep
    cond = cond[:batch_size] if cond is not None else None
    model_kwargs = {
        'c': cond,
        'cfg_scale': cfg_scale,
    }
    if model is None:
        sample_fn = trainer.ema.ema_model.p_sample_loop_progressive
        shape = (batch_size, trainer.ema.ema_model.num_in_channel, trainer.ema.ema_model.seq_length)
        total_step = trainer.ema.ema_model.num_timestep
    else:
        sample_fn = model.p_sample_loop_progressive
        shape = (batch_size, model.num_in_channel, model.seq_length)
        total_step = model.num_timestep
    if save_steps is None:
        save_steps = list(range(0, total_step+save_every, save_every))
            # make sure `total_step` is included -> later 0 step in included
        save_steps = [total_step-_step for _step in save_steps]
            # make sure including total_step - total_step 
    else:
        pass # use the given save_steps
    # sample loop
    step = total_step
    list_sample = []
    for sample_out in sample_fn(shape=shape, clip_denoised=clip_denoised, model_kwargs=model_kwargs):
        sample_batch = sample_out['pred_x_prev'] # also 'pred_x_start'
        step -= 1 # starting from T-1. manually add T sample later (noise)
        if step in save_steps:
            list_sample.append(sample_batch)
        
    # add T sample
    list_sample = [torch.randn_like(list_sample[0])] + list_sample
    assert len(list_sample) == len(save_steps)
    all_sample = torch.stack(list_sample, dim=1)
        # shape: [batch_size, num_save_step, channel, length]
    
    # post transforms
    for post_trans in post_transforms:
        b, t, c, l = all_sample.shape
        sample_batch = post_trans(all_sample.view(b*t, c, l))
        sample_batch = sample_batch.view(b, t, c, l)
        
    # gather if necessary
    if model is None:
        gathered_all_sample = trainer.accelerator.gather(all_sample)
        return gathered_all_sample, save_steps
    return all_sample, save_steps