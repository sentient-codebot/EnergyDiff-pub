""" Version history

0.3.0:
    date: vroeger dan 2023-12-07
    feature: 
        - conditional and unconditional ddpm
        - untested classifier free guidance
    
0.4.0:
    date: 2023-12-07
    feature:
        - refactored code (diffusion / models)
        - new vlb losses
        - able to learn variance with according loss (not sure whether compatible with v-prediction)

1.0.0:
    date: 2023-12-13
    feature:
        - spaced diffusion
        - deleted type_transformer option, now only allowing gpt2 backbone for transformer
        - unet backbone is deprecated, not updated for conditioning, etc. 
        
1.1.0:
    date: 2023-12-15
    feature:
        - allow varying num_sampling_step without retraining. 
        - !!! NOTE: for backward compatibility, model.load_state_dict and ema.load_state_dict are set to `strict=False` !!!
        - BUG: why can't I sample normally now? even with the training setup 400/4000? 
        
2.0.0:
    date: 2024-01-03
    feature:
        - the bugs in 1.1.0 are fixed before this version already. 
        - updated unet arch and config. 
        - self.hidden_dim = default(dim_head, dim_head * num_head) in linear attention is wrong. 
            CHANGED TO self.hidden_dim = dim_head * num_head
            - meaning NO MORE backward compatibility with previous checkpoints.
        - slight change on only_central_dim decorator to be compatible with mps. 
        
2.1.0:
    date: 2024-04-08
    feature:
        - added mlp layer for transformer pos embedding. 
        - rescaled pos embedding for different time step. 

2.3.0:
    date: 2024-01-21
    feature:
        - truly use 4000 diffusion steps for training. 
        
2.4.0:
    date: 2024-02-11
    feature:
        - replace the attention module in GPT2 block with standard `SelfAttention`. (no softmax over heads)
        
2.5.0:
    date: 2024-02-17
    feature:
        - keep using the previous attention. 
        
2.5.1:
    date: 2024-03-13
    feature:
        - enable clip denoised
        
2.6.0:
    date: 2024-03-15
    feature:
        - QK normalization

# 2.6.1:
#     date: 2024-03-18
#     feature:
#         - scale RF step t from (0,1) to (0,1000)

2.6.2:
    date: 2024-12-19
    feature:
        - fixed typo `pred_noise = self.predict_noise_from_start(x_start, t, x)` -> `pred_noise = self.predict_noise_from_start(x, t, x_start)`
        
2.7.0:
    date: 2024-12-20
    feature:
        - refactor the project.

"""

__version__ = '2.7.0'
