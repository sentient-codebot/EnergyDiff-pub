import torch
import matplotlib.pyplot as plt
import numpy as np
from energydiff.utils.plot import plot_ddpm_progressive

def main():
    loaded = torch.load('ddpm_progressive/ddpm_20240222-8654_cossmic_grid-import_residential_samples_1min_winter_1000_1000.pt')
    samples = loaded['samples'] # (n, t, c, l)
    steps = loaded['steps'] # (n, )
    samples = samples.flatten(2) # (n, t, c*l)
    # special 'samples'
    mean_sample = samples.mean(dim=0, keepdim=True) # (1, t, l)
    fft_sample = torch.fft.rfft(samples, dim=-1, norm='ortho').abs() # magnitude, (n, t, l)
    mean_fft_sample = fft_sample.mean(dim=0, keepdim=True) # (1, t, l)
    true_fft = mean_fft_sample[:, -1:, :] # (1, 1, l)
    error_fft = (mean_fft_sample - true_fft).abs() # (1, t, l)
    # select sample
    selected = samples[4:7] # (3, t, l)
    to_plot = {}
    to_plot_mean = {}
    to_plot_fft = {}
    to_plot_error_fft = {}
    for idx, step in enumerate(steps):
        to_plot[step] = selected[:,idx,:].cpu().numpy() # (3, l)
        to_plot_mean[step] = mean_sample[:,idx,:].cpu().numpy() # (1, l)
        to_plot_fft[step] = mean_fft_sample[:,idx,:].cpu().numpy() # (1, l)
        to_plot_error_fft[step] = error_fft[:,idx,:].cpu().numpy() # (1, l)
        
    plot_ddpm_progressive(
        to_plot,
        save_dir='ddpm_progressive',
        save_suffix='samples',
        scale_per_step=True,
        log_scale=False,
    )
    plot_ddpm_progressive(
        to_plot_mean,
        save_dir='ddpm_progressive',
        save_suffix='mean'
    )
    plot_ddpm_progressive(
        to_plot_fft,
        save_dir='ddpm_progressive',
        save_suffix='fft',
        scale_per_step=False,
        log_scale=True,
    )
    plot_ddpm_progressive(
        to_plot_error_fft,
        save_dir='ddpm_progressive',
        save_suffix='error_fft',
        scale_per_step=False,
        log_scale=True,
    )
    pass
    
if __name__ == '__main__':
    main()