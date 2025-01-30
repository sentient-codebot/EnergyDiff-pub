import os
from typing import Sequence, Optional, Annotated
from typing import Annotated as Float
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from numpy import ndarray
from scipy import stats
import torch
from torch import Tensor

@plt.style.context('utils.article')
def plot_sampled_data(dataset_each_season: dict[str, Annotated[np.ndarray, 'batch, sequence']], 
                      samples_each_season: Optional[dict[str, Annotated[np.ndarray, 'batch, sequence']]] = None, 
                      name_seasons: Sequence[str]|None = None, 
                      save_filepath: Optional[str] = None,
                      num_samples_to_plot: Optional[int] = 1000):
    """ this plots only one season """
    name_seasons = name_seasons or list(dataset_each_season.keys())
    # alpha adjustment factor
    dot_alpha_coef = 10 # 20 / 1000 = 0.020
    line_alpha_coef = 72 # 5 / 1000 = 0.005
    
    num_seasons = len(name_seasons)
    if save_filepath is None:
        save_filepath = 'results/untitled_plot_sampled_data.png'
    
    # Step 2: Format
    if isinstance(list(dataset_each_season.values())[0], torch.Tensor):
        dataset_each_season = {season: data.cpu().numpy() for season, data in dataset_each_season.items()}
    if isinstance(list(samples_each_season.values())[0], torch.Tensor):
        samples_each_season = {season: sample.cpu().numpy() for season, sample in samples_each_season.items()}
        
    # Step 3: Visualize samples
    rng = np.random.default_rng() # for selecting indices to plot
    fig = plt.figure(figsize=(10, 6*num_seasons)) 
    gspec = gridspec.GridSpec(nrows=num_seasons, ncols=2, figure=fig)
    # fig, axes = plt.subplots(2, num_seasons, figsize=(8*num_seasons, 10)) # TODO change figure width, using gridspec
    # if num_seasons == 1:
    #     axes = axes.reshape(2, 1)
    for season_index, season in enumerate(name_seasons):
        # Get data
        rng = np.random.default_rng()
        
        indices = rng.permutation(samples_each_season[season].shape[0])[:num_samples_to_plot]
        samples = samples_each_season[season][indices]
        
        indices = rng.permutation(dataset_each_season[season].shape[0])[:num_samples_to_plot]
        data = dataset_each_season[season][indices]
        
        #  -- sort by total consumption -- 
        samples_daily_sum = samples.sum(axis=1)
        data_daily_sum = data.sum(axis=1)
        sample_idx = np.argsort(samples_daily_sum)
        data_idx = np.argsort(data_daily_sum)
        
        samples = samples[sample_idx]
        data = data[data_idx]
        # -- end of sorting --
        
        # Statistics
        sample_mean = np.mean(samples, axis=0)
        sample_std = np.std(samples, axis=0)
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        
        max_day_consumption = data.sum(axis=1).max()
        min_day_consumption = data.sum(axis=1).min()
        line_cmap = mpl.cm.get_cmap('RdYlBu_r')
        _line_color_norm = mpl.colors.Normalize(vmin=min_day_consumption, vmax=max_day_consumption)
        _get_color = lambda sample_day: line_cmap(_line_color_norm(sample_day.sum()))
        
        # Normalize
        data_vec_dim = data.shape[1] # data.shape = [num_samples_to_plot, 96]
        common_minimum = min([np.min(samples), np.min(data)])
        common_maximum = max([np.max(samples), np.max(data)])
        _norm = lambda x: (x - common_minimum) / (common_maximum - common_minimum)
        # plt.fill_between(range(means.shape[0]), means - stds, means + stds, alpha=0.2)
        # num_samples_to_plot = num_samples_to_plot
        
        ### subplot 0
        ax_sample = fig.add_subplot(gspec[season_index, 0])
        ax_generated = fig.add_subplot(gspec[season_index, 1])
        for index in range(num_samples_to_plot):
            ax_sample.scatter(
                np.arange(data_vec_dim), 
                data[index, :], 
                color=_get_color(data[index,:]), 
                s=0.5, 
                alpha=min(dot_alpha_coef/num_samples_to_plot,1),
                rasterized=True, 
            )
            ax_sample.plot(
                np.arange(data_vec_dim), 
                data[index, :], 
                linewidth=2.5,
                color=_get_color(data[index,:]), 
                alpha=min(line_alpha_coef/num_samples_to_plot,1),
                rasterized=True,
            )
        
        ax_sample.set_xlabel('Time Step [-]')
        ax_sample.set_ylabel('HP Active Power [W]')
        ax_sample.set_xlim([0, data_vec_dim])
        ax_sample.set_ylim([common_minimum, common_maximum])
        ax_sample.set_title('Target Samples' + f' ({season})')
        ### subplot 1
        for index in range(num_samples_to_plot):
            ax_generated.scatter(
                np.arange(data_vec_dim), 
                samples[index, :], 
                color=_get_color(samples[index, :]), 
                s=0.5, 
                alpha=min(dot_alpha_coef/num_samples_to_plot,1),
                rasterized=True,
            )
            ax_generated.plot(
                np.arange(data_vec_dim), 
                samples[index, :], 
                linewidth=2.5,
                color=_get_color(samples[index, :]),
                # alpha=min(line_alpha_coef/num_samples_to_plot,1)
                alpha = 0.36,
                rasterized=True,
            )

        ax_generated.set_xlabel('Time')
        ax_generated.set_yticks([])
        ax_generated.set_xlim([0, data_vec_dim])
        ax_generated.set_ylim([common_minimum, common_maximum])
        ax_generated.set_title('Generated Samples' + f' ({season})')
    
    plt.savefig(save_filepath, dpi=300, bbox_inches='tight')
    
    return fig

# @plt.style.context('utils.small_fig_new')
@plt.style.context('energydiff.utils.article')
def plot_sampled_data_v2(samples: dict[str, Annotated[np.ndarray, 'batch, sequence']], 
                      save_filepath: Optional[str] = None,
                      num_samples_to_plot: Optional[int] = 1000,
                      seed=0000):
    """ this plots only one season """
    # alpha adjustment factor
    dot_alpha_coef = 10 # 20 / 1000 = 0.020
    line_alpha_coef = 72 # 5 / 1000 = 0.005
    
    num_subfigure = len(samples)
    if save_filepath is None:
        save_filepath = 'results/untitled_plot_sampled_data.png'
    
    # Step 2: Format
    if isinstance(list(samples.values())[0], torch.Tensor):
        samples = {season: data.cpu().numpy() for season, data in samples.items()}
        
    # Step 3: Visualize samples
    rng = np.random.default_rng() # for selecting indices to plot
    fig = plt.figure(figsize=(10, 2.5*num_subfigure)) 
    gspec = gridspec.GridSpec(nrows=num_subfigure, ncols=1, figure=fig)
    # fig, axes = plt.subplots(2, num_seasons, figsize=(8*num_seasons, 10)) # TODO change figure width, using gridspec
    # if num_seasons == 1:
    #     axes = axes.reshape(2, 1)
    for idx, (name, sample) in enumerate(samples.items()):
        # Get data
        rng = np.random.default_rng(seed)
        
        indices = rng.permutation(sample.shape[0])[:num_samples_to_plot]
        sample = sample[indices]
        
        #  -- sort by total consumption -- 
        samples_daily_sum = sample.sum(axis=1)
        sample_idx = np.argsort(samples_daily_sum)
        
        sample = sample[sample_idx]
        # -- end of sorting --
        
        # Statistics
        sample_mean = np.mean(sample, axis=0)
        sample_std = np.std(sample, axis=0)
        
        max_day_consumption = sample.sum(axis=1).max()
        min_day_consumption = sample.sum(axis=1).min()
        line_cmap = mpl.cm.get_cmap('RdYlBu_r')
        _line_color_norm = mpl.colors.Normalize(vmin=min_day_consumption, vmax=max_day_consumption)
        _get_color = lambda sample_day: line_cmap(_line_color_norm(sample_day.sum()))
        
        # Normalize
        data_vec_dim = sample.shape[1] # data.shape = [num_samples_to_plot, 96]
        # _norm = lambda x: (x - common_minimum) / (common_maximum - common_minimum)
        # plt.fill_between(range(means.shape[0]), means - stds, means + stds, alpha=0.2)
        # num_samples_to_plot = num_samples_to_plot
        
        ### subplot 0
        ax_sample = fig.add_subplot(gspec[idx, 0])
        ax_sample.grid(True)
        for index in range(num_samples_to_plot):
            ax_sample.scatter(
                np.arange(data_vec_dim), 
                sample[index, :], 
                color=_get_color(sample[index,:]), 
                s=0.5, 
                alpha=min(dot_alpha_coef/num_samples_to_plot,1),
                rasterized=True, 
            )
            ax_sample.plot(
                np.arange(data_vec_dim), 
                sample[index, :], 
                linewidth=2.5,
                color=_get_color(sample[index,:]), 
                alpha=min(line_alpha_coef/num_samples_to_plot,1),
                rasterized=True,
            )
        
        ax_sample.set_xlabel('Time Step [-]')
        ax_sample.set_ylabel('Power [W]')
        ax_sample.set_xlim([0, data_vec_dim-1])
        # ax_sample.set_ylim([common_minimum, common_maximum])
        ax_sample.set_title(f'{name}')
    
    plt.savefig(save_filepath, dpi=300, bbox_inches='tight')
    
    return fig

@plt.style.context('utils.article')
def plot_sampled_data_pv(samples: dict[str, Annotated[np.ndarray, 'batch, sequence']], 
                      save_filepath: Optional[str] = None,
                      num_samples_to_plot: Optional[int] = 1000):
    """ this plots only one season """
    # alpha adjustment factor
    dot_alpha_coef = 10 # 20 / 1000 = 0.020
    line_alpha_coef = 72 # 5 / 1000 = 0.005
    
    num_subfigure = len(samples)
    if save_filepath is None:
        save_filepath = 'results/untitled_plot_sampled_data.png'
    
    # Step 2: Format
    if isinstance(list(samples.values())[0], torch.Tensor):
        samples = {season: data.cpu().numpy() for season, data in samples.items()}
        
    # Step 3: Visualize samples
    rng = np.random.default_rng() # for selecting indices to plot
    fig = plt.figure(figsize=(10, 2.5*num_subfigure)) 
    gspec = gridspec.GridSpec(nrows=num_subfigure, ncols=1, figure=fig)
    # fig, axes = plt.subplots(2, num_seasons, figsize=(8*num_seasons, 10)) # TODO change figure width, using gridspec
    # if num_seasons == 1:
    #     axes = axes.reshape(2, 1)
    for idx, (name, sample) in enumerate(samples.items()):
        # Get data
        rng = np.random.default_rng()
        
        indices = rng.permutation(sample.shape[0])[:num_samples_to_plot]
        sample = sample[indices]
        
        #  -- sort by total consumption -- 
        samples_daily_sum = sample.sum(axis=1)
        sample_idx = np.argsort(samples_daily_sum)
        
        sample = sample[sample_idx]
        # -- end of sorting --
        
        # Statistics
        sample_mean = np.mean(sample, axis=0)
        sample_std = np.std(sample, axis=0)
        
        max_day_consumption = sample.sum(axis=1).max()
        min_day_consumption = sample.sum(axis=1).min()
        line_cmap = mpl.cm.get_cmap('RdYlBu_r')
        _line_color_norm = mpl.colors.Normalize(vmin=min_day_consumption, vmax=max_day_consumption)
        _get_color = lambda sample_day: line_cmap(_line_color_norm(sample_day.sum()))
        
        # Normalize
        data_vec_dim = sample.shape[1] # data.shape = [num_samples_to_plot, 96]
        xticks = np.linspace(0, 1440, sample.shape[1], dtype=np.int32)
        
        ### subplot 0
        ax_sample = fig.add_subplot(gspec[idx, 0])
        ax_sample.grid(True)
        for index in range(num_samples_to_plot):
            ax_sample.scatter(
                xticks, 
                sample[index, :], 
                color=_get_color(sample[index,:]), 
                s=0.5, 
                alpha=min(dot_alpha_coef/num_samples_to_plot,1),
                rasterized=True, 
            )
            ax_sample.plot(
                xticks, 
                sample[index, :], 
                linewidth=2.5,
                color=_get_color(sample[index,:]), 
                alpha=min(line_alpha_coef/num_samples_to_plot,1),
                rasterized=True,
            )
        
        ax_sample.set_xlabel('Time Step [-]')
        ax_sample.set_ylabel('Power [W]')
        ax_sample.set_xlim([0, xticks[-1]])
        ax_sample.set_ylim([-2000, 20000])
        ax_sample.set_title(f'{name}')
    
    plt.savefig(save_filepath, dpi=300, bbox_inches='tight')
    
    return fig
    
def plot_autocorrelation(acf: Annotated[np.ndarray, 'sequence, sequence'], model_name: str, season: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    image = ax.imshow(acf, cmap='PuOr_r')
    ax.set_title(f'Autocorrelation of {model_name} samples ({season})')
    plt.colorbar(image,ax=ax)
    
    plt.savefig(f'results/{model_name}/acf_{model_name}_{season}.png', dpi=300)
    
@plt.style.context('energydiff.utils.article_ieee')
def plot_acf_comparison(
    dict_acf: dict[str, Annotated[np.ndarray, 'sequence, sequence']],
    dict_error: dict[str, float] = {},
    filename: str = 'cov_comparison.pdf',
    save_dir: str = 'results/figures/covariance') -> mpl.figure.Figure:
    # arguments checking
    ...
    # mpl.rcParams['figure.constrained_layout.use'] = False
    
    # prepare colormap
    vmin = min([acf.min() for acf in dict_acf.values()])
    vmax = max([acf.max() for acf in dict_acf.values()])
    
    # Create evenly spaced boundaries between vmin and vmax
    lin_bounds = np.linspace(vmin, vmax, 31)
    norm = mpl.colors.BoundaryNorm(boundaries=lin_bounds, ncolors=256)
    
    # prepare figure
    num_plot = len(dict_acf)
    fig = plt.figure(figsize=(3.5, 18/(2+4*num_plot)))
    gspec = gridspec.GridSpec(nrows=2, ncols=num_plot+1, figure=fig, width_ratios=[1]*num_plot+[0.1], height_ratios=[0.05, 1])
    # plt.rcParams['axes.labelsize'] /= num_plot
    plt.rcParams['xtick.labelsize'] /= num_plot*0.5
    plt.rcParams['ytick.labelsize'] /= num_plot*0.5
    
    # axes for imshow
    im_axes = []
    for idx, (model_name, acf) in enumerate(dict_acf.items()):
        ax = fig.add_subplot(gspec[1, idx])
        im_axes.append(ax)
        image = ax.imshow(acf, norm=norm, cmap='Blues', interpolation='nearest', rasterized=True)
        num_timesteps = acf.shape[0]
        tick_locs = np.linspace(0.5, num_timesteps-0.5, 5)
        tick_labels = [f'{int(x+0.5)}' for x in tick_locs]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, rotation=-0)
        ax.set_xlabel('Time Step [-]')
        if idx == 0:
            ax.set_yticks(tick_locs)
            ax.set_yticklabels(tick_labels)
            ax.set_ylabel('Time Step [-]')
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])
        title = f'{model_name}'
        extra = dict_error.get(model_name, '')
        extra = f' (MSE={extra:.4f})' if extra != '' else ''
    
        title_ax = fig.add_subplot(gspec[0, idx])
        title_ax.axis('off')
        title_ax.text(0.5, 0.5, title + extra, ha='center', va='center', fontsize=6)
        
    # axes for colorbar
    cbar_ax = fig.add_subplot(gspec[1, -1])
    cbar = plt.colorbar(
        image, 
        cax=cbar_ax, 
        extend='neither',
    )
    
    # Set tick locations and labels
    num_ticks = 7
    tick_locs = np.linspace(vmin, vmax, num_ticks)
    tick_labels = [f'{x:.2f}' for x in tick_locs]
    cbar.set_ticks(tick_locs, minor=False)
    cbar.set_ticks([], minor=True)
    cbar.set_ticklabels(tick_labels, rotation=-0)
    cbar.outline.set_visible(False)
    
    # Set colorbar limits explicitly
    cbar.ax.set_ylim(vmin, vmax)
    # cbar.ax.set_ylabel('Covariance Matrix', size=10)
    
    # plt.show()
    # pass
    
    # save figure
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    
    return fig

@plt.style.context('energydiff.utils.article_ieee')
def plot_cov_comparison(
    dict_cov: dict[str, Annotated[np.ndarray, 'sequence, sequence']],
    dict_error: dict[str, float] = {},
    filename: str = 'cov_comparison.pdf',
    save_dir: str = 'results/figures/covariance') -> mpl.figure.Figure:
    # arguments checking
    ...
    assert len(dict_cov) % 2 == 0
    
    # prepare colormap
    vmin = min([acf.min() for acf in dict_cov.values()])
    vmax = max([acf.max() for acf in dict_cov.values()])
    
    # Create evenly spaced boundaries between vmin and vmax
    lin_bounds = np.linspace(vmin, vmax, 31)
    norm = mpl.colors.BoundaryNorm(boundaries=lin_bounds, ncolors=256)
    
    # prepare figure
    num_plot = len(dict_cov)
    fig = plt.figure(figsize=(3.5, 7*num_plot/10), layout='compressed')
    gspec = gridspec.GridSpec(nrows=num_plot//2, ncols=3, figure=fig, width_ratios=[1, 1, 0.1],)
    
    gspec_im = lambda idx: gspec[idx//2, idx%2]
    
    # axes for imshow
    im_axes = []
    for idx, (model_name, cov) in enumerate(dict_cov.items()):
        ax = fig.add_subplot(gspec_im(idx))
        im_axes.append(ax)
        image = ax.imshow(cov, norm=norm, cmap='Blues', interpolation='nearest', rasterized=True)
        num_timesteps = cov.shape[0]
        tick_locs = np.linspace(0.5, num_timesteps-0.5, 5)
        tick_labels = [f'{int(x+0.5)}' for x in tick_locs]
        if idx == num_plot-2 or idx == num_plot-1:
            ax.set_xticks(tick_locs)
            ax.set_xticklabels(tick_labels, rotation=-0)
            ax.set_xlabel('Time Step [-]')
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
        if idx%2 == 0:
            ax.set_yticks(tick_locs)
            ax.set_yticklabels(tick_labels)
            ax.set_ylabel('Time Step [-]')
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])
        title = f'{model_name}'
        extra = dict_error.get(model_name, '')
        extra = f' (MSE={extra:.4f})' if extra != '' else ''
    
        title_ax = ax # same Axes
        title_ax.text(0.5, 1.00, title + extra, 
                      ha='center', va='bottom', fontsize=6, clip_on=False, 
                      weight='bold',
                      transform=ax.transAxes)
        
    # axes for colorbar
    cbar_ax = fig.add_subplot(gspec[0:, 2])
    cbar = plt.colorbar(
        image, 
        cax=cbar_ax, 
        extend='neither',
    )
    
    # Set tick locations and labels
    num_ticks = 7
    tick_locs = np.linspace(vmin, vmax, num_ticks)
    tick_labels = [f'{x:.2f}' for x in tick_locs]
    cbar.set_ticks(tick_locs, minor=False)
    cbar.set_ticks([], minor=True)
    cbar.set_ticklabels(tick_labels, rotation=-0)
    cbar.outline.set_visible(False)
    
    # Set colorbar limits explicitly
    cbar.ax.set_ylim(vmin, vmax)
    # cbar.ax.set_ylabel('Covariance Matrix', size=10)
    
    # plt.show()
    # pass
    
    # save figure
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    
    return fig

# TODO: change list arguments -> dict
@plt.style.context('utils.small_fig')
def _deprecated_plot_embedding_comparison(embedding_list: Sequence[Annotated[np.ndarray, 'batch emb_dim']],
                         model_name_list: Sequence[str],
                         season: Sequence[str],
                         embedding_type: str = 'tsne',
                         save_dir: str = 'results/') -> mpl.figure.Figure:
    # arguments checking
    assert len(embedding_list) == len(model_name_list)
    
    # prepare xy limits
    xmin = min([embedding[:, 0].min() for embedding in embedding_list])
    xmax = max([embedding[:, 0].max() for embedding in embedding_list])
    ymin = min([embedding[:, 1].min() for embedding in embedding_list])
    ymax = max([embedding[:, 1].max() for embedding in embedding_list])
    
    # prepare figure
    num_plot = len(embedding_list)
    fig = plt.figure(figsize=(2+8*num_plot, 8))
    gspec = gridspec.GridSpec(nrows=1, ncols=num_plot, figure=fig)
    
    # axes for plot
    for idx, (model_name, embedding) in enumerate(zip(model_name_list, embedding_list)):
        ax = fig.add_subplot(gspec[0, idx])
        ax.scatter(embedding[:, 0], embedding[:, 1], s=16, alpha=0.2)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_title(f'{model_name} samples ({season})')
        
    fig.suptitle(f'{embedding_type.capitalize()} Visualization')
        
    filename = f'{embedding_type}_' + '_'.join(model_name_list) + '_side_by_side_' + season + '.pdf'
    plt.savefig(os.path.join(save_dir, f'{filename}'), dpi=300)
    
    return fig

# TODO: change list arguments -> dict
@plt.style.context('utils.small_fig')
def plot_embedding_overlay(embedding_list: Sequence[Annotated[np.ndarray, 'batch emb_dim']],
                         model_name_list: Sequence[str],
                         season: Sequence[str],
                         embedding_type: str = 'tsne',
                         save_dir: str = 'results/') -> mpl.figure.Figure:
    # arguments checking
    assert len(embedding_list) == len(model_name_list)
    MAX_NUM_POINT = 1000
    for embedding in embedding_list:
        if embedding.shape[0] > MAX_NUM_POINT:
            embedding = embedding[:MAX_NUM_POINT, :]
    
    # prepare xy limits
    xmin = min([embedding[:, 0].min() for embedding in embedding_list])
    xmax = max([embedding[:, 0].max() for embedding in embedding_list])
    ymin = min([embedding[:, 1].min() for embedding in embedding_list])
    ymax = max([embedding[:, 1].max() for embedding in embedding_list])
    
    # prepare figure
    fig = plt.figure(figsize=(10, 8))
    im_cbar_ratio = 10
    gspec = gridspec.GridSpec(nrows=1, ncols=im_cbar_ratio+1, figure=fig)
    
    # axes for plot
    ax = fig.add_subplot(gspec[0, :-1])
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    for idx, (model_name, embedding) in enumerate(zip(model_name_list, embedding_list)):
        ax.scatter(embedding[:, 0], embedding[:, 1], s=25, alpha=0.1, label=model_name)
        
    # axes for legend
    legend_ax = fig.add_subplot(gspec[0, -1])
    legend_ax.axis('off')
    legend_ax.legend(model_name_list)

    fig.suptitle(f'{embedding_type.capitalize()} Visualization ({season.capitalize()})')
        
    filename = f'{embedding_type}_' + '_'.join(model_name_list) + '_overlay_' + season + '.pdf'
    plt.savefig(os.path.join(save_dir, f'{filename}'), dpi=300)
    
    return fig

plot_tsne_overlay = partial(plot_embedding_overlay, embedding_type='tsne')
plot_pca_overlay = partial(plot_embedding_overlay, embedding_type='pca')

@plt.style.context('utils.small_fig')
def plot_embedding_comparison(
    embedding_dict: dict[str, Annotated[np.ndarray, 'batch emb_dim']],
    season: str|None = None,
    embedding_type: str = 'tsne',
    save_dir: str = 'results/',
    save_label: str = '',
    max_num_point: int = 1500
) -> mpl.figure.Figure:
    # pre-process data
    for model_name, emb in embedding_dict.items():
        if emb.shape[0] > max_num_point:
            embedding_dict[model_name] = emb[:max_num_point, :]
    
    # prepare xy limits
    xmin = min([embedding[:, 0].min() for embedding in embedding_dict.values()])
    xmax = max([embedding[:, 0].max() for embedding in embedding_dict.values()])
    ymin = min([embedding[:, 1].min() for embedding in embedding_dict.values()])
    ymax = max([embedding[:, 1].max() for embedding in embedding_dict.values()])
    xmin = min(xmin*1.05, xmin*0.95)
    xmax = max(xmax*1.05, xmax*0.95)
    ymin = min(ymin*1.05, ymin*0.95)
    ymax = max(ymax*1.05, ymax*0.95)
    
    # prepare colormap
    cmap = mpl.colormaps['tab10']
    indices = np.linspace(0, 1, len(embedding_dict.keys()))
    colors = {model_name: cmap(index) for model_name, index in zip(embedding_dict.keys(), indices)}
    # cmap = mpl.colormaps['Set1']
    # colors = [cmap(i) for i in range(9)]
    
    # prepare figure
    num_plot = len(embedding_dict.keys())
    fig = plt.figure(figsize=(2+8*num_plot, 8))
    gspec = gridspec.GridSpec(nrows=1, ncols=num_plot, figure=fig)
    
    # axes for plot
    scatter_handles = {}
    for idx, (model_name, embedding) in enumerate(embedding_dict.items()):
        ax = fig.add_subplot(gspec[0, idx])
        ax.grid(True)
        if model_name != 'Real Data':
            target_emb = embedding_dict['Real Data']
            ax.scatter(target_emb[:, 0], target_emb[:, 1], s=100, alpha=0.15, 
                       color=colors['Real Data'], 
                    #    color=colors[1],
                       label='target', marker='o', rasterized=True)
            h = ax.scatter(embedding[:, 0], embedding[:, 1], s=75, alpha=0.70, 
                           color=colors[model_name], 
                        #    color=colors[0],
                           label=model_name, marker='^', rasterized=True)
        else:
            h = ax.scatter(embedding[:, 0], embedding[:, 1], s=75, alpha=0.85, 
                           color=colors[model_name], 
                        #    color=colors[1],
                           label=model_name, marker='o', rasterized=True)
        scatter_handles[model_name] = h
        ax.set_xlabel('Dimension A [-]')
        ax.set_ylabel('Dimension B [-]')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        if season is not None:
            ax.set_title(f'{model_name} samples ({season})')
        else:
            ax.set_title(f'{model_name}')
        
    # axes for legend
    # legend_ax = fig.add_subplot(gspec[0, -1])
    legend_ax = ax
    legend_ax.legend(
        handles = [scatter_handles[model_name] for model_name in embedding_dict.keys()],
        labels = list(embedding_dict.keys()),
        markerscale = 3,
    )
        
    # fig.suptitle(f'{embedding_type} Visualization')
        
    _season = season or ''
    filename = f'{embedding_type}_' + '_'.join(list(embedding_dict.keys())) + '_comparison_new_' + _season + save_label + '.pdf'
    plt.savefig(os.path.join(save_dir, f'{filename}'), dpi=300)
    
    return fig

plot_tsne_comparison = partial(plot_embedding_comparison, embedding_type='tsne')
plot_pca_comparison = partial(plot_embedding_comparison, embedding_type='pca')

@plt.style.context('utils.article')
def plot_point2d(
    xy_list: Sequence[Annotated[np.ndarray, 'N 2']],  
    xlabel: str,
    ylabel: str,
    model_name_list: Sequence[str], 
    season: str
) -> mpl.figure.Figure:
    # Step 0: argument checking
    assert len(xy_list) == len(model_name_list)
    
    # Step 1: determine xy limits
    xmin = min([xy[:, 0].min() for xy in xy_list])
    xmax = max([xy[:, 0].max() for xy in xy_list])
    ymin = min([xy[:, 1].min() for xy in xy_list])
    ymax = max([xy[:, 1].max() for xy in xy_list])
    
    # prepare a rectangle based on the target's value range
    # !ASSUME the first xy is the target
    xy_target = xy_list[0]
    target_xmin, target_xmax = xy_target[:, 0].min(), xy_target[:, 0].max()
    target_ymin, target_ymax = xy_target[:, 1].min(), xy_target[:, 1].max()
    bottom_left = (target_xmin, target_ymin)
    width = target_xmax - target_xmin
    height = target_ymax - target_ymin
    
    # Step 2: prepare figure
    num_plot = len(xy_list)
    nrows = num_plot // 2 + num_plot % 2
    ncols = 1 if num_plot == 1 else 2
    fig = plt.figure(figsize=(8*ncols, 6*nrows))
    gspec = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=fig)
    for index, (xy) in enumerate(xy_list):
        # Step 3: prepare axes
        ax = fig.add_subplot(gspec[index])
        
        # Step 4: calculate 2d histogram (density map)
        hist, xedges, yedges = np.histogram2d(xy[:, 0], xy[:, 1], bins=10, density=True)
        hist = hist.T
        
        # Step 5: draw scatter plot
        ax.scatter(xy[:, 0], xy[:, 1], s=16, alpha=0.2, c='black')
        ax.imshow(hist, interpolation='bessel', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='Blues', norm=mpl.colors.LogNorm())
        
        # if index != 0: # not target
        rect = mpl.patches.Rectangle(bottom_left, width, height, 
                                     linewidth=2, 
                                     edgecolor='red', 
                                     alpha=0.5,
                                     fill=False
                                     )
        ax.add_patch(rect)
        if index == 0:
            ax.legend([rect], ['Target Range'])
        
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        
        ax.set_title(f'{model_name_list[index].upper().replace("_", " ")}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
    fig.suptitle(f'Scatter Plot of Samples ({season.capitalize()})')
    
    filename = f'scatter_' + '_'.join(model_name_list) + '_' + season + '.pdf'
    plt.savefig(f'results/{filename}', dpi=300)
    
    return fig

@plt.style.context('utils.article')
def plot_contour2d(xy_list: Sequence[Annotated[np.ndarray, 'batch 2']],  
                 xlabel: str,
                 ylabel: str,
                 model_name_list: Sequence[str], 
                 season: str) -> mpl.figure.Figure:
    assert len(xy_list) == len(model_name_list)
    # plt.style.use('utils.article')
    fig, ax = plt.subplots(1, len(xy_list), figsize=(4+8*len(xy_list), 6))
    if len(xy_list) == 1:
        ax = [ax]
    for index, (xy) in enumerate(xy_list):
        hist, xedges, yedges = np.histogram2d(xy[:, 0], xy[:, 1], bins=10, density=True)
        hist = hist.T
        
        ax[index].scatter(xy[:, 0], xy[:, 1], s=16, alpha=0.2, c='black')
        # draw histogram
        ax[index].imshow(hist, interpolation='bessel', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='Blues', norm=mpl.colors.LogNorm())
        # x, y = np.meshgrid(xedges, yedges)
        # ax[index].pcolormesh(x, y, hist, cmap='PuOr', shading='gouraud')
        
        ax[index].set_title(f'Scatter plot of {model_name_list[index]} samples ({season})')
        ax[index].set_xlabel(xlabel)
        ax[index].set_ylabel(ylabel)
    
    filename = f'scatter_' + '_'.join(model_name_list) + '_' + season + '.pdf'
    plt.savefig(f'results/{filename}', dpi=300)
    
    return fig

@plt.style.context('utils.small_fig')
def plot_histogram(
    dict_samples: dict[str, Annotated[np.ndarray, 'batch, sequence']],
    cmap: str = 'tab10',
    save_dir: str = 'results/',
    save_label: str = '',
    log_scale: bool = True,
) -> mpl.figure.Figure:
    # prepare figure
    num_plot = len(dict_samples.keys())
    fig = plt.figure(figsize=(2+8*num_plot, 6))
    gspec = gridspec.GridSpec(nrows=1, ncols=num_plot, figure=fig)
    
    global_min = min([samples.min() for samples in dict_samples.values()])
    global_max = max([samples.max() for samples in dict_samples.values()])
    
    cmap = mpl.cm.get_cmap(cmap)
    dict_colors = {key: cmap(i) for i, key in enumerate(dict_samples.keys())}
    
    # axes for plot
    list_bin_values = []
    list_bins = []
    list_ax = []
    for idx, (model_name, samples) in enumerate(dict_samples.items()):
        ax = fig.add_subplot(gspec[0, idx])
        ax.grid(True)
        ax.set_xlabel('Power [W]')
        ax.set_ylabel('Density [-]')
        list_ax.append(ax)
        if model_name != 'Real Data':
            ax.hist(
            dict_samples['Real Data'].flatten(), 
            bins=100, 
            range=(global_min, global_max),
            histtype='stepfilled',
            color=dict_colors['Real Data'],
            alpha=0.8,
            density=True,
            label='Real Data'
        )
        _bin_values, _bins, _ = ax.hist(
            samples.flatten(), 
            bins=100, 
            range=(global_min, global_max),
            histtype='stepfilled',
            color=dict_colors[model_name],
            alpha=0.7,
            density=True,
            label=model_name
        )
        list_bin_values.append(_bin_values)
        list_bins.append(_bins)
        ax.set_title(f'{model_name}')
        # ax.set_yscale('log')
        
    xmin, xmax = np.array(list_bins).min(), np.array(list_bins).max()
    ymin, ymax = np.array(list_bin_values).min(), np.array(list_bin_values).max()
    for ax in list_ax:
        if log_scale:
            ax.set_yscale('log')
        ax.set_xlim([xmin, xmax])
        # ax.set_ylim([ymin, ymax])
    # fig.suptitle(f'Histogram of Samples')
        
    filename = f'hist_' + '_'.join(list(dict_samples.keys())) +save_label+ '.pdf'
    plt.savefig(os.path.join(save_dir, f'{filename}'), dpi=300)
    
    return fig

@plt.style.context('utils.article')
def plot_histogram_stacked(
    dict_samples: dict[str, Annotated[np.ndarray, 'batch, sequence']],
    save_dir: str = 'results/',
    save_label: str = '',
    x_log_scale: bool = False,
) -> mpl.figure.Figure:
    # prepare figure
    num_plot = len(dict_samples.keys())
    fig = plt.figure(figsize=(12, 8))
    IL_ratio = 8
    gspec = gridspec.GridSpec(nrows=1, ncols=1+IL_ratio, figure=fig)
    
    global_min = min([samples.min() for samples in dict_samples.values()])
    global_max = max([samples.max() for samples in dict_samples.values()])
    
    # axes for plot
    ax = fig.add_subplot(gspec[0, 0:IL_ratio])
    list_hist = []
    for idx, (model_name, samples) in enumerate(dict_samples.items()):
        _bin_values, _bins, patches = ax.hist(
            samples.flatten(), 
            bins=100, 
            range=(global_min, global_max), 
            histtype='stepfilled',
            alpha=0.3,
            density=True,
            label = model_name.upper()
        )
        list_hist.append(patches[0])
    if x_log_scale:
        ax.set_xscale('symlog', linthresh=1)
    # ax.set_yscale('log')
    # axes for legend    
    legend_ax = fig.add_subplot(gspec[0, -1])
    legend_ax.axis('off')
    legend_ax.legend(
        handles = list_hist,
        labels = list(dict_samples.keys()),
    )
    # suptitle
    fig.suptitle(f'Histogram of Samples')
        
    filename = f'hist_stacked_' + '_'.join(list(dict_samples.keys())) + save_label + '.pdf'
    plt.savefig(os.path.join(save_dir, f'{filename}'), dpi=300)
    
    return fig

@plt.style.context('utils.article')
def plot_psd(
    dict_psd: dict[str, Float[Tensor, 'sequence window_size']],
    save_dir: str = 'results/',
):
    # prepare data
    vmin = min([psd.min() for psd in dict_psd.values()])
    vmax = max([psd.max() for psd in dict_psd.values()])
    cmap = mpl.cm.get_cmap('coolwarm')
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    # prepare fig
    num_plot = len(dict_psd.keys())
    fig = plt.figure(figsize=(10, 2*num_plot))
    imbar_ratio = 8
    gspec = gridspec.GridSpec(nrows=num_plot*imbar_ratio+1, ncols=1, figure=fig)
    
    # axes for plot
    for idx, (key, psd) in enumerate(dict_psd.items()):
        ax = fig.add_subplot(gspec[idx*imbar_ratio:(idx+1)*imbar_ratio, 0])
        image = ax.imshow(psd.T, cmap=cmap, norm=norm, interpolation='bessel', aspect=.3)
        ax.get_xaxis().set_visible(False)
        if idx == len(dict_psd.keys()) - 1:
            ax.set_xlabel('Frequency')
        ax.set_ylabel(f'{key}')
    
    # axes for colorbar
    cbar_ax = fig.add_subplot(gspec[-1, 0])
    cbar = plt.colorbar(image, cax=cbar_ax, aspect=10, orientation='horizontal')
    cbar.ax.set_xlabel('PSD Value')
    
    filename = f'psd.pdf'
    plt.savefig(os.path.join(save_dir, f'{filename}'), dpi=300)
    
    return fig
    
@plt.style.context('utils.article_new')
def plot_ddpm_progressive(
    dict_samples: dict[str|int, Annotated[np.ndarray, 'batch, sequence']],
    save_dir: str = 'results/',
    save_suffix: str = '',
    scale_per_step: bool = True,
    log_scale: bool = False,
):
    """
    dict_samples: dict[str|int, np.ndarray], step: samples
    save_dir: str
    """
    num_steps = len(dict_samples)
    num_samples = next(iter(dict_samples.values())).shape[0]
    ratio = 0.1
    max_val = max([samples.max() for samples in dict_samples.values()])
    min_val = min([samples.min() for samples in dict_samples.values()])
    fig = plt.figure(figsize=(0.5+num_steps, 0.5+num_samples))
    gspec = gridspec.GridSpec(
        nrows=num_samples+1,
        ncols=num_steps+1,
        figure=fig,
        height_ratios=[1]*num_samples+[ratio],
        width_ratios=[ratio]+[1]*num_steps
    )
    cmap = plt.get_cmap('coolwarm')
    for step_idx, (step, samples) in enumerate(dict_samples.items()):
        step_min = samples.min()
        step_max = samples.max()
        for sample_idx, sample in enumerate(samples):
            label_ax = fig.add_subplot(
                gspec[sample_idx, 0],
                frameon=False,
                xlim=[0,1], ylim=[0,1],
                xticks=[], yticks=[],
            )
            label_ax.text(1, 0.5, f'Sample {sample_idx}', 
                          rotation=90,
                          ha='right', va='center', 
                          size='small', weight='bold')
            ax = fig.add_subplot(
                gspec[sample_idx, 1+step_idx],
                frameon=True,
                xticks=[],
                yticks=[],
                ylim=[step_min, step_max] if scale_per_step else [min_val, max_val],
                adjustable='box'
            )
            ax.tick_params(axis='y', labelrotation=45)
            if log_scale:
                norm = mpl.colors.SymLogNorm(vmin=step_min if scale_per_step else min_val,
                                            vmax=step_max if scale_per_step else max_val,
                                            linthresh=0.01, linscale=0.01, base=1000)
            else:
                norm = mpl.colors.Normalize(vmin=step_min if scale_per_step else min_val, 
                                            vmax=step_max if scale_per_step else max_val)
            if log_scale:
                ax.set_yscale('symlog', base=1000)
                ax.set_yticks([])
                # ax.set_yticklabels([])
            ax.scatter(np.arange(sample.shape[0]), sample, s=0.5, c=sample, cmap=cmap, norm=norm)
            ax.plot(sample, linewidth=0.5, c='black', alpha=0.2)
        bottom_ax = fig.add_subplot(gspec[-1, 1+step_idx], frameon=False,
                                    xlim=[0,1], ylim=[0,1],
                                    xticks=[], yticks=[],
        )
        bottom_ax.text(0.5, 0, f'T={step}', ha='center', va='bottom', size='small', weight='bold')
    
    _save_suffix = '' if save_suffix == '' else f'_{save_suffix}'
    filename = 'ddpm_progressive' + _save_suffix + '.png'
    plt.savefig(os.path.join(save_dir, f'{filename}'), dpi=300)
    
@plt.style.context('utils.small_fig_new')
def plot_scatter_marginal_2d(
    dict_samples: dict[str, Annotated[np.ndarray, 'batch, 2']],
    xlabel: str,
    ylabel: str,
    save_dir: str = 'results/',
    save_label: str = '',
) -> mpl.figure.Figure:
    
    name_list = list(dict_samples.keys())
    xy_list = list(dict_samples.values())
    
    contour_levels = [0.95, 0.99]
    contour_level_labels = ['95%', '99%']
    contour_level_cmap = mpl.cm.get_cmap('autumn')

    # Determine the global x and y limits
    x_min = min([xy[:, 0].min() for xy in xy_list])
    x_max = max([xy[:, 0].max() for xy in xy_list])
    y_min = min([xy[:, 1].min() for xy in xy_list])
    y_max = max([xy[:, 1].max() for xy in xy_list])
    
    least_num_samples = min([xy.shape[0] for xy in xy_list])

    fig = plt.figure(figsize=(13, 3)) # for 3 subplots
    gs = gridspec.GridSpec(2, 2*len(xy_list), figure=fig, height_ratios=[4.5, 2], width_ratios=[4, 1]*len(xy_list))

    list_axes = []
    for index, (name, xy) in enumerate(dict_samples.items()):
        # Create main scatter plot
        ax_scatter = plt.subplot(gs[0, 2*index])
        ax_scatter.grid(True)
        x, y = xy[:, 0], xy[:, 1]

        # Create meshgrid for density
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = stats.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)

        # Sort the flattened density values and find levels for contours
        sorted_density = np.sort(f.ravel())
        levels = [sorted_density[int(p * len(sorted_density))] for p in contour_levels]

        # Plot scatter, contour fill, and contour lines
        ax_scatter.scatter(x[:least_num_samples], y[:least_num_samples], s=2, alpha=0.2, c='black', rasterized=True)
        ax_scatter.contourf(xx, yy, f, cmap='Blues', alpha=0.5)
        cont = ax_scatter.contour(xx, yy, f, levels=levels, cmap=contour_level_cmap, linewidths=1., alpha=0.7)
        
        # Annotate the contour lines
        fmt = {lvl: lbl for lvl, lbl in zip(levels, contour_level_labels)}
        ax_scatter.clabel(cont, cont.levels, inline=True, fmt=fmt, fontsize=8, colors='black')

        # Set the same x and y limits for each subplot
        ax_scatter.set_xlim(x_min, x_max)
        ax_scatter.set_ylim(y_min, y_max)

        # Create marginal histograms
        ax_histx = plt.subplot(gs[1, 2*index], sharex=ax_scatter)
        ax_histx.set_box_aspect(1.6/7)
        ax_histy = plt.subplot(gs[0, 2*index + 1], sharey=ax_scatter)
        ax_histx.grid(True, zorder=0); ax_histy.grid(True, zorder=0)
        # ax_histx.set_xticklabels([]); ax_histx.set_yticklabels([])
        # ax_histy.set_xticklabels([]); ax_histy.set_yticklabels([])
        

        # Plot marginal histograms
        ax_histx.hist(x, bins=30, density=True, color='C0', alpha=0.9, zorder=3)
        ax_histy.hist(y, bins=30, density=True, color='C0', alpha=0.9, orientation='horizontal', zorder=3)
        ax_histx.set_yscale('log')
        ax_histy.set_xscale('log')

        # Set titles and labels
        ax_scatter.set_title(f'{name_list[index]}')
        ax_scatter.set_xlabel(xlabel)
        ax_scatter.set_ylabel(ylabel)
        
        ax_histx.set_ylabel('Density [-]')
        ax_histy.set_xlabel('Density [-]')

        # Remove labels and ticks from histograms
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histx.tick_params(axis="y", labelleft=True)
        ax_histy.tick_params(axis="x", labelbottom=True)
        ax_histy.tick_params(axis="y", labelleft=False)
        
        list_axes.append({'scatter':ax_scatter, 'histx':ax_histx, 'histy':ax_histy})

    # global histogram y ticks
    histx_min = min([ax['histx'].get_ylim()[0] for ax in list_axes])
    histx_max = max([ax['histx'].get_ylim()[1] for ax in list_axes])
    histy_min = min([ax['histy'].get_xlim()[0] for ax in list_axes])
    histy_max = max([ax['histy'].get_xlim()[1] for ax in list_axes])
    # print(histx_min, histx_max)
    
    for ax_dict in list_axes:
        ax_dict['histx'].set_ylim(histx_min, histx_max)
        ax_dict['histy'].set_xlim(histy_min, histy_max)
    #     ax_dict['histx'].set_yticks([histx_min, (histx_min*histx_max)**0.5, histx_max])
    #     ax_dict['histy'].set_xticks([histy_min, (histy_min*histy_max)**0.5, histy_max])
    
    os.makedirs(save_dir, exist_ok=True)
    filename = f'scatter_marginal_' + '_'.join(list(dict_samples.keys())) +save_label+ '.pdf'
    plt.savefig(os.path.join(save_dir, f'{filename}'), dpi=300)