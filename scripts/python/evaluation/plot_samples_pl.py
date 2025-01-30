import os
import logging
import torch
import wandb
from einops import rearrange
from energydiff.utils.argument_parser import inference_parser
from energydiff.utils.argument_parser import argument_parser
from energydiff.utils.initializer import create_dataset, get_task_profile_condition
from energydiff.utils.plot import plot_sampled_data_v2

def load_all_generated_data(api, config) -> dict:
    # Get all artifacts in the project
    run = api.run(f"{config.wandb_project}/{config.wandb_id}")
    artifacts = [_ar for _ar in run.logged_artifacts() if _ar.type == 'generated_data']
    name_all_models = set([artifact.name.split('_generated_data-')[0] for artifact in artifacts])

    generated_datasets = {}
    # Look for artifacts with names ending in _generated_data
    for model_name in name_all_models:
        artifact = api.artifact(
            f"{config.wandb_project}/{model_name}_generated_data-{config.wandb_id}:latest",
        )
        data_dir = artifact.download()
        data = torch.load(data_dir+"/generated_data.pt", map_location='cpu')
        
        generated_datasets[model_name] = data
        
        print(f"Loaded generated data from {model_name}")
    
    return generated_datasets

def get_metric_results(source, target, metrics) -> dict:
    source = rearrange(source, "b c l -> b (l c)")
    # eval functions
    target = rearrange(target, 'b c l -> b (l c)')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    try:
        source = source.to(device)
        target = target.to(device)
        metrics = metrics.to(device)
        results = metrics(source, target)
    except Exception as e:
        print(f"Error: {e}")
        device = torch.device('cpu')
        source = source.to(device)
        target = target.to(device)
        metrics = metrics.to(device)
        results = metrics(source, target)
        
    return results
    
def main():
    config = inference_parser()
    
    # Get original run
    api = wandb.Api()
    original_run = wandb.init(project=config.wandb_project, id=config.wandb_id, resume='must')
    
    # Get real data
    dataset_collection = create_dataset(config.data)
    all_profile, _ = get_task_profile_condition(
        dataset_collection,
        season=config.data.train_season,
        conditioning=config.model.conditioning
    )
    target = torch.cat([all_profile['val'], all_profile['test']], dim=0)
    
    # Get artifacts
    generated_all = load_all_generated_data(api, config)
    
    # Denormalize
    scaling_factor = dataset_collection.dataset.scaling_factor
    target = dataset_collection.denormalize_fn(target, *scaling_factor).squeeze(1) # [b, l]
    for k, v in generated_all.items():
        generated_all[k] = dataset_collection.denormalize_fn(v, *scaling_factor).squeeze(1)
        
    # De-vectorize
    if config.data.vectorize:
        target = dataset_collection.inverse_vectorize_fn(target, style=config.data.style_vectorize)
        target = rearrange(target, 'b 1 l -> b l')
        
    # Plot
    fig = plot_sampled_data_v2(
        samples={
            'Real Data': target,
            'GMM': generated_all['gmm'],
            'EnergyDiff': generated_all['ddpm'],
            'E.Diff Calibrated': generated_all['ddpm_calibrated'],
        },
        save_filepath=f"results/figures/samples/samples_{config.time_id}_real_gmm_ddpm_ddpmc.pdf",
        num_samples_to_plot=100,
        seed=1234,
    )
    
    # Log to wandb 
    original_run.log({"samples": wandb.Image(fig)})
    
    try:
        fig = plot_sampled_data_v2(
            samples={
                'VAE': generated_all['vae'],
                'VAE Calibrated': generated_all['vae_calibrated'],
                'GAN': generated_all['gan'],
                'GAN Calibrated': generated_all['gan_calibrated'],
            },
            save_filepath=f"results/figures/samples/samples_{config.time_id}_vae_vaec_gan_ganc.pdf",
            num_samples_to_plot=100,
            seed=1234,
        )
        original_run.log({"samples+": wandb.Image(fig)})
    except:
        pass

if __name__ == '__main__':
    main()