"calculate the metrics with bootstrapping"
import math
import hashlib

import torch
import wandb
from einops import rearrange
from energydiff.utils.argument_parser import inference_parser
from energydiff.utils.argument_parser import argument_parser
from energydiff.utils.initializer import create_dataset, get_task_profile_condition
from energydiff.utils.eval import (
    MultiMetric, MkMMD, source_mean, source_std, 
    target_mean, target_std, kl_divergence, 
    ws_distance, ks_test_d, UMAPEvalCollection, 
    get_mapper_label, calculate_frechet
)

NUM_BOOTSTRAP = 10

def get_file_md5(filepath):
    md5_hash = hashlib.md5()
    with open(filepath, 'rb') as f:
        # Read in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

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
        
        print(f"Loaded generated data from {model_name}, md5: {get_file_md5(data_dir+'/generated_data.pt')}")
    
    return generated_datasets

def subsample(data, n_samples=None, fraction=None):
    """data: array of [B, L]
    """
    full_size = data.shape[0]
    assert n_samples or fraction, "either n_samples or fraction should be provided"
    if n_samples is not None: assert n_samples < full_size, "n_samples should be less than the full size"
    if fraction is not None: assert math.floor(fraction * full_size) < full_size, "fraction should be less than 1"
    if n_samples is None and fraction is not None:
        n_samples = math.floor(fraction * full_size)
        
    
    indices = torch.randperm(full_size)[:n_samples]
    return data[indices]

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
        return results
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
    
    # Setup wandb
    wandb.init(
        project=config.wandb_project,
        name=f"test-{config.time_id}",
        job_type='testing',
        tags=['testing'],
    )
    
    # Get original run
    api = wandb.Api()
    original_run = api.run(f"{config.wandb_project}/{config.wandb_id}")
    
    # Get real data
    dataset_collection = create_dataset(config.data)
    all_profile, _ = get_task_profile_condition(
        dataset_collection,
        season=config.data.train_season,
        conditioning=config.model.conditioning
    )
    target = torch.cat([all_profile['val'], all_profile['test']], dim=0)
    
    if config.data.vectorize:
        target = dataset_collection.inverse_vectorize_fn(target, style=config.data.style_vectorize)
    
    # Get artifacts
    generated_all = load_all_generated_data(api, config)
    
    # Setup metrics
    mkmmd = MkMMD(
        kernel_type='rbf',
        num_kernel=1,
        kernel_mul=2.0,
        coefficient='auto'
    )
    umap_eval = UMAPEvalCollection(full_dataset_name=get_mapper_label(config.data))
    dict_eval_fn = {
        'MkMMD': mkmmd, # for 3d tensor, in dim=1, use the central channel
        'DirectFD': calculate_frechet,
        'kl_divergence': kl_divergence,
        'ws_distance': ws_distance,
        'ks_test_d': ks_test_d,
    }
    umap_eval_metrics = umap_eval.generate_eval_sequence()
    for metric_name, metric_fn in umap_eval_metrics:
        dict_eval_fn[metric_name] = metric_fn
    metrics = MultiMetric(dict_eval_fn, compute_on_cpu=False)
    
    # Evaluate each run
    metrics_final = {}
    for model_name, generated_data in generated_all.items():
        source = generated_data
        print('-----------------------------------')
        print("Model: ", model_name)
        print("source shape: ", source.shape)
        print("target shape: ", target.shape)
        _result_bootstrap = []
        for i in range(NUM_BOOTSTRAP):
            source_subsample = subsample(source, fraction=0.5)
            target_subsample = subsample(target, fraction=0.5)
            results = get_metric_results(source_subsample, target_subsample, metrics)
            _result_bootstrap.append(results)
        # Get mean and std
        results = {}
        for metric_name in _result_bootstrap[0].keys():
            values = [result[metric_name] for result in _result_bootstrap]
            results[metric_name] = {
                'mean': torch.stack(values).mean().item(),
                'std': torch.stack(values).std().item()
            }
        
        # log per model
        wandb.log({
            f"{model_name}/{metric_name}": value['mean']
            for metric_name, value in results.items()
        })
        # log per metric
        wandb.log({
            f"comparison/{metric_name}/{model_name}": value['mean']
            for metric_name, value in results.items()
        })
        metrics_final[model_name] = results
        
        print("Model: ", model_name)
        for metric_name, value in results.items():
            print(f"{metric_name}: {value['mean']:.4f} ± {value['std']:.4f}")
    
    # Create comparison table
    table_data = []
    for model_name, model_results in metrics_final.items():
        row = [model_name] + [model_results[metric_name]['mean'] for metric_name in model_results.keys()]
        table_data.append(row)
        _std_row = [model_name + "_std"] + [model_results[metric_name]['std'] for metric_name in model_results.keys()]
        table_data.append(_std_row)
    
    table = wandb.Table(
        data=table_data,
        columns=['model'] + [f"{metric_name}" for metric_name in dict_eval_fn.keys()],
    )   # columns=["model"] + list(metrics_final[list(metrics_final.keys())[0]].keys()) 
    wandb.log({"comparison_table": table})


if __name__ == '__main__':
    main()