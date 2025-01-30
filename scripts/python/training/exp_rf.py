r'''
Create a rectified flow model with a transformer backbone.
Dummy data with a mixture of six 2D Gaussian distributions.
'''
import os

import torch
from einops import rearrange
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from tqdm import tqdm
from ema_pytorch import EMA

from energydiff.diffusion import Transformer1D
from energydiff.diffusion.rectified_flow import RectifiedFlow
from energydiff.utils.initializer import create_rectified_flow
from energydiff.utils.configuration import RectifiedFlowConfig

def create_gmm_data(n_samples, n_features, n_components):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_components, random_state=42)
    return X, y

def visualize_data(X, X_hat=None, save_path=None):
    plt.scatter(X[:, 0], X[:, 1], s=10, c='g')
    if X_hat is not None:
        plt.scatter(X_hat[:, 0], X_hat[:, 1], s=10, color='purple')
    
    save_path = save_path or 'data.png'
    plt.savefig(save_path, dpi=300)
    
def cycle(dataloader: DataLoader):
    " infinitely yield data from dataloader "
    while True:
        for data in dataloader:
            yield data
    
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    epoch_loss_sum = 0.
    it = tqdm(dataloader)
    for batch in it:
        # batch is a tuple/list of tensors (can be length-1)
        batch = batch[0].to(device)
        optimizer.zero_grad()
        loss_terms = model(batch)
        loss = loss_terms['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss_sum += loss.item()
        it.set_description(f'loss: {loss.item():.4f}')
    
    return epoch_loss_sum / len(dataloader)

def train_step(model, dataloader, optimizer, device):
    model.train()
    
    batch = next(dataloader)
    # batch is a tuple/list of tensors (can be length-1)
    batch = batch[0].to(device)
    optimizer.zero_grad()
    loss_terms = model(batch)
    loss = loss_terms['loss']
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def sample_rf(rf_model, batch_size):
    rf_model.eval()
    if isinstance(rf_model, EMA):
        rf_model = rf_model.ema_model
    samples = rf_model.reverse_sample_loop(batch_size)
    
    return samples

@torch.no_grad()
def sample_rf_trajectory(rf_model, batch_size):
    rf_model.eval()
    if isinstance(rf_model, EMA):
        rf_model = rf_model.ema_model
    
    sample_trajectory = []
    for sample_t in rf_model.reverse_sample_progressive(batch_size):
        sample_trajectory.append(sample_t) # shape: (batch_size, 1, seq_length)
    
    return sample_trajectory

def visualize_trajectory(sample_trajectory, true_data=None, save_path=None):
    """
    sample_trajectory: array of shape [batch_size, step, seq_length==2]
    true_data: array of shape [n_samples, n_features]
    """
    if true_data is not None:
        plt.scatter(true_data[:, 0], true_data[:, 1], s=10, c='g',alpha=0.5)
    for sample_idx in range(sample_trajectory.shape[0]):
        plt.plot(sample_trajectory[sample_idx, :, 0], sample_trajectory[sample_idx, :, 1], color='blue', alpha=0.5)
    sample_destination = sample_trajectory[:, -1, :]
    sample_start = sample_trajectory[:, 0, :]
    plt.scatter(sample_start[:, 0], sample_start[:, 1], s=10, color='red')
    plt.scatter(sample_destination[:, 0], sample_destination[:, 1], s=10, color='purple')
    
    save_path = save_path or 'trajectory.png'
    plt.savefig(save_path, dpi=300)

def main():
    # Simple parameters
        # 1000 samples/epoch, 100 epochs ~= 3k iterations
    num_training_steps = 25000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Create a dummy dataset
    data, _ = create_gmm_data(n_samples=1000, n_features=2, n_components=6)
    os.makedirs('rf_results', exist_ok=True)
    visualize_data(data, save_path='rf_results/true_data.png')
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32).unsqueeze(1)) # shape: (n_samples, 1, n_features)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    backbone = Transformer1D(
        dim_base = 512,
        num_in_channel=1,
        dim_out=None,
        type_pos_emb='sinusoidal',
        num_attn_head=8,
        num_decoder_layer=6,
        dim_feedforward=2048,
    ).to(device)
    rf_config = RectifiedFlowConfig(
        prediction_type='noise',
        use_rectified_flow=True,
        schedule_type='logit_normal',
    )
    rf_model = create_rectified_flow(
        base_model=backbone,
        seq_length=2,
        rf_config=rf_config,
        num_discretization_step=10,
    ).to(device)
    rf_ema = EMA(rf_model, beta=0.9999, update_every=5)
    optimizer = torch.optim.Adam(rf_model.parameters(), lr=1e-4)
    
    # Train the model
    dataloader = cycle(dataloader)
    step = 0
    with tqdm(total=num_training_steps) as pbar:
        while step < num_training_steps:
            loss = train_step(rf_model, dataloader, optimizer, device=device)
            rf_ema.update()
            pbar.set_description(f'loss: {loss:.4f}') # batch loss
            
            if (step+1) % (num_training_steps//10) == 0:
                samples = sample_rf(rf_model, batch_size=1000)
                samples_ema = sample_rf(rf_ema, batch_size=1000)
                samples = rearrange(samples, 'b 1 l -> b l').cpu().numpy()
                samples_ema = rearrange(samples_ema, 'b 1 l -> b l').cpu().numpy()
                os.makedirs('rf_results', exist_ok=True)
                visualize_data(X=data, X_hat=samples, 
                            save_path='rf_results/step_{}.png'.format(step+1))
                visualize_data(X=data, X_hat=samples_ema,
                               save_path='rf_results/step_{}_ema.png'.format(step+1))
            if (step+1) % (num_training_steps//5) == 0:
                sample_trajectory = sample_rf_trajectory(rf_model, batch_size=100)
                sample_trajectory = torch.cat(sample_trajectory, dim=1).cpu().numpy() # shape: (batch_size, num_sampling_timestep, seq_length)
                visualize_trajectory(sample_trajectory, true_data=data, 
                                    save_path='rf_results/trajectory_{}.png'.format(step+1))

            step += 1
            pbar.update(1)
    
    print('complete.')

if __name__ == "__main__":
    main()