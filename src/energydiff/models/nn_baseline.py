from typing import Optional, Callable
from multiprocessing import cpu_count

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from einops import rearrange
from einops.layers.torch import Rearrange
import wandb

from energydiff.diffusion.models_1d import Downsample, Upsample

class ResNetLayer(nn.Module):
    "for vae"
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.ln = nn.GroupNorm(1, out_channels) # LayerNorm
        self.residual = nn.Conv1d(in_channels, out_channels, 1, stride, 0) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return F.relu(self.ln(self.conv(x)) + self.residual(x))

class Encoder(nn.Module):
    """
    downsampling_factor: how many times to downsample the input
    """
    def __init__(self, in_channels, out_channels, hiddem_dim, downsampling_factor):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hiddem_dim = hiddem_dim
        self.downsampling_factor = downsampling_factor
        
        _layers = [
            ResNetLayer(in_channels, hiddem_dim, 7, 1, 3),
        ]
        for _ in range(downsampling_factor-1):
            _layers.append(ResNetLayer(hiddem_dim, hiddem_dim, 3, 1, 1))
            _layers.append(Downsample(hiddem_dim, hiddem_dim))
            
        _layers.append(ResNetLayer(hiddem_dim, hiddem_dim, 3, 1, 1))
        _layers.append(Downsample(hiddem_dim, out_channels))
        
        self.layers = nn.Sequential(*_layers)
        
    def forward(self, x):
        " (batch, channel_in, sequence) -> (batch, channel_out, sequence//2^f) "
        return self.layers(x)
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, upsampling_factor):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.upsampling_factor = upsampling_factor
        
        _layers = [
            ResNetLayer(in_channels, hidden_dim, 3, 1, 1),
        ]
        for _ in range(upsampling_factor-1):
            _layers.append(ResNetLayer(hidden_dim, hidden_dim, 3, 1, 1))
            _layers.append(Upsample(hidden_dim, hidden_dim))
        
        _layers.append(ResNetLayer(hidden_dim, hidden_dim, 3, 1, 1))
        _layers.append(Upsample(hidden_dim, out_channels))
        
        self.layers = nn.Sequential(*_layers)
        
    def forward(self, x):
        " (batch, channel_in, sequence//f) -> (batch, channel_out, sequence) "
        return self.layers(x)
    
class MLPEncoder(nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        _layers = [
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
        ]
        for _ in range(3):
            _layers.append(nn.LayerNorm(self.hidden_dim),)
            _layers.append(nn.Linear(self.hidden_dim, self.hidden_dim),)
            _layers.append(nn.GELU()),
            
        _layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.layers = nn.Sequential(*_layers)
        
    def forward(self, x):
        "x shape: (batch, 1, input_dim)"
        encoded = self.layers(x) # shape: (batch, 1, output_dim)
        return encoded
    
class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        _layers = [
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
        ]
        for _ in range(3):
            _layers.append(nn.LayerNorm(self.hidden_dim),)
            _layers.append(nn.Linear(self.hidden_dim, self.hidden_dim),)
            _layers.append(nn.GELU()),
            
        _layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.layers = nn.Sequential(*_layers)
        
    def forward(self, x):
        "x shape: (batch, 1, input_dim)"
        return self.layers(x)
    
class KLRegAutoencoder(nn.Module):
    def __init__(self, in_channels, seq_length, bottleneck_channels, hidden_dim, downsampling_factor):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.downsampling_factor = downsampling_factor
        self.seq_length = seq_length
        self.bottleneck_channels = bottleneck_channels
        # self.encoder = Encoder(in_channels, bottleneck_channels * 2, hidden_dim, downsampling_factor)
        # self.decoder = Decoder(bottleneck_channels, in_channels, hidden_dim, downsampling_factor)
        assert in_channels == 1
        self.encoder = MLPEncoder(hidden_dim, seq_length, bottleneck_channels * 2) # shape: (batch, 1, seq_length) -> (batch, 1, bottleneck_channels*2)
        self.decoder = MLPDecoder(hidden_dim, bottleneck_channels, seq_length) # shape: (batch, 1, bottleneck_channels) -> (batch, 1, seq_length)
        
    def _get_mean_logvar(self, encoded):
        """
        1. encoded shape: (batch, 1, bottleneck_channels*2)
        2. encoded shape: (batch, 2, bottleneck_channels)
        """
        if isinstance(self.encoder, Encoder): # case 2
            encoded_mean, encoded_logvar = encoded.chunk(2, dim=1)
        else: # case 1
            encoded_mean, encoded_logvar = encoded.chunk(2, dim=2)
        return encoded_mean, encoded_logvar

    def encode(self, x):
        encoded = self.encoder(x)
        encoded_mean, encoded_logvar = self._get_mean_logvar(encoded)
        return encoded_mean
    
    def decode(self, z):
        return self.decoder(z)
    
    def kl_reg(self, encoded):
        encoded_mean, encoded_logvar = self._get_mean_logvar(encoded)
        return -0.5 * torch.sum(1 + encoded_logvar - encoded_mean.pow(2) - encoded_logvar.exp(), dim=1).mean()
    
    def forward(self, x):
        encoded = self.encoder(x)
        kl_loss = self.kl_reg(encoded)
        encoded_mean, encoded_logvar = self._get_mean_logvar(encoded)
        z = encoded_mean + torch.randn_like(encoded_mean) * torch.exp(0.5 * encoded_logvar)
        decoded = self.decoder(z)
        return decoded, kl_loss
    
class PLVAE1D(pl.LightningModule):
    def __init__(
        self,
        # model configs
        seq_length: int,
        in_channels: int,
        bottleneck_channels: int,
        hidden_dim: int,
        downsampling_factor: int,
        
        # training configs
        trainset: Optional[Dataset] = None,
        valset: Optional[Dataset] = None,
        batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        kl_weight: float = 0.01,  # Beta weight for KL divergence term
        
        # validation configs
        # metrics_factory: Optional[Callable] = None,
        num_val_samples: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['vae_model', 'metrics_factory'])
        
        # Model
        vae_model = KLRegAutoencoder(
            seq_length=seq_length,
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,  # can be made configurable
            hidden_dim=hidden_dim,  # can be made configurable
            downsampling_factor=downsampling_factor  # can be made configurable
        )
        self.model = vae_model
        
        # Training params
        self.lr = lr
        self.betas = betas
        self.kl_weight = kl_weight
        
        # Data
        self.trainset = trainset
        self.valset = valset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        
        # Validation
        self.metrics = None
        self.num_val_samples = num_val_samples
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=self.betas
        )
        return optimizer
    
    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=True,
            persistent_workers=True,
        )
    
    def val_dataloader(self):
        if self.valset is None:
            return None
        return DataLoader(
            self.valset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=True,
            persistent_workers=True,
        )
    
    def training_step(self, batch, batch_idx):
        "batch shape: (batch, 1, full_seq_len)"
        self.train()
        profiles, conditions = batch  # Assuming same data format as diffusion
        
        # Forward pass
        reconstructed, kl_loss = self.model(profiles)
        
        # Compute reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, profiles)
        
        # Total loss with KL weight (beta-VAE formulation)
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        # Logging
        self.log('vae/train/total_loss', total_loss.item(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('vae/train/recon_loss', recon_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('vae/train/kl_loss', kl_loss, on_step=True, on_epoch=True, sync_dist=True)
        
        return total_loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        "batch shape: (batch, 1, full_seq_len)"
        self.eval()
        profiles, conditions = batch
        
        # Forward pass
        reconstructed, kl_loss = self.model(profiles)
        
        # Compute losses
        recon_loss = F.mse_loss(reconstructed, profiles)
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        # Logging
        self.log('vae/val/total_loss', total_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('vae/val/recon_loss', recon_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('vae/val/kl_loss', kl_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        # Update metrics if they exist
        if self.metrics is not None:
            # Reshape tensors if needed (assuming same format as diffusion)
            recon_flat = rearrange(reconstructed, 'b c l -> b (l c)')
            real_flat = rearrange(profiles, 'b c l -> b (l c)')
            
            # Gather samples across GPUs
            gathered_recon = self.all_gather(recon_flat)
            gathered_real = self.all_gather(real_flat)
            
            if self.trainer.is_global_zero:
                recon_concat = torch.cat([batch for batch in gathered_recon], dim=0)
                real_concat = torch.cat([batch for batch in gathered_real], dim=0)
                self.metrics.update(recon_concat, real_concat)
                
        return total_loss
    
    def on_validation_epoch_end(self):
        if self.metrics is None:
            return
            
        # Compute metrics
        metric_results = self.metrics.compute()
        
        # Log metrics only on global zero
        if self.trainer.is_global_zero:
            for k, v in metric_results.items():
                self.log(f'val/{k}', v, sync_dist=False)
                
        # Reset metrics
        self.metrics.reset()
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    
    @torch.no_grad()
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """Generate samples by sampling from latent space"""
        self.eval()
        # Sample from standard normal distribution (prior)
        if isinstance(self.model.decoder, Decoder):
            z = torch.randn(num_samples, self.model.bottleneck_channels, 
                       self.model.seq_length // (2 ** self.model.downsampling_factor), 
                       device=self.device)
        else:
            z = torch.randn(num_samples, 1, self.model.bottleneck_channels, device=self.device)
        
        # Decode
        samples = self.model.decode(z)
        return samples
    
    # def on_train_epoch_end(self):
    #     """Generate and log sample visualizations at the end of each epoch"""
    #     if self.trainer.is_global_zero:
    #         samples = self.generate_samples(min(16, self.num_val_samples))
    #         # Log to wandb if available
    #         try:
    #             wandb.log({
    #                 "samples/generated": samples.squeeze(1),
    #                 "train/epoch_recon_loss": self.train_recon_loss,
    #                 "train/epoch_kl_loss": self.train_kl_loss,
    #                 "epoch": self.current_epoch
    #             })
    #         except:
    #             pass
    
    def encode(self, x):
        """Convenience method to access encoder"""
        return self.model.encode(x)
    
    def decode(self, z):
        """Convenience method to access decoder"""
        return self.model.decode(z)
    
class ResBlock(nn.Module):
    "for gan"
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding))
        self.conv2 = spectral_norm(nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding))
        self.ln1 = nn.GroupNorm(1, out_channels)  # Layer Norm
        self.ln2 = nn.GroupNorm(1, out_channels)  # Layer Norm
        self.residual = nn.Conv1d(in_channels, out_channels, 1, stride) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.ln1(h)
        h = F.leaky_relu(h, 0.2)
        h = self.conv2(h)
        h = self.ln2(h)
        return F.leaky_relu(h + self.residual(x), 0.2)

class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_channels=1, dim_base=32, sequence_length=1440):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.dim_base = dim_base
        assert sequence_length in [24, 48, 96, 1440]
        
        # Initial projection from latent space
        if sequence_length == 1440:
            initial_sequence_length = 90  # Start from 45 for sequence_length=1440
        else:
            initial_sequence_length = 6 # Start from 6 for other sequences
        self.initial = nn.Sequential(
            nn.Linear(latent_dim, dim_base * 8 * initial_sequence_length), # input = latent z. shape: (batch_size, latent_dim)
            Rearrange('b (c l) -> b c l', c=dim_base * 8)
        )
        
        # Progressive upsampling with residual blocks
        self.layers = nn.ModuleList([])
        _out_seq_len = initial_sequence_length
        in_out_channels = [
            (dim_base * 8, dim_base * 8),
            (dim_base * 8, dim_base * 4),
            (dim_base * 4, dim_base * 4),
            (dim_base * 4, dim_base * 2),
            (dim_base * 2, dim_base),
        ]
        for _in_channels, _out_channels in in_out_channels:
            self.layers.append(
                ResBlock(_in_channels, _out_channels) # ResBlock(in_channels, out_channels)
            )
            if _out_seq_len < sequence_length:
                self.layers.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=False))
                _out_seq_len *= 2
        
        assert _out_seq_len == sequence_length
        
        # Final output projection
        self.final = nn.Sequential(
            ResBlock(dim_base, dim_base),
            nn.Conv1d(dim_base, out_channels, 7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, z):
        """
        z: (batch_size, latent_dim)
        returns: (batch_size, out_channels, sequence_length)
        """
        x = self.initial(z)
        for layer in self.layers:
            x = layer(x)
        return self.final(x)

class WGANDiscriminator(nn.Module):
    """Enhanced Discriminator with spectral normalization"""
    def __init__(self, in_channels=1, dim_base=32, sequence_length=1440):
        super().__init__()
        self.dim_base = dim_base
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        assert sequence_length in [24, 48, 96, 1440]

        if sequence_length == 24:
            final_feat_size = dim_base*8 * 2
        else:
            final_feat_size = dim_base*8 * (sequence_length//16)
        
        self.model = nn.Sequential(
            spectral_norm(nn.Conv1d(in_channels, dim_base, 5, 2, 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            spectral_norm(nn.Conv1d(dim_base, dim_base*2, 5, 2, 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            spectral_norm(nn.Conv1d(dim_base*2, dim_base*4, 5, 2, 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            spectral_norm(nn.Conv1d(dim_base*4, dim_base*8, 5, 2, 2)),
            nn.LeakyReLU(0.2),
            
            Rearrange('b c l -> b (c l)'),
            spectral_norm(nn.Linear(final_feat_size, 1))
        )

    def forward(self, x):
        return self.model(x)  # No sigmoid - using Wasserstein distance
    
class MLPGenerator(nn.Module):
    def __init__(self, latent_dim=100, dim_base=32, sequence_length=1440):
        "dim_base (suggest 32) * [8 -> 16 -> 16 -> 16 -> sequence_length]"
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.dim_base = dim_base
        assert sequence_length in [24, 48, 96, 1440]
        
        # Initial projection from latent space
        self.initial = nn.Sequential(
            nn.Linear(latent_dim, dim_base * 8), # input = latent z. shape: (batch_size, latent_dim)
            nn.LeakyReLU(0.2),
        )
        
        # Progressive upsampling
        self.layers = nn.ModuleList([])
        in_out_channels = [
            (dim_base * 8, dim_base * 16),
            (dim_base * 16, dim_base * 16),
            (dim_base * 16, dim_base * 16),
            (dim_base * 16, dim_base * 16),
        ]
        for _in_channels, _out_channels in in_out_channels:
            self.layers.append(nn.LayerNorm(_in_channels))
            self.layers.append(nn.Linear(_in_channels, _out_channels))
            self.layers.append(nn.LeakyReLU(0.2))
            
        self.layers.append(nn.Linear(dim_base * 16, sequence_length))
        self.layers.append(Rearrange('b l -> b 1 l'))
        self.layers.append(nn.Tanh())
        
    def forward(self, z):
        """
        z: (batch_size, latent_dim)
        returns: (batch_size, 1, sequence_length)
        """
        x = self.initial(z)
        for layer in self.layers:
            x = layer(x)
        return x
    
class MLPWGANDiscriminator(nn.Module):
    def __init__(self, latent_dim=100, dim_base=32, sequence_length=1440):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.dim_base = dim_base
        assert sequence_length in [24, 48, 96, 1440]

        # Initial projection from input space
        self.initial = nn.Sequential(
            Rearrange('b 1 l -> b l', l=sequence_length), # input = sequence. shape: (batch_size, 1, sequence_length)
            nn.Linear(sequence_length, dim_base * 16),
            nn.LeakyReLU(0.2),
        )

        # Progressive downsampling
        self.layers = nn.ModuleList([])
        in_out_channels = [
            (dim_base * 16, dim_base * 16),
            (dim_base * 16, dim_base * 16),
            (dim_base * 16, dim_base * 8),
            (dim_base * 8, dim_base * 8),
            (dim_base * 8, dim_base * 4),
        ]
        for _in_channels, _out_channels in in_out_channels:
            self.layers.append(nn.LayerNorm(_in_channels))
            self.layers.append(spectral_norm(nn.Linear(_in_channels, _out_channels)))
            self.layers.append(nn.LeakyReLU(0.2))
        
        self.layers.append(spectral_norm(nn.Linear(dim_base * 4, 1)))

    def forward(self, x):
        """
        x: (batch_size, 1, sequence_length)
        returns: (batch_size, 1)
        """
        x = self.initial(x)
        for layer in self.layers:
            x = layer(x)
        return x

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Gradient penalty for WGAN-GP"""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=real_samples.device)
    alpha = alpha.expand_as(real_samples)
    
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    d_interpolated = discriminator(interpolated)
    
    fake = torch.ones(d_interpolated.size(), device=interpolated.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class PLWGAN1D(pl.LightningModule):
    def __init__(
        self,
        # model configs
        # generator: nn.Module,
        # discriminator: nn.Module,
        latent_dim: int = 100,
        out_channels: int = 1,
        dim_base: int = 32,
        sequence_length: int = 24,
        in_channels: int = 1,
        
        # training configs
        trainset: Optional[Dataset] = None,
        valset: Optional[Dataset] = None,
        batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        lr: float = 1e-4,
        betas: tuple = (0.5, 0.9),
        n_critic: int = 5,
        lambda_gp: float = 10.0,
        
        # validation configs
        # metrics_factory: Optional[Callable] = None,
        num_val_samples: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['generator', 'discriminator', 'metrics_factory'])
        
        generator = Generator(
            latent_dim=latent_dim,
            out_channels=out_channels,
            dim_base=dim_base,
            sequence_length=sequence_length
        )
        discriminator = WGANDiscriminator(
            in_channels=in_channels,
            dim_base=dim_base,
            sequence_length=sequence_length
        )
        # generator = MLPGenerator(
        #     latent_dim=latent_dim,
        #     dim_base=dim_base,
        #     sequence_length=sequence_length
        # )
        # discriminator = MLPWGANDiscriminator(
        #     latent_dim=latent_dim,
        #     dim_base=dim_base,
        #     sequence_length=sequence_length
        # )
        
        # Models
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        
        # Training params
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.lr = lr
        self.betas = betas
        
        # Data
        self.trainset = trainset
        self.valset = valset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        
        # Validation
        self.metrics = None
        self.num_val_samples = num_val_samples
        
        # Training state
        self.critic_step = 0
        
        self.automatic_optimization = False
    
    def configure_optimizers(self):
        g_opt = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas=self.betas
        )
        d_opt = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=self.betas
        )
        return [d_opt, g_opt], []  # No schedulers
    
    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=True,
            persistent_workers=True,
        )
    
    def val_dataloader(self):
        if self.valset is None:
            return None
        return DataLoader(
            self.valset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=True,
            persistent_workers=True,
        )
        
    def forward(self, z):
        "required?"
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        "batch shape: (batch, 1, full_seq_len)"
        self.train()
        opt_d, opt_g = self.optimizers()
        
        profiles, _ = batch
        batch_size = profiles.shape[0]
        
        # Train Discriminator
        self.toggle_optimizer(opt_d)
        
        # Generate fake data
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        fake_profiles = self(z)
        
        # Real and fake predictions
        real_pred = self.discriminator(profiles)
        fake_pred = self.discriminator(fake_profiles.detach())
        
        # Wasserstein loss
        d_loss = fake_pred.mean() - real_pred.mean()
        
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)
        
        # Logging
        self.log('gan/train/d_loss', d_loss, prog_bar=True, sync_dist=True)
        self.log('gan/train/wasserstein_d', -d_loss, sync_dist=True)
        
        # Train Generator (every n_critic steps)
        self.critic_step += 1
        if self.critic_step % self.hparams.n_critic == 0:
            self.toggle_optimizer(opt_g)
            
            # Generate fake data
            z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
            fake_profiles = self(z)
            fake_pred = self.discriminator(fake_profiles)
            
            g_loss = -fake_pred.mean()
            
            self.manual_backward(g_loss)
            opt_g.step()
            opt_g.zero_grad()
            self.untoggle_optimizer(opt_g)
            
            # Logging
            self.log('gan/train/g_loss', g_loss, prog_bar=True, sync_dist=True)
            
        return None
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        "batch shape: (batch, 1, full_seq_len)"
        self.eval()
        profiles, conditions = batch
        batch_size = profiles.shape[0]
        
        # Compute validation losses
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_profiles = self.generator(z)
        
        real_pred = self.discriminator(profiles)
        fake_pred = self.discriminator(fake_profiles)
        
        # Wasserstein distance
        w_dist = fake_pred.mean() - real_pred.mean()
        
        # Logging
        self.log('gan/val/wasserstein_distance', w_dist, sync_dist=True)
        
        # Update metrics if they exist
        if self.metrics is not None:
            # Generate samples for metrics computation
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_samples = self.generator(z)
            
            # Reshape tensors if needed (assuming same format as diffusion)
            fake_samples = rearrange(fake_samples, 'b c l -> b (l c)')
            real_samples = rearrange(profiles, 'b c l -> b (l c)')
            
            # Gather samples across GPUs
            gathered_fake = self.all_gather(fake_samples)
            gathered_real = self.all_gather(real_samples)
            
            if self.trainer.is_global_zero:
                fake_concat = torch.cat([batch for batch in gathered_fake], dim=0)
                real_concat = torch.cat([batch for batch in gathered_real], dim=0)
                self.metrics.update(fake_concat, real_concat)
                
        return w_dist
    
    def on_validation_epoch_end(self):
        if self.metrics is None:
            return
            
        # Compute metrics
        metric_results = self.metrics.compute()
        
        # Log metrics only on global zero
        if self.trainer.is_global_zero:
            for k, v in metric_results.items():
                self.log(f'val/{k}', v, sync_dist=False)
                
        # Reset metrics
        self.metrics.reset()
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    
    @torch.no_grad()
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """Generate samples for visualization/evaluation"""
        self.eval()
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        samples = self.generator(z)
        return samples
    
    # def on_train_epoch_end(self):
    #     """Generate and log sample visualizations at the end of each epoch"""
    #     if self.trainer.is_global_zero:
    #         samples = self.generate_samples(min(16, self.num_val_samples))
    #         # Log to wandb if available
    #         try:
    #             wandb.log({
    #                 "samples": samples.squeeze(1),
    #                 "epoch": self.current_epoch
    #             })
    #         except:
    #             pass