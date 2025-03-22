import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass


@dataclass
class VSAEMixConfig:
    """Configuration for Variational Sparse Autoencoder with Gaussian Mixture Prior"""
    d_input: int             # Input dimension (e.g., d_mlp)
    d_hidden: int            # Hidden dimension (usually d_input * dict_mult)
    dict_mult: int = 8       # Dictionary multiplier (how many times larger is d_hidden than d_input)
    kl_coeff: float = 3e-4   # KL divergence coefficient (equivalent to l1_coeff in SAE)
    bias_decay: float = 0.95  # Decay rate for bias updates
    var_flag: int = 0        # Flag to determine if variance is learned (0: fixed, 1: learned)
    
    # Correlation structure parameters
    n_correlated_pairs: int = 0     # Number of correlated feature pairs
    n_anticorrelated_pairs: int = 0  # Number of anticorrelated feature pairs
    correlation_prior_scale: float = 1.0  # Scale of prior means for correlated features
    
    # Resample parameters
    dead_neuron_threshold: float = 1e-8  # Threshold for identifying dead neurons
    dead_neuron_window: int = 400        # Window to check for dead neurons
    resample_interval: int = 3000        # How often to check and resample dead neurons
    resample_scale: float = 0.2         # Scale of new random vectors for dead neurons
    
    # Training parameters
    lr: float = 1e-4          # Learning rate
    batch_size: int = 4096    # Batch size
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32


class VSAEMixGaussian(nn.Module):
    """
    Variational Sparse Autoencoder with Gaussian Mixture Prior
    
    This extends the isotropic Gaussian VSAE by using a mixture of Gaussians
    as the prior distribution to better model correlated and anti-correlated
    feature pairs.
    
    The prior means are structured as follows:
    - Correlated pairs: Both features have positive means (e.g., +1)
    - Anti-correlated pairs: One feature has positive mean, the other negative (e.g., +1, -1)
    - Uncorrelated features: Zero mean (standard prior)
    
    This structure encourages the model to learn representations that respect
    the known correlation patterns in the data.
    """
    
    def __init__(self, config: VSAEMixConfig):
        super().__init__()
        self.config = config
        
        # Main parameters for encoder mean
        self.W_enc = nn.Parameter(torch.empty(config.d_input, config.d_hidden, dtype=config.dtype))
        self.b_enc = nn.Parameter(torch.zeros(config.d_hidden, dtype=config.dtype))
        
        # Parameters for encoder variance (only used when var_flag=1)
        if config.var_flag == 1:
            self.W_enc_var = nn.Parameter(torch.empty(config.d_input, config.d_hidden, dtype=config.dtype))
            self.b_enc_var = nn.Parameter(torch.zeros(config.d_hidden, dtype=config.dtype))
        
        # Decoder parameters
        self.W_dec = nn.Parameter(torch.empty(config.d_hidden, config.d_input, dtype=config.dtype))
        self.b_dec = nn.Parameter(torch.zeros(config.d_input, dtype=config.dtype))
        
        # Initialize parameters
        self._init_parameters()
        
        # Initialize the prior means for correlated and anticorrelated features
        self.prior_means = self._initialize_prior_means()
        
        # Initialize tracking variables for dead neurons
        self.activation_history = []
        
        self.to(config.device)
    
    def _init_parameters(self):
        """Initialize parameters using Kaiming uniform"""
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        
        if self.config.var_flag == 1:
            nn.init.kaiming_uniform_(self.W_enc_var)
        
        # Normalize decoder weights
        self.normalize_decoder()
    
    def _initialize_prior_means(self) -> torch.Tensor:
        """
        Initialize the prior means for the latent variables based on 
        the specified correlation structure.
        
        Returns:
            prior_means: Tensor of shape [d_hidden] with means for the prior distribution
        """
        means = torch.zeros(self.config.d_hidden, device=self.config.device, dtype=self.config.dtype)
        scale = self.config.correlation_prior_scale
        
        # Process correlated pairs
        for i in range(self.config.n_correlated_pairs):
            # Both features in a correlated pair have positive means
            means[2*i] = scale
            means[2*i + 1] = scale
        
        # Process anticorrelated pairs
        offset = 2 * self.config.n_correlated_pairs
        for i in range(self.config.n_anticorrelated_pairs):
            # First feature has positive mean, second has negative mean
            means[offset + 2*i] = scale
            means[offset + 2*i + 1] = -scale
        
        # The remaining features have zero mean (standard prior)
        return means
    
    @torch.no_grad()
    def normalize_decoder(self):
        """Normalize decoder weights to have unit norm"""
        norm = self.W_dec.norm(dim=-1, keepdim=True)
        self.W_dec.data = self.W_dec.data / norm.clamp(min=1e-6)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the variational autoencoder
        
        Args:
            x: Input tensor of shape [batch_size, d_input]
            
        Returns:
            Dict containing loss components and intermediate activations
        """
        # Center the input using the decoder bias
        x_cent = x - self.b_dec
        
        # Encode to get mean of latent distribution
        mu = F.relu(x_cent @ self.W_enc + self.b_enc)
        
        # Get log variance of latent distribution
        if self.config.var_flag == 1:
            log_var = F.relu(x_cent @ self.W_enc_var + self.b_enc_var)
        else:
            # Fixed variance when var_flag=0
            log_var = torch.zeros_like(mu)
        
        # Sample from the latent distribution using reparameterization trick
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_recon = z @ self.W_dec + self.b_dec
        
        # Compute losses
        l2_loss = ((x_recon - x) ** 2).sum(-1).mean()  # Reconstruction loss
        kl_loss = self.compute_kl_divergence(mu, log_var)  # KL divergence with mixture prior
        loss = l2_loss + self.config.kl_coeff * kl_loss
        
        # Record activations for tracking dead neurons
        frac_active = (mu.abs() > self.config.dead_neuron_threshold).float().mean(0)
        self.activation_history.append(frac_active.detach().cpu())
        
        # Only keep the most recent window of activations
        if len(self.activation_history) > self.config.dead_neuron_window:
            self.activation_history.pop(0)
        
        return {
            "loss": loss,
            "l2_loss": l2_loss,
            "kl_loss": kl_loss,
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "x_recon": x_recon,
            "frac_active": frac_active
        }
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Apply the reparameterization trick:
        z = mu + eps * sigma, where eps ~ N(0, 1)
        
        Args:
            mu: Mean of latent distribution, shape [batch_size, d_hidden]
            log_var: Log variance of latent distribution, shape [batch_size, d_hidden]
            
        Returns:
            Sampled latent variable z, shape [batch_size, d_hidden]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def compute_kl_divergence(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between q(z|x) = N(mu, sigma^2) and 
        the mixture prior distribution p(z) with structured means.
        
        For a Gaussian with non-zero mean prior:
        KL(N(mu, sigma^2) || N(prior_mu, 1)) = 
            0.5 * [log(1/sigma^2) + sigma^2 + (mu-prior_mu)^2 - 1]
        
        Args:
            mu: Mean of latent distribution, shape [batch_size, d_hidden]
            log_var: Log variance of latent distribution, shape [batch_size, d_hidden]
            
        Returns:
            KL divergence (scalar)
        """
        # Expand prior_means to match batch dimension [1, d_hidden] -> [batch_size, d_hidden]
        prior_means = self.prior_means.expand_as(mu)
        
        # Calculate KL divergence with non-zero mean prior
        # KL = 0.5 * (log(1/sigma^2) + sigma^2 + (mu-prior_mu)^2 - 1)
        kl = 0.5 * (
            -log_var + 
            log_var.exp() + 
            (mu - prior_means).pow(2) - 
            1
        )
        
        return kl.sum(-1).mean()
    
    @torch.no_grad()
    def detect_dead_neurons(self) -> torch.Tensor:
        """
        Detect neurons that haven't activated in the last window.
        
        Returns:
            Boolean mask of dead neurons
        """
        if not self.activation_history:
            return torch.zeros(self.config.d_hidden, dtype=torch.bool, device=self.config.device)
        
        # Stack activation history
        activation_window = torch.stack(self.activation_history, dim=0)
        
        # Neuron is dead if it hasn't activated at all in the window
        dead_neurons = (activation_window.sum(0) < self.config.dead_neuron_threshold)
        
        return dead_neurons
    
    @torch.no_grad()
    def resample_dead_neurons(self, activations: torch.Tensor) -> Tuple[List[int], int]:
        """
        Resample dead neurons using activations from the batch.
        
        Args:
            activations: Tensor of activations, shape [batch_size, d_input]
            
        Returns:
            List of resampled neuron indices and total count
        """
        # Detect dead neurons
        dead_mask = self.detect_dead_neurons()
        dead_indices = torch.where(dead_mask)[0].tolist()
        n_dead = len(dead_indices)
        
        if n_dead == 0:
            return [], 0
        
        # For resampling, we'll use the activations that had high reconstruction loss
        with torch.no_grad():
            # Forward pass to measure reconstruction loss per example
            outputs = self.forward(activations)
            recon_loss_per_example = ((outputs["x_recon"] - activations) ** 2).sum(-1)
            
            # Choose examples with highest reconstruction loss for resampling
            _, indices = torch.topk(recon_loss_per_example, k=min(n_dead, len(recon_loss_per_example)))
            selected_activations = activations[indices]
            
            # If we have fewer examples than dead neurons, repeat them
            if len(indices) < n_dead:
                repeats = (n_dead + len(indices) - 1) // len(indices)  # ceiling division
                selected_activations = selected_activations.repeat(repeats, 1)[:n_dead]
            
            # Get norms of alive neurons for scaling
            alive_mask = ~dead_mask
            if alive_mask.any():
                W_enc_norm_alive_mean = self.W_enc[:, alive_mask].norm(dim=0).mean().item()
            else:
                W_enc_norm_alive_mean = 1.0
                
            # Scale the activation vectors for resampling
            for i, neuron_idx in enumerate(dead_indices):
                # Get a replacement vector
                repl_vec = selected_activations[i % len(selected_activations)] - self.b_dec
                
                # Normalize and scale it
                repl_vec = repl_vec / repl_vec.norm().clamp(min=1e-6)
                repl_vec = repl_vec * W_enc_norm_alive_mean * self.config.resample_scale
                
                # Set the new encoder weights and reset the bias
                self.W_enc.data[:, neuron_idx] = repl_vec
                self.b_enc.data[neuron_idx] = 0.0
                
                # Update variance encoder weights if var_flag=1
                if self.config.var_flag == 1:
                    self.W_enc_var.data[:, neuron_idx] = torch.zeros_like(repl_vec)
                    self.b_enc_var.data[neuron_idx] = 0.0
        
        # Clear activation history after resampling
        self.activation_history = []
        
        return dead_indices, n_dead
    
    def encode(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Encode inputs to sparse features
        
        Args:
            x: Input tensor of shape [batch_size, d_input]
            deterministic: If True, return mean without sampling
            
        Returns:
            Sparse activations of shape [batch_size, d_hidden]
        """
        x_cent = x - self.b_dec
        mu = F.relu(x_cent @ self.W_enc + self.b_enc)
        
        if deterministic:
            return mu
        
        # Get log variance of latent distribution
        if self.config.var_flag == 1:
            log_var = F.relu(x_cent @ self.W_enc_var + self.b_enc_var)
        else:
            # Fixed variance when var_flag=0
            log_var = torch.zeros_like(mu)
        
        # Sample from the latent distribution
        return self.reparameterize(mu, log_var)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to inputs
        
        Args:
            z: Sparse activations of shape [batch_size, d_hidden]
            
        Returns:
            Reconstructed input of shape [batch_size, d_input]
        """
        return z @ self.W_dec + self.b_dec
    
    def train_step(self, 
                  x: torch.Tensor, 
                  optimizer: torch.optim.Optimizer, 
                  step: int) -> Dict[str, float]:
        """
        Perform a single training step
        
        Args:
            x: Input tensor of shape [batch_size, d_input]
            optimizer: PyTorch optimizer
            step: Current training step
            
        Returns:
            Dict of metrics
        """
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = self.forward(x)
        loss = outputs["loss"]
        
        # Backward pass
        loss.backward()
        
        # Remove parallel component of gradient for orthogonality
        with torch.no_grad():
            self.remove_parallel_component_of_grads()
        
        # Update parameters
        optimizer.step()
        
        # Normalize decoder after optimization step
        self.normalize_decoder()
        
        # Resample dead neurons periodically
        resampled_neurons = []
        n_dead = 0
        if step > 0 and step % self.config.resample_interval == 0:
            resampled_neurons, n_dead = self.resample_dead_neurons(x)
        
        # Convert torch tensors to Python types for logging
        metrics = {
            "loss": outputs["loss"].item(),
            "l2_loss": outputs["l2_loss"].item(),
            "kl_loss": outputs["kl_loss"].item(),
            "resampled_neurons": n_dead,
            "sparsity": outputs["frac_active"].mean().item(),
        }
        
        return metrics
    
    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        """
        Remove the parallel component of gradients for decoder weights
        This maintains the unit norm constraint during gradient descent
        """
        if self.W_dec.grad is None:
            return
            
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
