import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass


@dataclass
class VSAEMultiConfig:
    """Configuration for VSAE with Multivariate Gaussian prior"""
    d_input: int             # Input dimension (e.g., d_mlp)
    d_hidden: int            # Hidden dimension (usually d_input * dict_mult)
    dict_mult: int = 8       # Dictionary multiplier (how many times larger is d_hidden than d_input)
    kl_coeff: float = 3e-4   # KL coefficient for variational loss
    bias_decay: float = 0.95 # Decay rate for bias updates
    var_flag: int = 0        # Flag to indicate if variance is learned (0: fixed, 1: learned)
    
    # Correlation parameters
    corr_rate: Optional[float] = 0.5  # Default correlation rate if no matrix provided
    corr_matrix: Optional[torch.Tensor] = None  # Custom correlation matrix
    
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


class VSAEMultiGaussian(nn.Module):
    """
    VSAE with Multivariate Gaussian prior implementation
    Designed to handle general correlation structures in the latent space
    """
    
    def __init__(self, config: VSAEMultiConfig):
        super().__init__()
        self.config = config
        
        # Main parameters
        self.W_enc = nn.Parameter(torch.empty(config.d_input, config.d_hidden, dtype=config.dtype))
        self.W_dec = nn.Parameter(torch.empty(config.d_hidden, config.d_input, dtype=config.dtype))
        self.b_enc = nn.Parameter(torch.zeros(config.d_hidden, dtype=config.dtype))
        self.b_dec = nn.Parameter(torch.zeros(config.d_input, dtype=config.dtype))
        
        # Variance parameters (only used if var_flag=1)
        if self.config.var_flag == 1:
            self.W_enc_var = nn.Parameter(torch.empty(config.d_input, config.d_hidden, dtype=config.dtype))
            self.b_enc_var = nn.Parameter(torch.zeros(config.d_hidden, dtype=config.dtype))
        
        # Initialize parameters
        self._init_parameters()
        
        # Initialize tracking variables for dead neurons
        self.activation_history = []
        
        # Setup correlation matrix
        self.prior_covariance = self.construct_prior_covariance(config)
        self.prior_precision = self.compute_prior_precision()
        self.prior_cov_logdet = self.compute_prior_logdet()
        
        self.to(config.device)
    
    def _init_parameters(self):
        """Initialize parameters using Kaiming uniform"""
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        
        if self.config.var_flag == 1:
            nn.init.kaiming_uniform_(self.W_enc_var)
        
        # Normalize decoder weights
        self.normalize_decoder()
    
    def construct_prior_covariance(self, config):
        """
        Construct the prior covariance matrix based on config
        
        Returns:
            torch.Tensor: Covariance matrix of shape [d_hidden, d_hidden]
        """
        # Always use d_hidden for the correlation matrix dimensions
        d_hidden = config.d_hidden
        
        if config.corr_matrix is not None and config.corr_matrix.shape[0] == d_hidden:
            corr_matrix = config.corr_matrix
        elif config.corr_rate is not None:
            # Create a matrix with uniform correlation matching the VSAE's hidden dimension
            corr_matrix = torch.full((d_hidden, d_hidden), config.corr_rate, 
                                dtype=config.dtype, device=config.device)
            torch.diagonal(corr_matrix)[:] = 1.0
        else:
            # Default to identity (no correlation)
            return torch.eye(d_hidden, dtype=config.dtype, device=config.device)
        
        # Ensure the correlation matrix is valid
        if not torch.allclose(corr_matrix, corr_matrix.t()):
            raise ValueError("Correlation matrix must be symmetric")
        
        # Compute eigenvalues to check positive definiteness
        try:
            eigenvalues = torch.linalg.eigvalsh(corr_matrix)
            if not (eigenvalues > 0).all():
                raise ValueError("Correlation matrix must be positive definite")
        except:
            # If decomposition fails, adjust matrix to ensure positive definiteness
            print("Warning: Adjusting correlation matrix to ensure positive definiteness")
            corr_matrix = corr_matrix + torch.eye(corr_matrix.shape[0]) * 1e-4
        
        return corr_matrix.to(config.device)
    
    def compute_prior_precision(self):
        """
        Compute the precision matrix (inverse of covariance)
        
        Returns:
            torch.Tensor: Precision matrix of shape [d_hidden, d_hidden]
        """
        try:
            return torch.linalg.inv(self.prior_covariance)
        except:
            # Add small jitter to ensure invertibility
            jitter = torch.eye(self.config.d_hidden, device=self.config.device) * 1e-4
            return torch.linalg.inv(self.prior_covariance + jitter)
    
    def compute_prior_logdet(self):
        """
        Compute log determinant of prior covariance
        
        Returns:
            torch.Tensor: Log determinant (scalar)
        """
        return torch.logdet(self.prior_covariance)
    
    @torch.no_grad()
    def normalize_decoder(self):
        """Normalize decoder weights to have unit norm"""
        norm = self.W_dec.norm(dim=-1, keepdim=True)
        self.W_dec.data = self.W_dec.data / norm.clamp(min=1e-6)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VSAE
        
        Args:
            x: Input tensor of shape [batch_size, d_input]
            
        Returns:
            Dict containing loss components and intermediate activations
        """
        batch_size = x.shape[0]
        
        # Center the input using the decoder bias
        x_cent = x - self.b_dec
        
        # Encoder: compute mean
        mu = F.relu(x_cent @ self.W_enc + self.b_enc)
        
        # Encoder: compute log variance
        if self.config.var_flag == 1:
            log_var = F.relu(x_cent @ self.W_enc_var + self.b_enc_var)
        else:
            # Fixed variance (log(1) = 0)
            log_var = torch.zeros_like(mu)
        
        # Sample from the latent distribution using reparameterization trick
        z = self.reparameterize(mu, log_var)
        
        # Decoder
        x_recon = z @ self.W_dec + self.b_dec
        
        # Compute losses
        l2_loss = ((x_recon - x) ** 2).sum(-1).mean()  # Reconstruction loss
        kl_loss = self.compute_kl_divergence(mu, log_var)  # KL divergence
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
    
    def compute_kl_divergence(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between the approximate posterior q(z|x) and the prior p(z)
        For multivariate Gaussian, the KL divergence is:
        
        KL(q||p) = 0.5 * [tr(Σp^-1 * Σq) + (μq - μp)^T * Σp^-1 * (μq - μp) - k + ln(|Σp|/|Σq|)]
        
        where:
        - Σp is the prior covariance
        - Σq is the approximate posterior covariance (diagonal in our case)
        - μp is the prior mean (0 in our case)
        - μq is the approximate posterior mean
        - k is the dimension of the latent space
        
        Args:
            mu: Mean of approximate posterior q(z|x), shape [batch_size, d_hidden]
            log_var: Log variance of approximate posterior q(z|x), shape [batch_size, d_hidden]
            
        Returns:
            KL divergence (scalar)
        """
        batch_size = mu.shape[0]
        
        # For computational efficiency, we'll average across batch first
        mu_avg = mu.mean(0)  # [d_hidden]
        var_avg = log_var.exp().mean(0)  # [d_hidden]
        
        # Compute trace term: tr(Σp^-1 * Σq)
        # Since Σq is diagonal, this is simple
        trace_term = (self.prior_precision.diagonal() * var_avg).sum()
        
        # Quadratic term: (μq - μp)^T * Σp^-1 * (μq - μp), where μp = 0
        # This is equivalent to mu_avg^T * Σp^-1 * mu_avg
        quad_term = mu_avg @ self.prior_precision @ mu_avg
        
        # Log determinant term: ln(|Σp|/|Σq|)
        # |Σq| is product of diagonal elements (variances)
        log_det_q = log_var.sum(1).mean()  # Average across batch
        log_det_term = self.prior_cov_logdet - log_det_q
        
        # Combine terms
        kl = 0.5 * (trace_term + quad_term - self.config.d_hidden + log_det_term)
        
        # Ensure KL is non-negative (can be negative due to numerical issues)
        kl = torch.clamp(kl, min=0.0)
        
        return kl
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to enable backpropagation through sampling
        
        Args:
            mu: Mean tensor of shape [batch_size, d_hidden]
            log_var: Log variance tensor of shape [batch_size, d_hidden]
            
        Returns:
            Sampled latent vector of shape [batch_size, d_hidden]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
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
            x_cent = activations - self.b_dec
            mu = F.relu(x_cent @ self.W_enc + self.b_enc)
            
            # Use deterministic encoding for resampling
            x_recon = mu @ self.W_dec + self.b_dec
            recon_loss_per_example = ((x_recon - activations) ** 2).sum(-1)
            
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
                
                # Also reset variance encoder if used
                if self.config.var_flag == 1:
                    self.W_enc_var.data[:, neuron_idx] = 0.0
                    self.b_enc_var.data[neuron_idx] = 0.0
        
        # Clear activation history after resampling
        self.activation_history = []
        
        return dead_indices, n_dead
    
    def encode(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Encode inputs to latent representation
        
        Args:
            x: Input tensor of shape [batch_size, d_input]
            deterministic: If True, return mean without sampling
            
        Returns:
            Latent representation of shape [batch_size, d_hidden]
        """
        x_cent = x - self.b_dec
        mu = F.relu(x_cent @ self.W_enc + self.b_enc)
        
        if deterministic:
            return mu
        
        # Compute log variance
        if self.config.var_flag == 1:
            log_var = F.relu(x_cent @ self.W_enc_var + self.b_enc_var)
        else:
            log_var = torch.zeros_like(mu)
        
        # Sample using reparameterization trick
        return self.reparameterize(mu, log_var)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to inputs
        
        Args:
            z: Latent representation of shape [batch_size, d_hidden]
            
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
