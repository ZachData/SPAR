import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder"""
    d_input: int             # Input dimension (e.g., d_mlp)
    d_hidden: int            # Hidden dimension (usually d_input * dict_mult)
    dict_mult: int = 8       # Dictionary multiplier (how many times larger is d_hidden than d_input)
    l1_coeff: float = 3e-4   # L1 coefficient for sparsity loss
    bias_decay: float = 0.95  # Decay rate for bias updates
    
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


class VanillaSAE(nn.Module):
    """
    Vanilla Sparse Autoencoder implementation
    Based on Anthropic's SAE architecture
    """
    
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        
        # Main parameters
        self.W_enc = nn.Parameter(torch.empty(config.d_input, config.d_hidden, dtype=config.dtype))
        self.W_dec = nn.Parameter(torch.empty(config.d_hidden, config.d_input, dtype=config.dtype))
        self.b_enc = nn.Parameter(torch.zeros(config.d_hidden, dtype=config.dtype))
        self.b_dec = nn.Parameter(torch.zeros(config.d_input, dtype=config.dtype))
        
        # Initialize parameters
        self._init_parameters()
        
        # Initialize tracking variables for dead neurons
        self.activation_history = []
        
        self.to(config.device)
    
    def _init_parameters(self):
        """Initialize parameters using Kaiming uniform"""
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        
        # Normalize decoder weights
        self.normalize_decoder()
    
    @torch.no_grad()
    def normalize_decoder(self):
        """Normalize decoder weights to have unit norm"""
        norm = self.W_dec.norm(dim=-1, keepdim=True)
        self.W_dec.data = self.W_dec.data / norm.clamp(min=1e-6)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the autoencoder
        
        Args:
            x: Input tensor of shape [batch_size, d_input]
            
        Returns:
            Dict containing loss components and intermediate activations
        """
        # Center the input using the decoder bias
        x_cent = x - self.b_dec
        
        # Encoder
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        
        # Decoder 
        x_recon = acts @ self.W_dec + self.b_dec
        
        # Compute losses
        l2_loss = ((x_recon - x) ** 2).sum(-1).mean()  # Reconstruction loss
        l1_loss = self.config.l1_coeff * acts.abs().sum()  # Sparsity loss
        loss = l2_loss + l1_loss
        
        # Record activations for tracking dead neurons
        frac_active = (acts.abs() > self.config.dead_neuron_threshold).float().mean(0)
        self.activation_history.append(frac_active.detach().cpu())
        
        # Only keep the most recent window of activations
        if len(self.activation_history) > self.config.dead_neuron_window:
            self.activation_history.pop(0)
        
        return {
            "loss": loss,
            "l2_loss": l2_loss,
            "l1_loss": l1_loss,
            "acts": acts,
            "x_recon": x_recon,
            "frac_active": frac_active
        }
    
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
            acts = F.relu(x_cent @ self.W_enc + self.b_enc)
            x_recon = acts @ self.W_dec + self.b_dec
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
        
        # Clear activation history after resampling
        self.activation_history = []
        
        return dead_indices, n_dead
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode inputs to sparse features
        
        Args:
            x: Input tensor of shape [batch_size, d_input]
            
        Returns:
            Sparse activations of shape [batch_size, d_hidden]
        """
        x_cent = x - self.b_dec
        return F.relu(x_cent @ self.W_enc + self.b_enc)
    
    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to inputs
        
        Args:
            acts: Sparse activations of shape [batch_size, d_hidden]
            
        Returns:
            Reconstructed input of shape [batch_size, d_input]
        """
        return acts @ self.W_dec + self.b_dec
    
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
            "l1_loss": outputs["l1_loss"].item(),
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
