import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, Any, List, Union, Callable
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class ToyModelConfig:
    """Configuration for the Toy Model"""
    n_features: int          # Number of features
    n_hidden: int            # Number of hidden dimensions (bottleneck)
    n_instances: int = 1     # Number of instances to train in parallel
    
    # Feature correlation parameters
    n_correlated_pairs: int = 0    # Number of correlated feature pairs
    n_anticorrelated_pairs: int = 0  # Number of anticorrelated feature pairs
    corr_rate: Optional[float] = None # Correlation rate for general correlation
    corr_matrix: Optional[torch.Tensor] = None  # Custom correlation matrix
    
    # Training parameters
    batch_size: int = 1024
    steps: int = 10_000
    lr: float = 1e-3
    optimizer: str = "adam"  # "adam" or "adamw"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ToyModel(nn.Module):
    """
    A toy model of bottleneck superposition from the
    "Toy Models of Superposition" paper by Anthropic
    """
    
    def __init__(
        self,
        config: ToyModelConfig,
        feature_probability: Optional[Union[float, torch.Tensor]] = None,
        importance: Optional[Union[float, torch.Tensor]] = None,
    ):
        super().__init__()
        self.config = config
        
        # Set up feature probability (sparsity)
        if feature_probability is None:
            feature_probability = torch.ones(())
        if isinstance(feature_probability, float):
            feature_probability = torch.tensor(feature_probability)
        self.feature_probability = feature_probability.to(config.device)
        
        # Set up feature importance
        if importance is None:
            importance = torch.ones(())
        if isinstance(importance, float):
            importance = torch.tensor(importance)
        self.importance = importance.to(config.device)
        
        # Initialize W with Xavier normal initialization
        # Shape: [n_instances, n_hidden, n_features]
        self.W = nn.Parameter(torch.empty((config.n_instances, config.n_hidden, config.n_features), 
                                         device=config.device))
        nn.init.xavier_normal_(self.W)
        
        # Initialize bias
        self.b_final = nn.Parameter(torch.zeros((config.n_instances, config.n_features), 
                                              device=config.device))
        
        # Prepare correlation matrix if needed
        if config.corr_matrix is not None:
            self.correlation_matrix = config.corr_matrix.to(config.device)
        elif config.corr_rate is not None:
            # Create correlation matrix with uniform correlation rate
            corr_matrix = torch.ones((config.n_features, config.n_features)) * config.corr_rate
            torch.diagonal(corr_matrix)[:] = 1.0
            self.correlation_matrix = corr_matrix.to(config.device)
        else:
            self.correlation_matrix = None
            
    def forward(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            features: Tensor of shape [..., n_instances, n_features]
            
        Returns:
            Reconstructed features of shape [..., n_instances, n_features]
        """
        # Compute hidden state
        # features: [..., instances, n_features]
        # W: [instances, n_hidden, n_features]
        hidden = torch.einsum("...if,ihf->...ih", features, self.W)
        
        # Reconstruct features
        out = torch.einsum("...ih,ihf->...if", hidden, self.W)
        out = out + self.b_final
        
        # Apply ReLU activation
        out = F.relu(out)
        
        return out
    
    def generate_batch(self, batch_size: int) -> torch.Tensor:
        """
        Generate a batch of synthetic data based on the model's characteristics
        
        Args:
            batch_size: Number of examples to generate
            
        Returns:
            Tensor of shape [batch_size, n_instances, n_features]
        """
        if self.config.n_correlated_pairs > 0 or self.config.n_anticorrelated_pairs > 0:
            return self.generate_structured_batch(batch_size)
        elif self.correlation_matrix is not None:
            return self.generate_correlated_batch(batch_size)
        else:
            return self.generate_simple_batch(batch_size)
    
    def generate_simple_batch(self, batch_size: int) -> torch.Tensor:
        """
        Generate a simple batch with independent features
        
        Args:
            batch_size: Number of examples to generate
            
        Returns:
            Tensor of shape [batch_size, n_instances, n_features]
        """
        # Generate random feature values
        feat = torch.rand((batch_size, self.config.n_instances, self.config.n_features), 
                        device=self.W.device)
        
        # Determine which features are active based on feature probability
        feat_seeds = torch.rand((batch_size, self.config.n_instances, self.config.n_features), 
                              device=self.W.device)
        feat_is_present = feat_seeds <= self.feature_probability
        
        # Zero out inactive features
        batch = torch.where(
            feat_is_present,
            feat,
            torch.zeros((), device=self.W.device),
        )
        
        return batch
    
    def generate_structured_batch(self, batch_size: int) -> torch.Tensor:
        """
        Generate a batch with correlated and/or anticorrelated feature pairs
        
        Args:
            batch_size: Number of examples to generate
            
        Returns:
            Tensor of shape [batch_size, n_instances, n_features]
        """
        data = []
        
        # Generate correlated pairs
        if self.config.n_correlated_pairs > 0:
            data.append(self.generate_correlated_features(batch_size, self.config.n_correlated_pairs))
        
        # Generate anticorrelated pairs
        if self.config.n_anticorrelated_pairs > 0:
            data.append(self.generate_anticorrelated_features(batch_size, self.config.n_anticorrelated_pairs))
        
        # Generate uncorrelated features
        n_uncorrelated = (self.config.n_features - 
                         2 * self.config.n_correlated_pairs - 
                         2 * self.config.n_anticorrelated_pairs)
        
        if n_uncorrelated > 0:
            data.append(self.generate_uncorrelated_features(batch_size, n_uncorrelated))
        
        # Concatenate all features
        batch = torch.cat(data, dim=-1)
        return batch
    
    def generate_correlated_features(self, batch_size: int, n_pairs: int) -> torch.Tensor:
        """
        Generate pairs of correlated features (appear together or not at all)
        
        Args:
            batch_size: Number of examples to generate
            n_pairs: Number of correlated pairs to generate
            
        Returns:
            Tensor of shape [batch_size, n_instances, 2*n_pairs]
        """
        # Generate potential feature values
        feat = torch.rand((batch_size, self.config.n_instances, 2 * n_pairs), 
                        device=self.W.device)
        
        # Seed values for each pair
        feat_set_seeds = torch.rand((batch_size, self.config.n_instances, n_pairs), 
                                  device=self.W.device)
        
        # Set is present if seed <= feature_probability
        feat_set_is_present = feat_set_seeds <= self.feature_probability[:, [0]]
        
        # Expand to apply to both features in the pair
        feat_is_present = torch.repeat_interleave(feat_set_is_present, 2, dim=-1)
        
        # Apply presence mask
        return torch.where(feat_is_present, feat, 0.0)
    
    def generate_anticorrelated_features(self, batch_size: int, n_pairs: int) -> torch.Tensor:
        """
        Generate pairs of anticorrelated features (only one can appear at a time)
        
        Args:
            batch_size: Number of examples to generate
            n_pairs: Number of anticorrelated pairs to generate
            
        Returns:
            Tensor of shape [batch_size, n_instances, 2*n_pairs]
        """
        # Generate potential feature values
        feat = torch.rand((batch_size, self.config.n_instances, 2 * n_pairs), 
                        device=self.W.device)
        
        # Seed for whether the pair is present at all
        feat_set_seeds = torch.rand((batch_size, self.config.n_instances, n_pairs), 
                                  device=self.W.device)
        
        # Seed for which feature in the pair is present (first or second)
        first_feat_seeds = torch.rand((batch_size, self.config.n_instances, n_pairs), 
                                    device=self.W.device)
        
        # Set is present if seed <= 2*feature_probability (doubled chance since only one will appear)
        feat_set_is_present = feat_set_seeds <= 2 * self.feature_probability[:, [0]]
        
        # For each pair, choose which feature is present with 50% probability
        first_feat_is_present = first_feat_seeds <= 0.5
        
        # First feature is present if the pair is present AND it was chosen
        first_feats = torch.where(
            feat_set_is_present & first_feat_is_present, 
            feat[:, :, :n_pairs], 
            0.0
        )
        
        # Second feature is present if the pair is present AND first wasn't chosen
        second_feats = torch.where(
            feat_set_is_present & (~first_feat_is_present), 
            feat[:, :, n_pairs:], 
            0.0
        )
        
        # Interleave the features
        result = torch.zeros((batch_size, self.config.n_instances, 2 * n_pairs), 
                           device=self.W.device)
        result[:, :, 0::2] = first_feats
        result[:, :, 1::2] = second_feats
        
        return result
    
    def generate_uncorrelated_features(self, batch_size: int, n_uncorrelated: int) -> torch.Tensor:
        """
        Generate uncorrelated features
        
        Args:
            batch_size: Number of examples to generate
            n_uncorrelated: Number of uncorrelated features to generate
            
        Returns:
            Tensor of shape [batch_size, n_instances, n_uncorrelated]
        """
        # Generate potential feature values
        feat = torch.rand((batch_size, self.config.n_instances, n_uncorrelated), 
                        device=self.W.device)
        
        # Determine which features are active
        feat_seeds = torch.rand((batch_size, self.config.n_instances, n_uncorrelated), 
                              device=self.W.device)
        feat_is_present = feat_seeds <= self.feature_probability[:, [0]]
        
        # Apply presence mask
        return torch.where(feat_is_present, feat, 0.0)
    
    def generate_correlated_batch(self, batch_size: int) -> torch.Tensor:
        """
        Generate batch with correlated features based on correlation matrix
        
        Args:
            batch_size: Number of examples to generate
            
        Returns:
            Tensor of shape [batch_size, n_instances, n_features]
        """
        # Generate uncorrelated feature seeds
        uncorr_feat = torch.rand((batch_size, self.config.n_instances, self.config.n_features), 
                               device=self.W.device)
        
        # Apply Cholesky decomposition to create correlation
        L = torch.linalg.cholesky(self.correlation_matrix)
        
        # Transform uncorrelated features into correlated ones
        corr_feat = torch.matmul(uncorr_feat, L)
        
        # Determine which features are present
        feat_seeds = torch.rand((batch_size, self.config.n_instances, self.config.n_features), 
                              device=self.W.device)
        feat_is_present = feat_seeds <= self.feature_probability
        
        # Apply presence mask
        batch = torch.where(
            feat_is_present,
            corr_feat,
            torch.zeros((), device=self.W.device),
        )
        
        return batch
    
    def calculate_loss(
        self,
        output: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate weighted reconstruction loss
        
        Args:
            output: Model output of shape [batch_size, n_instances, n_features]
            batch: Input batch of shape [batch_size, n_instances, n_features]
            
        Returns:
            Loss tensor
        """
        # Apply importance weighting to squared error
        error = self.importance * ((batch - output) ** 2)
        
        # Average over batch and sum over instances
        loss = torch.mean(error, dim=(0, 2)).sum()
        
        return loss
    
    def optimize(
        self,
        batch_size: Optional[int] = None,
        steps: Optional[int] = None,
        log_freq: int = 100,
        lr: Optional[float] = None,
        lr_scale: Callable[[int, int], float] = lambda step, steps: 1.0,
    ):
        """
        Optimize the model using the configured hyperparameters
        
        Args:
            batch_size: Batch size for training, uses config if None
            steps: Number of training steps, uses config if None
            log_freq: Frequency of logging
            lr: Learning rate, uses config if None
            lr_scale: Function to scale learning rate over training
        """
        # Use config values if not specified
        batch_size = batch_size or self.config.batch_size
        steps = steps or self.config.steps
        lr = lr or self.config.lr
        
        # Create optimizer
        if self.config.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(list(self.parameters()), lr=lr)
        else:
            optimizer = torch.optim.Adam(list(self.parameters()), lr=lr)
        
        # Training loop
        progress_bar = tqdm(range(steps))
        for step in progress_bar:
            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group['lr'] = step_lr
            
            # Generate batch and optimize
            optimizer.zero_grad()
            batch = self.generate_batch(batch_size)
            out = self(batch)
            loss = self.calculate_loss(out, batch)
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(
                    loss=loss.item() / self.config.n_instances, 
                    lr=step_lr
                )
                
        return loss.item()


def linear_lr(step, steps):
    """Linear learning rate decay"""
    return 1.0 - (step / steps)


def constant_lr(step, steps):
    """Constant learning rate"""
    return 1.0


def cosine_decay_lr(step, steps):
    """Cosine decay learning rate"""
    return np.cos(0.5 * np.pi * step / (steps - 1))
