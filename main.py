#!/usr/bin/env python3
import argparse
import torch
import numpy as np
import os
import time
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import json
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from toy_model import ToyModel, ToyModelConfig, linear_lr, constant_lr, cosine_decay_lr
from gelu_1l_model import GELU1LModel, GELU1LConfig
from gpt2_model import GPT2Model, GPT2Config

from vanilla_sae import VanillaSAE, SAEConfig
from vsae_iso import VSAEIsoGaussian, VSAEIsoConfig
from vsae_mix import VSAEMixGaussian, VSAEMixConfig
from vsae_multi import VSAEMultiGaussian, VSAEMultiConfig

class ModelTrainer:
    """
    Generic trainer for sparse autoencoders on various source models.
    
    This class handles the training loop for SAEs and VSAEs on different
    source models like the toy model, GELU-1L, or GPT-2.
    """
    
    def __init__(
        self, 
        source_model: Union[ToyModel, GELU1LModel, GPT2Model],
        autoencoder: Union[VanillaSAE, VSAEIsoGaussian, VSAEMixGaussian, VSAEMultiGaussian],
        lr: float = 1e-4,
        steps: int = 10000,
        batch_size: Optional[int] = None,
        lr_scale: Callable[[int, int], float] = lambda step, steps: 1.0,
        log_freq: int = 100,
        use_tensorboard: bool = True,
        tensorboard_dir: str = "./runs",
        experiment_name: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the model trainer.
        
        Args:
            source_model: Source model to get activations from
            autoencoder: Autoencoder model to train
            lr: Learning rate
            steps: Number of training steps
            batch_size: Batch size for training
            lr_scale: Learning rate scaling function
            log_freq: Logging frequency
            use_tensorboard: Whether to use TensorBoard for logging
            tensorboard_dir: Directory to store TensorBoard logs
            experiment_name: Name of the experiment for TensorBoard
            device: Device to use
        """
        self.source_model = source_model
        self.autoencoder = autoencoder
        self.lr = lr
        self.steps = steps
        self.batch_size = batch_size
        self.lr_scale = lr_scale
        self.log_freq = log_freq
        self.device = device
        self.use_tensorboard = use_tensorboard
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        
        # Set batch size if not provided
        if self.batch_size is None:
            if hasattr(source_model, 'config') and hasattr(source_model.config, 'batch_size'):
                self.batch_size = source_model.config.batch_size
            else:
                self.batch_size = 512  # Default value
                
        # Setup TensorBoard
        if use_tensorboard:
            if experiment_name is None:
                # Create experiment name based on models used
                source_name = self._get_model_name(source_model)
                ae_name = self._get_model_name(autoencoder)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                experiment_name = f"{source_name}_{ae_name}_{timestamp}"
            
            self.tensorboard_dir = Path(tensorboard_dir) / experiment_name
            self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
            print(f"TensorBoard logs will be saved to {self.tensorboard_dir}")
    
    def _get_model_name(self, model):
        """Get a simple name for the model type"""
        class_name = model.__class__.__name__
        
        if "Toy" in class_name:
            return "toy"
        elif "GELU" in class_name:
            return "gelu1l"
        elif "GPT2" in class_name:
            return "gpt2"
        elif "VanillaSAE" in class_name:
            return "sae"
        elif "VSAEIso" in class_name:
            return "vsae_iso"
        elif "VSAEMix" in class_name:
            return "vsae_mix" 
        elif "VSAEMulti" in class_name:
            return "vsae_multi"
        else:
            return class_name.lower()
    
    def _log_to_tensorboard(self, metrics: Dict[str, float], step: int):
        """
        Log metrics to TensorBoard
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
        """
        if not self.use_tensorboard:
            return
            
        # Log all metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"train/{key}", value, step)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar("train/learning_rate", current_lr, step)
        
        # Every 1000 steps, log histograms of weights and activations
        if step % 1000 == 0 or step == self.steps - 1:
            # Log weight distributions
            self.writer.add_histogram("weights/encoder", self.autoencoder.W_enc, step)
            self.writer.add_histogram("weights/decoder", self.autoencoder.W_dec, step)
            self.writer.add_histogram("biases/encoder", self.autoencoder.b_enc, step)
            self.writer.add_histogram("biases/decoder", self.autoencoder.b_dec, step)
            
            # Log feature sparsity if available
            if hasattr(self.autoencoder, 'activation_history') and self.autoencoder.activation_history:
                latest_activations = self.autoencoder.activation_history[-1]
                self.writer.add_histogram("activations/feature_sparsity", latest_activations, step)
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the autoencoder on activations from the source model.
        
        Returns:
            Dict of training metrics
        """
        # Initialize metrics tracking
        metrics_history = {
            "loss": [],
            "l2_loss": [],
            "sparsity": []
        }
        
        # Add specific metrics based on autoencoder type
        if isinstance(self.autoencoder, VanillaSAE):
            metrics_history["l1_loss"] = []
        elif isinstance(self.autoencoder, (VSAEIsoGaussian, VSAEMixGaussian, VSAEMultiGaussian)):
            metrics_history["kl_loss"] = []
        
        # Training loop
        progress_bar = tqdm(range(self.steps))
        for step in progress_bar:
            # Update learning rate
            step_lr = self.lr * self.lr_scale(step, self.steps)
            for group in self.optimizer.param_groups:
                group['lr'] = step_lr
            
            # Get batch of activations from source model
            if isinstance(self.source_model, ToyModel):
                # For toy model: generate batch and compute hidden state
                batch = self.source_model.generate_batch(self.batch_size)
                with torch.no_grad():
                    # Focus on first instance for simplicity
                    hidden = torch.einsum("bif,ihf->bih", batch, self.source_model.W)[:, 0, :]
            elif isinstance(self.source_model, GELU1LModel):
                # For GELU-1L model: get batch of MLP activations
                hidden = self.source_model.get_batch_activations(self.batch_size)
            elif isinstance(self.source_model, GPT2Model):
                # For GPT-2 model: get batch of MLP activations
                hidden = self.source_model.get_batch_activations(self.batch_size)
            else:
                raise ValueError(f"Unsupported source model type: {type(self.source_model)}")
            
            # Train autoencoder for one step
            metrics = self.autoencoder.train_step(hidden, self.optimizer, step)
            
            # Record metrics
            for key in metrics_history:
                if key in metrics:
                    metrics_history[key].append(metrics[key])
            
            # Log to TensorBoard
            if self.use_tensorboard:
                self._log_to_tensorboard(metrics, step)
            
            # Update progress bar
            if step % self.log_freq == 0 or step == self.steps - 1:
                log_info = ", ".join([f"{k}: {metrics[k]:.6f}" for k in metrics_history if k in metrics])
                progress_bar.set_postfix_str(log_info)
        
        # Close TensorBoard writer
        if self.use_tensorboard:
            self.writer.close()
        
        return metrics_history

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train and evaluate sparse autoencoders")
    
    # Model type arguments
    model_group = parser.add_argument_group("Model Type")
    model_group.add_argument("--vanilla_sae", action="store_true", help="Use vanilla sparse autoencoder")
    model_group.add_argument("--vsae_iso", action="store_true", help="Use VSAE with isotropic Gaussian prior")
    model_group.add_argument("--vsae_mix", action="store_true", help="Use VSAE with Gaussian mixture prior")
    model_group.add_argument("--vsae_multi", action="store_true", help="Use VSAE with multivariate Gaussian prior")
    
    # Target model arguments
    target_group = parser.add_argument_group("Target Model")
    target_group.add_argument("--toy_model", action="store_true", help="Use toy model for training")
    target_group.add_argument("--l1_gelu", action="store_true", help="Use 1-layer GELU model for training")
    target_group.add_argument("--gpt2_small", action="store_true", help="Use GPT-2 small for training")
    
    # Toy model configuration
    toy_group = parser.add_argument_group("Toy Model Configuration")
    toy_group.add_argument("--n_features", type=int, default=100, help="Number of features in toy model")
    toy_group.add_argument("--n_hidden", type=int, default=20, help="Hidden dimension in toy model")
    toy_group.add_argument("--n_instances", type=int, default=10, help="Number of parallel instances in toy model")
    toy_group.add_argument("--n_correlated_pairs", type=int, default=0, help="Number of correlated feature pairs")
    toy_group.add_argument("--n_anticorrelated_pairs", type=int, default=0, help="Number of anticorrelated feature pairs")
    toy_group.add_argument("--corr_rate", type=float, default=None, help="Correlation rate for general correlation")
    
    # SAE configuration
    sae_group = parser.add_argument_group("SAE Configuration")
    sae_group.add_argument("--sae_dim", type=int, default=None, help="Dictionary dimension for SAE")
    sae_group.add_argument("--dict_mult", type=int, default=8, help="Dictionary multiplier for SAE")
    sae_group.add_argument("--l1_coeff", type=float, default=3e-4, help="L1 coefficient for sparsity loss")
    
    # VSAE configuration
    vsae_group = parser.add_argument_group("VSAE Configuration")
    vsae_group.add_argument("--kl_coeff", type=float, default=3e-4, help="KL coefficient for VAE loss")
    vsae_group.add_argument("--var_flag", type=int, default=0, choices=[0, 1], 
                        help="Whether to learn variance (0: fixed, 1: learned)")
    
    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--batch_size", type=int, default=4096, help="Batch size for training")
    train_group.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    train_group.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_group.add_argument("--lr_decay", choices=["constant", "linear", "cosine"], default="constant", 
                            help="Learning rate decay schedule")
    
    # Visualization arguments
    viz_group = parser.add_argument_group("Visualization")
    viz_group.add_argument("--use_tensorboard", action="store_true", help="Enable TensorBoard logging")
    viz_group.add_argument("--tensorboard_dir", type=str, default="./runs", help="Directory for TensorBoard logs")
    viz_group.add_argument("--experiment_name", type=str, default=None, help="Experiment name for TensorBoard")
    
    # Utility arguments
    util_group = parser.add_argument_group("Utility")
    util_group.add_argument("--run_tests", action="store_true", help="Run tests before training")
    util_group.add_argument("--save_model", action="store_true", help="Save model after training")
    util_group.add_argument("--load_model", type=str, default=None, help="Load model from path")
    util_group.add_argument("--output_dir", type=str, default="./results", help="Directory for saving results")
    util_group.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_lr_scheduler(schedule: str):
    """Get learning rate scheduler function"""
    if schedule == "linear":
        return linear_lr
    elif schedule == "cosine":
        return cosine_decay_lr
    else:
        return constant_lr

def run_tests():
    """Run unit tests to verify components"""
    import unittest
    from tests import TestToyModel, TestVanillaSAE, TestIntegration
    
    print("Running tests...")
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestToyModel))
    test_suite.addTest(unittest.makeSuite(TestVanillaSAE))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    runner = unittest.TextTestRunner()
    runner.run(test_suite)

def setup_gpt2_model(args):
    """
    Set up a GPT-2 model based on the provided arguments
    
    Args:
        args: Command line arguments
        
    Returns:
        Configured GPT-2 model
    """
    print("Setting up GPT-2 model...")
    from gpt2_model import GPT2Model, GPT2Config
    
    # Override batch size if specified for GPT-2 specifically
    batch_size = args.gpt2_batch_size if hasattr(args, 'gpt2_batch_size') else args.batch_size
    
    # Create GPT-2 config
    gpt2_config = GPT2Config(
        model_name="gpt2",
        activation_name=args.gpt2_act_name,
        layer_num=args.gpt2_layer,
        batch_size=batch_size,
        context_length=args.gpt2_context_length,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Create GPT-2 model
    gpt2_model = GPT2Model(gpt2_config)
    
    print(f"GPT-2 model initialized: Layer {args.gpt2_layer}, Activation {args.gpt2_act_name}")
    print(f"MLP dimension: {gpt2_model.d_mlp}")
    
    return gpt2_model
    
def setup_gelu_1l_model(args):
    """Create a 1-layer GELU transformer model instance"""
    print("Setting up 1-layer GELU model...")
    from gelu_1l_model import GELU1LModel, GELU1LConfig
    
    config = GELU1LConfig(
        model_name="gelu-1l",
        activation_name="post",  # MLP output activations
        layer_num=0,             # First and only layer
        batch_size=args.batch_size,
        context_length=128,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Create the model
    model = GELU1LModel(config)
    
    print(f"Model loaded: {config.model_name}")
    print(f"MLP dimension: {model.d_mlp}")
    
    return model

def setup_toy_model(args):
    """Create and train a toy model instance"""
    print("Setting up toy model...")
    
    config = ToyModelConfig(
        n_features=args.n_features,
        n_hidden=args.n_hidden,
        n_instances=args.n_instances,
        n_correlated_pairs=args.n_correlated_pairs,
        n_anticorrelated_pairs=args.n_anticorrelated_pairs, 
        corr_rate=args.corr_rate,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
    )
    
    # Create feature importance (exponential decay)
    importance = (100 ** -torch.linspace(0, 1, config.n_features))[None, :]
    
    # Create feature probability (sparsity varies across instances)
    feature_probability = (20 ** -torch.linspace(0, 1, config.n_instances))[:, None]
    
    model = ToyModel(
        config=config,
        importance=importance,
        feature_probability=feature_probability,
    )
    
    # Set up TensorBoard for toy model if enabled
    if args.use_tensorboard:
        writer = SummaryWriter(log_dir=f"{args.tensorboard_dir}/toy_model_{time.strftime('%Y%m%d-%H%M%S')}")
        print(f"TensorBoard logs for toy model will be saved to {writer.log_dir}")
    else:
        writer = None
    
    # Train the model
    print("Training toy model...")
    lr_scheduler = get_lr_scheduler(args.lr_decay)
    
    # Create optimizer here instead of accessing model.optimizer
    if config.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(list(model.parameters()), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
    
    # Track loss during training for TensorBoard
    progress_bar = tqdm(range(args.steps))
    for step in progress_bar:
        # Update learning rate
        step_lr = args.lr * lr_scheduler(step, args.steps)
        for group in optimizer.param_groups:  # Using the local optimizer
            group['lr'] = step_lr
        
        # Generate batch and optimize
        optimizer.zero_grad()
        batch = model.generate_batch(args.batch_size)
        out = model(batch)
        loss = model.calculate_loss(out, batch)
        loss.backward()
        optimizer.step()
        
        # Log to TensorBoard if enabled
        if writer is not None and (step % 100 == 0 or step == args.steps - 1):
            writer.add_scalar("train/loss", loss.item() / model.config.n_instances, step)
            writer.add_scalar("train/learning_rate", step_lr, step)
            
            # Visualize feature weights
            if step % 1000 == 0 or step == args.steps - 1:
                for i in range(min(5, model.config.n_instances)):  # Log first 5 instances
                    # Reshape weights for visualization as images
                    features = model.W[i].detach().cpu().numpy()
                    writer.add_image(f"weights/instance_{i}", 
                                   features.reshape(1, model.config.n_hidden, model.config.n_features), 
                                   step, dataformats='CHW')
        
        # Update progress bar
        if step % 100 == 0 or (step + 1 == args.steps):
            progress_bar.set_postfix(
                loss=loss.item() / model.config.n_instances, 
                lr=step_lr
            )
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    return model
    
    # Create feature importance (exponential decay)
    importance = (100 ** -torch.linspace(0, 1, config.n_features))[None, :]
    
    # Create feature probability (sparsity varies across instances)
    feature_probability = (20 ** -torch.linspace(0, 1, config.n_instances))[:, None]
    
    model = ToyModel(
        config=config,
        importance=importance,
        feature_probability=feature_probability,
    )
    
    # Train the model
    print("Training toy model...")
    lr_scheduler = get_lr_scheduler(args.lr_decay)
    model.optimize(
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        lr_scale=lr_scheduler,
    )
    
    return model

def train_vsae_multi(args, target_model=None):
    """Train a VSAE with multivariate Gaussian prior"""
    print("Training VSAE with multivariate Gaussian prior...")
    
    # Determine input dimension (from the TARGET MODEL'S HIDDEN DIMENSION)
    if args.toy_model and target_model is not None:
        d_input = target_model.config.n_hidden  # This is 2 in our example
        corr_rate = args.corr_rate  # Use the command line argument
        corr_matrix = None  # Don't use the toy model's correlation matrix
    elif args.gpt2_small:
        d_input = 768  # GPT-2 small hidden dimension
        corr_rate = args.corr_rate
        corr_matrix = None
    elif args.l1_gelu and target_model is not None:
        d_input = target_model.d_mlp  # 1L-GELU MLP dimension
        corr_rate = args.corr_rate
        corr_matrix = None
    else:
        raise ValueError("No target model specified")
    
    # Determine hidden dimension - this is the VSAE's latent dimension
    if args.sae_dim is not None:
        d_hidden = args.sae_dim
    else:
        d_hidden = d_input * args.dict_mult  # 2 * 4 = 8 in our example
    
    # Create VSAE config
    config = VSAEMultiConfig(
        d_input=d_input,
        d_hidden=d_hidden,
        dict_mult=args.dict_mult,
        kl_coeff=args.kl_coeff,
        var_flag=args.var_flag,
        corr_rate=corr_rate,
        corr_matrix=None  # We'll construct this properly below
    )
    
    # Create VSAE model
    vsae = VSAEMultiGaussian(config)
    
    # Train the model if we have a target
    metrics_history = None
    if (args.toy_model or args.l1_gelu) and target_model is not None:
        # Use the model trainer for training
        lr_scheduler = get_lr_scheduler(args.lr_decay)
        
        trainer = ModelTrainer(
            source_model=target_model,
            autoencoder=vsae,
            lr=args.lr,
            steps=args.steps,
            batch_size=min(args.batch_size, 16384),  # Limit batch size to prevent OOM
            lr_scale=lr_scheduler,
            log_freq=100,
            use_tensorboard=args.use_tensorboard,
            tensorboard_dir=args.tensorboard_dir,
            experiment_name=args.experiment_name
        )
        
        print(f"Training VSAE Multi on {'toy model' if args.toy_model else 'GELU-1L'} activations for {args.steps} steps...")
        
        # Run training
        metrics_history = trainer.train()
    
    # Save model if requested
    if args.save_model:
        save_dir = Path(args.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_type = "toy" if args.toy_model else "gelu1l" if args.l1_gelu else "gpt2"
        save_path = save_dir / f"vsae_multi_{model_type}_{timestamp}.pt"
        
        save_dict = {
            "state_dict": vsae.state_dict(),
            "config": vars(vsae.config),
            "args": vars(args),
        }
        
        if metrics_history:
            save_dict["final_metrics"] = {k: values[-1] if values else None for k, values in metrics_history.items()}
        
        torch.save(save_dict, save_path)
        
        print(f"Model saved to {save_path}")
    
    return vsae

def train_vsae_mix(args, target_model=None):
    """Train a VSAE with Gaussian mixture prior"""
    print("Training VSAE with Gaussian mixture prior...")
    
    # Determine input dimension
    if args.toy_model and target_model is not None:
        d_input = target_model.config.n_hidden
        n_correlated_pairs = target_model.config.n_correlated_pairs
        n_anticorrelated_pairs = target_model.config.n_anticorrelated_pairs
    elif args.gpt2_small:
        d_input = 768  # GPT-2 small hidden dimension
        n_correlated_pairs = args.n_correlated_pairs
        n_anticorrelated_pairs = args.n_anticorrelated_pairs
    elif args.l1_gelu and target_model is not None:
        d_input = target_model.d_mlp  # 1L-GELU MLP dimension
        n_correlated_pairs = args.n_correlated_pairs
        n_anticorrelated_pairs = args.n_anticorrelated_pairs
    else:
        raise ValueError("No target model specified")
    
    # Determine hidden dimension
    if args.sae_dim is not None:
        d_hidden = args.sae_dim
    else:
        d_hidden = d_input * args.dict_mult
    
    # Create VSAE config
    config = VSAEMixConfig(
        d_input=d_input,
        d_hidden=d_hidden,
        dict_mult=args.dict_mult,
        kl_coeff=args.kl_coeff,
        var_flag=args.var_flag,
        n_correlated_pairs=n_correlated_pairs,
        n_anticorrelated_pairs=n_anticorrelated_pairs
    )
    
    # Create VSAE model
    vsae = VSAEMixGaussian(config)
    
    # Train the model if we have a target
    metrics_history = None
    if (args.toy_model or args.l1_gelu) and target_model is not None:
        # Use the model trainer for training
        lr_scheduler = get_lr_scheduler(args.lr_decay)
        
        trainer = ModelTrainer(
            source_model=target_model,
            autoencoder=vsae,
            lr=args.lr,
            steps=args.steps,
            batch_size=min(args.batch_size, 16384),  # Limit batch size to prevent OOM
            lr_scale=lr_scheduler,
            log_freq=100,
            use_tensorboard=args.use_tensorboard,
            tensorboard_dir=args.tensorboard_dir,
            experiment_name=args.experiment_name
        )
        
        print(f"Training VSAE Mix on {'toy model' if args.toy_model else 'GELU-1L'} activations for {args.steps} steps...")
        
        # Run training
        metrics_history = trainer.train()
    
    # Save model if requested
    if args.save_model:
        save_dir = Path(args.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_type = "toy" if args.toy_model else "gelu1l" if args.l1_gelu else "gpt2"
        save_path = save_dir / f"vsae_mix_{model_type}_{timestamp}.pt"
        
        save_dict = {
            "state_dict": vsae.state_dict(),
            "config": vars(vsae.config),
            "args": vars(args),
        }
        
        if metrics_history:
            save_dict["final_metrics"] = {k: values[-1] if values else None for k, values in metrics_history.items()}
        
        torch.save(save_dict, save_path)
        
        print(f"Model saved to {save_path}")
    
    return vsae

def train_vsae_iso(args, target_model=None):
    """Train a VSAE with isotropic Gaussian prior"""
    print("Training VSAE with isotropic Gaussian prior...")
    
    # Determine input dimension
    if args.toy_model and target_model is not None:
        d_input = target_model.config.n_hidden
    elif args.gpt2_small:
        d_input = 768  # GPT-2 small hidden dimension
    elif args.l1_gelu and target_model is not None:
        d_input = target_model.d_mlp  # 1L-GELU MLP dimension
    else:
        raise ValueError("No target model specified")
    
    # Determine hidden dimension
    if args.sae_dim is not None:
        d_hidden = args.sae_dim
    else:
        d_hidden = d_input * args.dict_mult
    
    # Create VSAE config
    config = VSAEIsoConfig(
        d_input=d_input,
        d_hidden=d_hidden,
        dict_mult=args.dict_mult,
        kl_coeff=args.kl_coeff,
        var_flag=args.var_flag,
    )
    
    # Create VSAE model
    vsae = VSAEIsoGaussian(config)
    
    # Train the model if we have a target
    metrics_history = None
    if (args.toy_model or args.l1_gelu) and target_model is not None:
        # Use the model trainer for training
        lr_scheduler = get_lr_scheduler(args.lr_decay)
        
        trainer = ModelTrainer(
            source_model=target_model,
            autoencoder=vsae,
            lr=args.lr,
            steps=args.steps,
            batch_size=min(args.batch_size, 16384),  # Limit batch size to prevent OOM
            lr_scale=lr_scheduler,
            log_freq=100,
            use_tensorboard=args.use_tensorboard,
            tensorboard_dir=args.tensorboard_dir,
            experiment_name=args.experiment_name
        )
        
        print(f"Training VSAE Iso on {'toy model' if args.toy_model else 'GELU-1L'} activations for {args.steps} steps...")
        
        # Run training
        metrics_history = trainer.train()
    
    # Save model if requested
    if args.save_model:
        save_dir = Path(args.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_type = "toy" if args.toy_model else "gelu1l" if args.l1_gelu else "gpt2"
        save_path = save_dir / f"vsae_iso_{model_type}_{timestamp}.pt"
        
        save_dict = {
            "state_dict": vsae.state_dict(),
            "config": vars(vsae.config),
            "args": vars(args),
        }
        
        if metrics_history:
            save_dict["final_metrics"] = {k: values[-1] if values else None for k, values in metrics_history.items()}
            
        torch.save(save_dict, save_path)
        
        print(f"Model saved to {save_path}")
    
    return vsae

def train_vanilla_sae(args, target_model=None):
    """Train a vanilla sparse autoencoder"""
    print("Training vanilla SAE...")
    
    # Determine input dimension
    if args.toy_model and target_model is not None:
        d_input = target_model.config.n_hidden
    elif args.gpt2_small:
        d_input = 768  # GPT-2 small hidden dimension
    elif args.l1_gelu and target_model is not None:
        d_input = target_model.d_mlp  # 1L-GELU MLP dimension
    else:
        raise ValueError("No target model specified")
    
    # Determine hidden dimension
    if args.sae_dim is not None:
        d_hidden = args.sae_dim
    else:
        d_hidden = d_input * args.dict_mult
    
    # Create SAE config
    config = SAEConfig(
        d_input=d_input,
        d_hidden=d_hidden,
        dict_mult=args.dict_mult,
        l1_coeff=args.l1_coeff,
    )
    
    # Create SAE model
    sae = VanillaSAE(config)
    
    # Train the model if we have a target
    metrics_history = None
    if (args.toy_model or args.l1_gelu) and target_model is not None:
        # Use the model trainer for training
        lr_scheduler = get_lr_scheduler(args.lr_decay)
        
        trainer = ModelTrainer(
            source_model=target_model,
            autoencoder=sae,
            lr=args.lr,
            steps=args.steps,
            batch_size=min(args.batch_size, 16384),  # Limit batch size to prevent OOM
            lr_scale=lr_scheduler,
            log_freq=100,
            use_tensorboard=args.use_tensorboard,
            tensorboard_dir=args.tensorboard_dir,
            experiment_name=args.experiment_name
        )
        
        print(f"Training SAE on {'toy model' if args.toy_model else 'GELU-1L'} activations for {args.steps} steps...")
        
        # Run training
        metrics_history = trainer.train()
    
    # Save model if requested
    if args.save_model:
        save_dir = Path(args.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_type = "toy" if args.toy_model else "gelu1l" if args.l1_gelu else "gpt2"
        save_path = save_dir / f"vanilla_sae_{model_type}_{timestamp}.pt"
        
        save_dict = {
            "state_dict": sae.state_dict(),
            "config": vars(sae.config),
            "args": vars(args),
        }
        
        if metrics_history:
            save_dict["final_metrics"] = {k: values[-1] if values else None for k, values in metrics_history.items()}
        
        torch.save(save_dict, save_path)
        
        print(f"Model saved to {save_path}")
    
    return sae

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Set up TensorBoard directory if using it
    if args.use_tensorboard:
        tensorboard_dir = Path(args.tensorboard_dir)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        print(f"TensorBoard logs will be saved to {tensorboard_dir}")
        print("To visualize training, run: tensorboard --logdir=./runs")
    
    # Run tests if requested
    if args.run_tests:
        run_tests()
    
    # Create target model
    target_model = None
    if args.toy_model:
        target_model = setup_toy_model(args)
    elif args.l1_gelu:
        target_model = setup_gelu_1l_model(args)
    elif args.gpt2_small:
        target_model = setup_gpt2_model(args)
        return
    
    # Train SAE
    sae = None
    if args.vanilla_sae:
        sae = train_vanilla_sae(args, target_model)
    elif args.vsae_iso:
        vsae = train_vsae_iso(args, target_model)
    elif args.vsae_mix:
        vsae = train_vsae_mix(args, target_model)
    elif args.vsae_multi:
        vsae = train_vsae_multi(args, target_model)

if __name__ == "__main__":
    main()