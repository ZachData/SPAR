#!/usr/bin/env python3
"""
Test script for training SAEs on the GPT-2 small model.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import time

from gpt2_model import GPT2Model, GPT2Config
from vanilla_sae import VanillaSAE, SAEConfig
from vsae_iso import VSAEIsoGaussian, VSAEIsoConfig
from main import ModelTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAE or VSAE on GPT-2 small model")
    
    # Model selection
    parser.add_argument("--use_vsae", action="store_true", help="Use VSAE instead of vanilla SAE")
    parser.add_argument("--var_flag", type=int, default=0, choices=[0, 1], 
                       help="For VSAE: whether to learn variance (0: fixed, 1: learned)")
    
    # GPT-2 configuration
    parser.add_argument("--layer", type=int, default=0, help="Layer to extract activations from (0-11)")
    parser.add_argument("--act_name", type=str, default="post", choices=["post", "pre", "mid"],
                       help="Activation type (post: after FFN, pre: before FFN, mid: after first FFN layer)")
    
    # SAE hyperparameters
    parser.add_argument("--dict_mult", type=int, default=8, help="Dictionary multiplier")
    parser.add_argument("--l1_coeff", type=float, default=3e-4, help="L1 coefficient (for SAE)")
    parser.add_argument("--kl_coeff", type=float, default=3e-4, help="KL coefficient (for VSAE)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--steps", type=int, default=10000, help="Number of steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_decay", choices=["constant", "linear", "cosine"], default="constant", 
                       help="Learning rate decay schedule")
    
    # Utility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_model", action="store_true", help="Save model after training")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (smaller model, fewer steps)")
    
    return parser.parse_args()


def get_lr_scheduler(schedule: str):
    """Get learning rate scheduler function"""
    if schedule == "linear":
        return lambda step, steps: 1.0 - (step / steps)
    elif schedule == "cosine":
        return lambda step, steps: np.cos(0.5 * np.pi * step / (steps - 1))
    else:
        return lambda step, steps: 1.0


def main():
    args = parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Debug mode
    if args.debug:
        print("Running in debug mode with reduced parameters")
        args.steps = 100
        args.batch_size = 16
    
    # Create GPT-2 model
    gpt2_config = GPT2Config(
        model_name="gpt2",  # Use GPT-2 small
        activation_name=args.act_name,
        layer_num=args.layer,
        batch_size=args.batch_size,
        seed=args.seed
    )
    gpt2_model = GPT2Model(gpt2_config)
    
    # Print model info
    print(f"Layer: {args.layer}, Activation: {args.act_name}")
    print(f"MLP dimension: {gpt2_model.d_mlp}")
    
    # Create SAE or VSAE
    if args.use_vsae:
        # Create VSAE
        vsae_config = VSAEIsoConfig(
            d_input=gpt2_model.d_mlp,
            d_hidden=gpt2_model.d_mlp * args.dict_mult,
            dict_mult=args.dict_mult,
            kl_coeff=args.kl_coeff,
            var_flag=args.var_flag,
        )
        autoencoder = VSAEIsoGaussian(vsae_config)
        print(f"Created VSAE with var_flag={args.var_flag}, KL coeff={args.kl_coeff}")
    else:
        # Create SAE
        sae_config = SAEConfig(
            d_input=gpt2_model.d_mlp,
            d_hidden=gpt2_model.d_mlp * args.dict_mult,
            dict_mult=args.dict_mult,
            l1_coeff=args.l1_coeff,
        )
        autoencoder = VanillaSAE(sae_config)
        print(f"Created vanilla SAE with L1 coeff={args.l1_coeff}")
    
    # Get learning rate scheduler
    lr_scheduler = get_lr_scheduler(args.lr_decay)
    
    # Create trainer
    trainer = ModelTrainer(
        source_model=gpt2_model,
        autoencoder=autoencoder,
        lr=args.lr,
        steps=args.steps,
        batch_size=args.batch_size,
        lr_scale=lr_scheduler,
        log_freq=10 if args.debug else 100
    )
    
    # Train autoencoder
    print(f"Training on GPT-2 small activations for {args.steps} steps...")
    start_time = time.time()
    metrics_history = trainer.train()
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save model if requested
    if args.save_model:
        save_dir = Path(args.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_type = "vsae" if args.use_vsae else "sae"
        save_path = save_dir / f"{model_type}_gpt2small_l{args.layer}_{args.act_name}_{timestamp}.pt"
        
        torch.save({
            "state_dict": autoencoder.state_dict(),
            "config": vars(autoencoder.config),
            "args": vars(args),
            "layer": args.layer,
            "act_name": args.act_name,
            "training_time": training_time,
            "final_metrics": {k: values[-1] if values else None for k, values in metrics_history.items()},
        }, save_path)
        
        print(f"Model saved to {save_path}")
    
    # Print final metrics
    print("\nFinal metrics:")
    for key, values in metrics_history.items():
        if values:
            print(f"{key}: {values[-1]:.6f}")


if __name__ == "__main__":
    main()
