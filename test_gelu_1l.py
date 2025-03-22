#!/usr/bin/env python3
"""
Test script for training SAEs on the 1-layer GELU transformer model.
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from gelu_1l_model import GELU1LModel, GELU1LConfig
from vanilla_sae import VanillaSAE, SAEConfig
from model_trainer import ModelTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Test SAE training on GELU-1L model")
    
    parser.add_argument("--dict_mult", type=int, default=8, help="Dictionary multiplier")
    parser.add_argument("--l1_coeff", type=float, default=3e-4, help="L1 coefficient")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--steps", type=int, default=10000, help="Number of steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_model", action="store_true", help="Save model after training")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create GELU-1L model
    gelu_config = GELU1LConfig(
        batch_size=args.batch_size,
        seed=args.seed
    )
    gelu_model = GELU1LModel(gelu_config)
    
    # Print model info
    print(f"Model loaded: {gelu_config.model_name}")
    print(f"MLP dimension: {gelu_model.d_mlp}")
    
    # Create SAE
    sae_config = SAEConfig(
        d_input=gelu_model.d_mlp,
        d_hidden=gelu_model.d_mlp * args.dict_mult,
        dict_mult=args.dict_mult,
        l1_coeff=args.l1_coeff,
    )
    sae = VanillaSAE(sae_config)
    
    # Create trainer
    trainer = ModelTrainer(
        source_model=gelu_model,
        autoencoder=sae,
        lr=args.lr,
        steps=args.steps,
        batch_size=args.batch_size,
        log_freq=100
    )
    
    # Train SAE
    print(f"Training SAE on GELU-1L activations for {args.steps} steps...")
    metrics_history = trainer.train()
    
    # Save model if requested
    if args.save_model:
        save_dir = Path(args.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / "sae_gelu1l_test.pt"
        torch.save({
            "state_dict": sae.state_dict(),
            "config": vars(sae.config),
        }, save_path)
        
        print(f"Model saved to {save_path}")
    
    # Print final metrics
    print("\nFinal metrics:")
    for key, values in metrics_history.items():
        if values:
            print(f"{key}: {values[-1]:.6f}")


if __name__ == "__main__":
    main()
