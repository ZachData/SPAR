#!/usr/bin/env python3
"""
Analyze a trained SAE/VSAE on GPT-2 small model activations.
This script loads a trained model and performs various analyses and visualizations.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple, Optional
import os

from gpt2_model import GPT2Model, GPT2Config
from vanilla_sae import VanillaSAE, SAEConfig
from vsae_iso import VSAEIsoGaussian, VSAEIsoConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze trained SAE/VSAE on GPT-2")
    
    # Model loading
    parser.add_argument("model_path", type=str, help="Path to saved model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./analysis_results", 
                       help="Directory to save analysis results")
    
    # Analysis options
    parser.add_argument("--n_samples", type=int, default=1000, 
                       help="Number of activation samples to analyze")
    parser.add_argument("--feature_sparsity", action="store_true", 
                       help="Analyze feature sparsity")
    parser.add_argument("--neuron_activations", action="store_true",
                       help="Analyze neuron activation patterns")
    parser.add_argument("--reconstruction_quality", action="store_true",
                       help="Analyze reconstruction quality")
    parser.add_argument("--dictionary_similarity", action="store_true",
                       help="Analyze dictionary feature similarity")
    parser.add_argument("--latent_space", action="store_true",
                       help="Analyze latent space structure (requires UMAP)")
    
    # GPT-2 options for activation extraction
    parser.add_argument("--context_length", type=int, default=128,
                       help="Context length for GPT-2 text")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for activation extraction")
    
    # Utility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, 
                       help="Device to use (defaults to CUDA if available)")
    
    return parser.parse_args()


def load_model_checkpoint(model_path: str, device: str) -> Tuple[torch.nn.Module, Dict]:
    """
    Load a saved model checkpoint
    
    Args:
        model_path: Path to saved model
        device: Device to load model on
        
    Returns:
        Loaded model and checkpoint info
    """
    print(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = checkpoint["state_dict"]
    config_dict = checkpoint["config"]
    args_dict = checkpoint.get("args", {})
    
    # Determine model type from saved file or config
    is_vsae = "kl_coeff" in config_dict or ("use_vsae" in args_dict and args_dict["use_vsae"])
    
    if is_vsae:
        # Create VSAE config
        config = VSAEIsoConfig(
            d_input=config_dict["d_input"],
            d_hidden=config_dict["d_hidden"],
            dict_mult=config_dict["dict_mult"],
            kl_coeff=config_dict.get("kl_coeff", 3e-4),
            var_flag=config_dict.get("var_flag", 0),
            device=device
        )
        model = VSAEIsoGaussian(config)
    else:
        # Create SAE config
        config = SAEConfig(
            d_input=config_dict["d_input"],
            d_hidden=config_dict["d_hidden"],
            dict_mult=config_dict["dict_mult"],
            l1_coeff=config_dict.get("l1_coeff", 3e-4),
            device=device
        )
        model = VanillaSAE(config)
    
    # Load state dict
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Extract training info
    info = {
        "is_vsae": is_vsae,
        "layer": checkpoint.get("layer", 0),
        "act_name": checkpoint.get("act_name", "post"),
        "training_time": checkpoint.get("training_time", 0),
        "final_metrics": checkpoint.get("final_metrics", {}),
    }
    
    print(f"Model loaded: {'VSAE' if is_vsae else 'SAE'}, d_input={config.d_input}, d_hidden={config.d_hidden}")
    
    return model, info


def setup_gpt2_model(layer: int, act_name: str, context_length: int, 
                   batch_size: int, device: str) -> GPT2Model:
    """
    Set up a GPT-2 model for activation extraction
    
    Args:
        layer: Layer to extract activations from
        act_name: Activation type (post, pre, mid)
        context_length: Context length for GPT-2 text
        batch_size: Batch size for activation extraction
        device: Device to use
        
    Returns:
        GPT-2 model wrapper
    """
    config = GPT2Config(
        model_name="gpt2",
        activation_name=act_name,
        layer_num=layer,
        batch_size=batch_size,
        context_length=context_length,
        device=device
    )
    
    model = GPT2Model(config)
    
    return model


def extract_activations(gpt2_model: GPT2Model, n_samples: int) -> torch.Tensor:
    """
    Extract activations from GPT-2 model
    
    Args:
        gpt2_model: GPT-2 model wrapper
        n_samples: Number of activation samples to extract
        
    Returns:
        Extracted activations
    """
    print(f"Extracting {n_samples} activation samples...")
    batch_size = gpt2_model.config.batch_size
    samples_per_batch = batch_size * gpt2_model.config.context_length
    num_batches = (n_samples + samples_per_batch - 1) // samples_per_batch
    
    all_activations = []
    for i in range(num_batches):
        activations = gpt2_model.get_batch_activations()
        all_activations.append(activations)
    
    # Concatenate and trim to required number of samples
    activations = torch.cat(all_activations, dim=0)[:n_samples]
    
    print(f"Extracted activations shape: {activations.shape}")
    
    return activations


def analyze_feature_sparsity(model, activations: torch.Tensor, output_dir: Path) -> Dict:
    """
    Analyze feature sparsity
    
    Args:
        model: Trained SAE/VSAE model
        activations: Input activations
        output_dir: Output directory for visualizations
        
    Returns:
        Analysis results
    """
    print("Analyzing feature sparsity...")
    
    with torch.no_grad():
        if hasattr(model, 'encode'):
            # Standard SAE encode function
            features = model.encode(activations)
        else:
            # VSAE might have different encode function
            features = model(activations)["acts"]
    
    # Calculate activation frequency
    active_threshold = 1e-5
    is_active = (features.abs() > active_threshold).float()
    activation_frequency = is_active.mean(0)
    
    # Calculate average activation value when active
    masked_features = features * is_active
    activation_sum = masked_features.sum(0)
    activation_count = is_active.sum(0).clamp(min=1)  # Avoid division by zero
    average_activation = activation_sum / activation_count
    
    # Calculate percent of dead neurons (never activate)
    dead_neuron_mask = activation_frequency == 0
    percent_dead = dead_neuron_mask.float().mean().item() * 100
    
    # Plot activation frequency histogram
    plt.figure(figsize=(10, 6))
    plt.hist(activation_frequency.cpu().numpy(), bins=50, alpha=0.7)
    plt.axvline(x=0.01, color='r', linestyle='--', label='1% threshold')
    plt.title(f"Feature Activation Frequency (Dead: {percent_dead:.2f}%)")
    plt.xlabel("Activation Frequency")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(output_dir / "feature_activation_frequency.png")
    
    # Plot average activation value
    plt.figure(figsize=(10, 6))
    # Filter to only active features
    active_values = average_activation[~dead_neuron_mask].cpu().numpy()
    plt.hist(active_values, bins=50, alpha=0.7)
    plt.title("Average Feature Activation Value (When Active)")
    plt.xlabel("Activation Value")
    plt.ylabel("Count")
    plt.savefig(output_dir / "feature_activation_values.png")
    
    # Log scale plot to see the distribution better
    plt.figure(figsize=(10, 6))
    plt.hist(activation_frequency.cpu().numpy(), bins=50, alpha=0.7)
    plt.xscale('log')
    plt.title("Feature Activation Frequency (Log Scale)")
    plt.xlabel("Activation Frequency (Log Scale)")
    plt.ylabel("Count")
    plt.savefig(output_dir / "feature_activation_frequency_log.png")

    # Return results
    results = {
        "percent_dead_neurons": percent_dead,
        "median_activation_frequency": activation_frequency.median().item(),
        "mean_activation_frequency": activation_frequency.mean().item(),
        "mean_activation_value": average_activation[~dead_neuron_mask].mean().item(),
    }
    
    return results


def analyze_reconstruction_quality(model, activations: torch.Tensor, output_dir: Path) -> Dict:
    """
    Analyze reconstruction quality
    
    Args:
        model: Trained SAE/VSAE model
        activations: Input activations
        output_dir: Output directory for visualizations
        
    Returns:
        Analysis results
    """
    print("Analyzing reconstruction quality...")
    
    with torch.no_grad():
        # Forward pass through model
        if hasattr(model, 'forward'):
            outputs = model(activations)
            reconstructions = outputs["x_recon"]
        else:
            # Manually encode and decode
            features = model.encode(activations)
            reconstructions = model.decode(features)
        
        # Calculate reconstruction error
        mse = ((reconstructions - activations) ** 2).mean(dim=1)
        mse_mean = mse.mean().item()
        mse_std = mse.std().item()
        mse_median = mse.median().item()
        
        # Calculate cosine similarity
        normalized_activations = activations / activations.norm(dim=1, keepdim=True).clamp(min=1e-10)
        normalized_reconstructions = reconstructions / reconstructions.norm(dim=1, keepdim=True).clamp(min=1e-10)
        cosine_sim = (normalized_activations * normalized_reconstructions).sum(dim=1)
        cosine_sim_mean = cosine_sim.mean().item()
        cosine_sim_std = cosine_sim.std().item()
        
        # Calculate per-feature error
        feature_mse = ((reconstructions - activations) ** 2).mean(dim=0)
        feature_mse_mean = feature_mse.mean().item()
        feature_mse_std = feature_mse.std().item()
        
        # Plot MSE distribution
        plt.figure(figsize=(10, 6))
        plt.hist(mse.cpu().numpy(), bins=50, alpha=0.7)
        plt.axvline(x=mse_mean, color='r', linestyle='--', label=f'Mean: {mse_mean:.5f}')
        plt.axvline(x=mse_median, color='g', linestyle='--', label=f'Median: {mse_median:.5f}')
        plt.title("Reconstruction MSE Distribution")
        plt.xlabel("MSE")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(output_dir / "reconstruction_mse.png")
        
        # Plot cosine similarity distribution
        plt.figure(figsize=(10, 6))
        plt.hist(cosine_sim.cpu().numpy(), bins=50, alpha=0.7)
        plt.axvline(x=cosine_sim_mean, color='r', linestyle='--', label=f'Mean: {cosine_sim_mean:.5f}')
        plt.title("Reconstruction Cosine Similarity Distribution")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(output_dir / "reconstruction_cosine_sim.png")
        
        # Plot per-feature MSE
        plt.figure(figsize=(10, 6))
        feature_indices = np.arange(len(feature_mse))
        plt.bar(feature_indices, feature_mse.cpu().numpy(), alpha=0.7)
        plt.title("Per-Feature Reconstruction MSE")
        plt.xlabel("Feature Index")
        plt.ylabel("MSE")
        plt.savefig(output_dir / "per_feature_mse.png")
    
    # Return results
    results = {
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "mse_median": mse_median,
        "cosine_sim_mean": cosine_sim_mean,
        "cosine_sim_std": cosine_sim_std,
        "feature_mse_mean": feature_mse_mean,
        "feature_mse_std": feature_mse_std,
    }
    
    return results


def analyze_dictionary_similarity(model, output_dir: Path) -> Dict:
    """
    Analyze dictionary feature similarity
    
    Args:
        model: Trained SAE/VSAE model
        output_dir: Output directory for visualizations
        
    Returns:
        Analysis results
    """
    print("Analyzing dictionary feature similarity...")
    
    # Extract encoder and decoder weights
    W_enc = model.W_enc.detach().cpu()
    W_dec = model.W_dec.detach().cpu()
    
    # Calculate decoder weight norms
    decoder_norms = torch.norm(W_dec, dim=1)
    
    # Calculate cosine similarity between decoder features
    normalized_W_dec = W_dec / decoder_norms.unsqueeze(1)
    decoder_cosine_sim = torch.matmul(normalized_W_dec, normalized_W_dec.t())
    
    # Calculate encoder-decoder alignment
    normalized_W_enc = W_enc / torch.norm(W_enc, dim=0, keepdim=True)
    encoder_decoder_alignment = torch.abs(torch.matmul(normalized_W_enc.t(), normalized_W_dec.t()))
    
    # Get cosine similarity statistics
    decoder_cosine_sim_flat = decoder_cosine_sim.flatten()
    decoder_cosine_sim_triu = torch.triu(decoder_cosine_sim, diagonal=1).flatten()
    decoder_cosine_sim_triu = decoder_cosine_sim_triu[decoder_cosine_sim_triu != 0]
    
    # Calculate statistics
    mean_cosine_sim = decoder_cosine_sim_triu.mean().item()
    median_cosine_sim = decoder_cosine_sim_triu.median().item()
    max_cosine_sim = decoder_cosine_sim_triu.max().item()
    
    # Plot decoder feature cosine similarity heatmap
    plt.figure(figsize=(10, 8))
    # Sample if too large
    if decoder_cosine_sim.shape[0] > 500:
        indices = np.random.choice(decoder_cosine_sim.shape[0], 500, replace=False)
        sampled_sim = decoder_cosine_sim[indices][:, indices]
        plt.imshow(sampled_sim, cmap='viridis')
        plt.title("Dictionary Cosine Similarity (500 Random Features)")
    else:
        plt.imshow(decoder_cosine_sim, cmap='viridis')
        plt.title("Dictionary Cosine Similarity")
    plt.colorbar()
    plt.savefig(output_dir / "dictionary_cosine_sim.png")
    
    # Plot cosine similarity distribution
    plt.figure(figsize=(10, 6))
    plt.hist(decoder_cosine_sim_triu.numpy(), bins=50, alpha=0.7)
    plt.axvline(x=mean_cosine_sim, color='r', linestyle='--', label=f'Mean: {mean_cosine_sim:.5f}')
    plt.axvline(x=median_cosine_sim, color='g', linestyle='--', label=f'Median: {median_cosine_sim:.5f}')
    plt.title("Dictionary Cosine Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(output_dir / "dictionary_cosine_sim_dist.png")
    
    # Plot encoder-decoder alignment
    plt.figure(figsize=(10, 8))
    if encoder_decoder_alignment.shape[0] > 500:
        indices = np.random.choice(encoder_decoder_alignment.shape[0], 500, replace=False)
        sampled_alignment = encoder_decoder_alignment[indices][:, :500]
        plt.imshow(sampled_alignment, cmap='viridis')
        plt.title("Encoder-Decoder Alignment (500 Random Features)")
    else:
        plt.imshow(encoder_decoder_alignment, cmap='viridis')
        plt.title("Encoder-Decoder Alignment")
    plt.colorbar()
    plt.savefig(output_dir / "encoder_decoder_alignment.png")
    
    # Calculate alignment diagonal vs off-diagonal
    diagonal_indices = torch.arange(min(encoder_decoder_alignment.shape))
    alignment_diagonal = encoder_decoder_alignment[diagonal_indices, diagonal_indices]
    mean_diagonal_alignment = alignment_diagonal.mean().item()
    
    # Return results
    results = {
        "mean_cosine_sim": mean_cosine_sim,
        "median_cosine_sim": median_cosine_sim,
        "max_cosine_sim": max_cosine_sim,
        "mean_diagonal_alignment": mean_diagonal_alignment,
        "std_decoder_norms": decoder_norms.std().item(),
    }
    
    return results


def analyze_latent_space(model, activations: torch.Tensor, output_dir: Path) -> Dict:
    """
    Analyze latent space structure using UMAP
    
    Args:
        model: Trained SAE/VSAE model
        activations: Input activations
        output_dir: Output directory for visualizations
        
    Returns:
        Analysis results
    """
    print("Analyzing latent space structure...")
    
    try:
        import umap
    except ImportError:
        print("UMAP not installed. Please install with: pip install umap-learn")
        return {"error": "UMAP not installed"}
    
    with torch.no_grad():
        if hasattr(model, 'encode'):
            # Standard SAE encode function
            features = model.encode(activations)
        else:
            # VSAE might have different encode function
            features = model(activations)["acts"]
    
    # Convert to numpy for UMAP
    features_np = features.cpu().numpy()
    
    # Calculate sparsity for coloring
    sparsity = (features_np > 0).sum(axis=1) / features_np.shape[1]
    
    # Create UMAP embedding
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(features_np)
    
    # Plot embedding colored by sparsity
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=sparsity, cmap='viridis', alpha=0.7, s=5)
    plt.colorbar(label='Sparsity (Fraction of Active Neurons)')
    plt.title("UMAP Projection of Latent Space")
    plt.savefig(output_dir / "latent_umap.png")
    
    # Try to estimate cluster structure
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(embedding)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Plot embedding colored by cluster
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20', alpha=0.7, s=5)
    plt.title(f"UMAP Projection with Clustering (Found {n_clusters} clusters)")
    plt.savefig(output_dir / "latent_umap_clusters.png")
    
    # Return results
    results = {
        "n_clusters": n_clusters,
        "noise_points": (labels == -1).sum(),
    }
    
    return results


def analyze_neuron_activations(model, activations: torch.Tensor, output_dir: Path) -> Dict:
    """
    Analyze neuron activation patterns
    
    Args:
        model: Trained SAE/VSAE model
        activations: Input activations
        output_dir: Output directory for visualizations
        
    Returns:
        Analysis results
    """
    print("Analyzing neuron activation patterns...")
    
    with torch.no_grad():
        if hasattr(model, 'encode'):
            # Standard SAE encode function
            features = model.encode(activations)
        else:
            # VSAE might have different encode function
            features = model(activations)["acts"]
    
    # Calculate neuron statistics
    active_threshold = 1e-5
    is_active = (features > active_threshold).float()
    neuron_activation_rate = is_active.mean(0)
    
    # Calculate co-activation patterns
    co_activation = torch.matmul(is_active.t(), is_active) / is_active.shape[0]
    
    # Calculate average activation value
    active_values = features * is_active
    activation_sum = active_values.sum(0)
    activation_count = is_active.sum(0).clamp(min=1)
    average_activation = activation_sum / activation_count
    
    # Sort neurons by activation rate for visualization
    sorted_indices = torch.argsort(neuron_activation_rate, descending=True)
    sorted_rates = neuron_activation_rate[sorted_indices]
    
    # Plot sorted activation rates
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_rates.cpu().numpy())
    plt.title("Neuron Activation Rates (Sorted)")
    plt.xlabel("Neuron Index (Sorted)")
    plt.ylabel("Activation Rate")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "neuron_activation_rates.png")
    
    # Plot co-activation heatmap for top neurons
    n_top_neurons = min(100, len(sorted_indices))
    top_indices = sorted_indices[:n_top_neurons]
    top_co_activation = co_activation[top_indices][:, top_indices]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(top_co_activation.cpu().numpy(), cmap='viridis')
    plt.title(f"Top {n_top_neurons} Neurons Co-activation Matrix")
    plt.savefig(output_dir / "neuron_coactivation.png")
    
    # Calculate activation correlation
    features_np = features.cpu().numpy()
    correlation = np.corrcoef(features_np.T)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation[:n_top_neurons, :n_top_neurons], cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"Top {n_top_neurons} Neurons Correlation Matrix")
    plt.savefig(output_dir / "neuron_correlation.png")
    
    # Return results
    results = {
        "mean_activation_rate": neuron_activation_rate.mean().item(),
        "median_activation_rate": neuron_activation_rate.median().item(),
        "mean_activation_value": average_activation.mean().item(),
        "mean_coactivation": co_activation.mean().item(),
    }
    
    return results


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model checkpoint
    model, model_info = load_model_checkpoint(args.model_path, device)
    
    # Set up GPT-2 model
    gpt2_model = setup_gpt2_model(
        layer=model_info["layer"],
        act_name=model_info["act_name"],
        context_length=args.context_length,
        batch_size=args.batch_size,
        device=device
    )
    
    # Extract activations
    activations = extract_activations(gpt2_model, args.n_samples)
    
    # Run analyses
    results = {}
    
    # Always include some basic model info
    results["model_info"] = {
        "is_vsae": model_info["is_vsae"],
        "layer": model_info["layer"],
        "act_name": model_info["act_name"],
        "training_time": model_info["training_time"],
        "d_input": model.config.d_input,
        "d_hidden": model.config.d_hidden,
        "dict_mult": model.config.dict_mult,
    }
    
    # Add final training metrics if available
    if model_info["final_metrics"]:
        results["training_metrics"] = model_info["final_metrics"]
    
    # Run requested analyses
    if args.feature_sparsity:
        results["feature_sparsity"] = analyze_feature_sparsity(model, activations, output_dir)
    
    if args.reconstruction_quality:
        results["reconstruction_quality"] = analyze_reconstruction_quality(model, activations, output_dir)
    
    if args.dictionary_similarity:
        results["dictionary_similarity"] = analyze_dictionary_similarity(model, output_dir)
    
    if args.neuron_activations:
        results["neuron_activations"] = analyze_neuron_activations(model, activations, output_dir)
    
    if args.latent_space:
        results["latent_space"] = analyze_latent_space(model, activations, output_dir)
    
    # Save results
    results_path = output_dir / "analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis complete. Results saved to {results_path}")


if __name__ == "__main__":
    main()
