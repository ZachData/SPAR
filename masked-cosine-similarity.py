import torch
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer, utils
import sys
import os

# Import the GELU1LModel class
from gelu_1l_model import GELU1LModel, GELU1LConfig

def get_model_dimension(sae_model):
    """
    Extract the hidden dimension from an SAE model, handling different formats.
    
    Args:
        sae_model: Loaded SAE model (could be a model object or dictionary)
        
    Returns:
        Hidden dimension of the SAE model
    """
    # Case 1: Model has a config attribute with d_hidden
    if hasattr(sae_model, 'config') and hasattr(sae_model.config, 'd_hidden'):
        return sae_model.config.d_hidden
    
    # Case 2: Model is a dictionary with a 'config' key
    if isinstance(sae_model, dict) and 'config' in sae_model:
        config = sae_model['config']
        if isinstance(config, dict) and 'd_hidden' in config:
            return config['d_hidden']
    
    # Case 3: Model is a dictionary with a 'state_dict' key
    if isinstance(sae_model, dict) and 'state_dict' in sae_model:
        state_dict = sae_model['state_dict']
        # Try to find encoder weights to infer dimension
        for key in state_dict.keys():
            if 'W_enc' in key:
                return state_dict[key].shape[1]  # Assume W_enc shape is [d_input, d_hidden]
    
    # Case 4: Model itself is a state dict
    if isinstance(sae_model, dict):
        # Try to find encoder weights to infer dimension
        for key in sae_model.keys():
            if 'W_enc' in key:
                return sae_model[key].shape[1]  # Assume W_enc shape is [d_input, d_hidden]
    
    # If all else fails, look at any parameter shapes
    if isinstance(sae_model, dict):
        for key, value in sae_model.items():
            if isinstance(value, torch.Tensor) and len(value.shape) == 2:
                # Return the second dimension of the first matrix we find
                return value.shape[1]
    
    # If we can't determine it, raise an error
    raise ValueError("Could not determine hidden dimension of SAE model")

def encode_data(sae_model, input_data):
    """
    Encode data using an SAE model, handling different model formats.
    
    Args:
        sae_model: Loaded SAE model (could be a model object or dictionary)
        input_data: Input data to encode
        
    Returns:
        Encoded activations
    """
    # Case 1: Model has an encode method
    if hasattr(sae_model, 'encode'):
        return sae_model.encode(input_data, deterministic=True)
    
    # Case 2: Model is a dictionary with state_dict
    if isinstance(sae_model, dict):
        # Extract parameters from state dict
        if 'state_dict' in sae_model:
            state_dict = sae_model['state_dict']
        else:
            state_dict = sae_model
        
        # Find encoder parameters
        W_enc = None
        b_enc = None
        b_dec = None
        
        for key, value in state_dict.items():
            if key.endswith('W_enc'):
                W_enc = value
            elif key.endswith('b_enc'):
                b_enc = value
            elif key.endswith('b_dec'):
                b_dec = value
        
        if W_enc is None or b_enc is None or b_dec is None:
            raise ValueError("Could not find encoder parameters in state dict")
        
        # Apply encoding operation manually
        x_cent = input_data - b_dec
        return torch.relu(x_cent @ W_enc + b_enc)
    
    # If we can't encode, raise an error
    raise ValueError("Could not encode data with provided SAE model")

def masked_cosine_similarity(
    feature1_activations: torch.Tensor,
    feature2_activations: torch.Tensor,
    threshold: float = 1e-8
) -> float:
    """
    Compute the masked cosine similarity between two feature activation vectors.
    
    The masked cosine similarity is defined as the maximum of:
    1. Cosine similarity between activations on the subset where feature1 fires
    2. Cosine similarity between activations on the subset where feature2 fires
    
    Args:
        feature1_activations: Activation values for feature 1 across tokens [n_tokens]
        feature2_activations: Activation values for feature 2 across tokens [n_tokens]
        threshold: Minimum activation value to consider a feature "firing"
        
    Returns:
        Maximum of the two masked cosine similarities
    """
    # Check dimensions match
    assert feature1_activations.shape == feature2_activations.shape, "Feature activation dimensions must match"
    
    # Get masks where each feature fires
    mask1 = feature1_activations > threshold
    mask2 = feature2_activations > threshold
    
    # If either feature never fires, return 0
    if not torch.any(mask1) or not torch.any(mask2):
        return 0.0
    
    # Calculate cosine similarity on mask1 (where feature1 fires)
    sim1 = 0.0
    if torch.any(mask1):
        f1_masked = feature1_activations[mask1]
        f2_masked = feature2_activations[mask1]
        
        # Normalize vectors
        f1_norm = torch.norm(f1_masked)
        f2_norm = torch.norm(f2_masked)
        
        # Handle zero norms
        if f1_norm > 0 and f2_norm > 0:
            dot_product = torch.dot(f1_masked, f2_masked)
            # Convert to Python float
            if isinstance(dot_product, torch.Tensor):
                sim1 = float(dot_product.item()) / (float(f1_norm.item()) * float(f2_norm.item()))
            else:
                sim1 = float(dot_product) / (float(f1_norm) * float(f2_norm))
    
    # Calculate cosine similarity on mask2 (where feature2 fires)
    sim2 = 0.0
    if torch.any(mask2):
        f1_masked = feature1_activations[mask2]
        f2_masked = feature2_activations[mask2]
        
        # Normalize vectors
        f1_norm = torch.norm(f1_masked)
        f2_norm = torch.norm(f2_masked)
        
        # Handle zero norms
        if f1_norm > 0 and f2_norm > 0:
            dot_product = torch.dot(f1_masked, f2_masked)
            # Convert to Python float
            if isinstance(dot_product, torch.Tensor):
                sim2 = float(dot_product.item()) / (float(f1_norm.item()) * float(f2_norm.item()))
            else:
                sim2 = float(dot_product) / (float(f1_norm) * float(f2_norm))
    
    # Return the maximum of the two similarities
    return max(sim1, sim2)

def get_sae_activations(sae, base_model, num_samples=1000, batch_size=16, device="cuda"):
    """
    Get activations for an SAE using the GELU1LModel to provide input activations.
    
    Args:
        sae: The sparse autoencoder model
        base_model: The GELU1LModel instance
        num_samples: Number of token samples to process
        batch_size: Batch size for processing
        device: Device to use for computation
        
    Returns:
        Tensor of activations [n_features, n_tokens]
    """
    # Get the hidden dimension of the SAE model
    n_features = get_model_dimension(sae)
    print(f"Detected SAE hidden dimension: {n_features}")
    
    all_token_activations = []
    all_sae_activations = []
    
    print(f"Getting activations for {num_samples} tokens...")
    total_tokens = 0
    pbar = tqdm(total=num_samples)
    
    with torch.no_grad():
        while total_tokens < num_samples:
            # Generate batch of tokens and get their activations from base model
            tokens = base_model.generate_batch(batch_size)
            mlp_activations = base_model.get_activations(tokens)
            
            # Use our helper function to encode with the SAE
            sae_activations = encode_data(sae, mlp_activations)
            
            # Track activations
            all_sae_activations.append(sae_activations)
            
            # Update progress
            batch_tokens = sae_activations.size(0)
            total_tokens += batch_tokens
            pbar.update(min(batch_tokens, num_samples - (total_tokens - batch_tokens)))
    
    pbar.close()
    
    # Concatenate all activations
    all_sae_activations = torch.cat(all_sae_activations, dim=0)
    
    # Limit to exactly num_samples
    all_sae_activations = all_sae_activations[:num_samples]
    
    # Transpose to get [n_features, n_tokens]
    return all_sae_activations.T

def compare_feature_to_sae(
    target_feature_idx: int,
    sae1_activations: torch.Tensor,
    sae2_activations: torch.Tensor,
    threshold: float = 1e-8,
    min_density: float = 0.001
) -> List[Tuple[int, float]]:
    """
    Compare a specific feature from SAE1 with all features from SAE2,
    and return a sorted list from most similar to least similar.
    
    Args:
        target_feature_idx: Index of the feature from SAE1 to compare
        sae1_activations: Activation tensor for first SAE [n_features_1, n_tokens]
        sae2_activations: Activation tensor for second SAE [n_features_2, n_tokens]
        threshold: Minimum activation value to consider a feature "firing"
        min_density: Minimum firing rate to keep a feature
        
    Returns:
        List of tuples (feature_index, similarity_score) sorted by similarity
    """
    # Check that the target feature exists
    assert 0 <= target_feature_idx < sae1_activations.shape[0], f"Target feature index out of range: {target_feature_idx}"
    
    # Get the activations for the target feature
    target_activations = sae1_activations[target_feature_idx]
    
    # Calculate firing rate for the target feature
    target_firing_rate = (target_activations > threshold).float().mean().item()
    print(f"Target feature {target_feature_idx} firing rate: {target_firing_rate:.6f}")
    
    # Skip comparison if target feature has too low density
    if target_firing_rate < min_density:
        print(f"Warning: Target feature {target_feature_idx} has very low firing rate ({target_firing_rate:.6f}), below threshold {min_density}")
    
    # Calculate similarities with all features from SAE2
    n_features_2 = sae2_activations.shape[0]
    similarities = []
    
    print(f"Comparing target feature {target_feature_idx} with {n_features_2} features from SAE2...")
    for i in tqdm(range(n_features_2)):
        # Calculate density for this feature
        feature_firing_rate = (sae2_activations[i] > threshold).float().mean().item()
        
        # Skip low-density features
        if feature_firing_rate < min_density:
            similarities.append((i, 0.0))
            continue
        
        # Calculate similarity
        sim = masked_cosine_similarity(target_activations, sae2_activations[i], threshold)
        similarities.append((i, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities

def plot_top_similarities(similarities, top_k=20, title=None):
    """
    Plot the top-k most similar features.
    
    Args:
        similarities: List of tuples (feature_index, similarity_score)
        top_k: Number of top features to plot
        title: Plot title
    """
    top_k = min(top_k, len(similarities))
    top_features = similarities[:top_k]
    
    # Extract indices and scores
    indices = [x[0] for x in top_features]
    scores = [x[1] for x in top_features]
    
    plt.figure(figsize=(10, 8))
    bars = plt.barh(range(top_k), scores, align='center')
    plt.yticks(range(top_k), [f"Feature {idx}" for idx in indices])
    plt.xlabel('Similarity Score')
    plt.title(title or f"Top {top_k} Most Similar Features")
    plt.gca().invert_yaxis()  # Highest similarity at the top
    
    # Add scores as text labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(score + 0.01, i, f"{score:.4f}", va='center')
    
    plt.tight_layout()
    plt.show()

def main_comparison(sae1, sae2, target_feature_idx=269, top_k=50, threshold=1e-8, min_density=0.001, num_samples=1000):
    """
    Main function to compare a specific feature from SAE1 with all features from SAE2.
    
    Args:
        sae1: First sparse autoencoder model
        sae2: Second sparse autoencoder model
        target_feature_idx: Index of the feature to compare from SAE1
        top_k: Number of top similar features to display
        threshold: Minimum activation value to consider a feature "firing"
        min_density: Minimum firing rate to keep a feature
        num_samples: Number of token samples to process
        
    Returns:
        DataFrame containing sorted similarity results
    """
    # Set up a GELU1LModel to provide input activations
    print("Initializing GELU1L model...")
    gelu_config = GELU1LConfig(
        model_name="gelu-1l",
        activation_name="post",
        layer_num=0,
        batch_size=16,
        context_length=128,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dataset_name="roneneldan/TinyStories"
    )
    gelu_model = GELU1LModel(gelu_config)
    
    # Get activations for both SAEs using the same GELU model
    print("Getting activations for SAE1...")
    sae1_activations = get_sae_activations(sae1, gelu_model, num_samples=num_samples)
    
    print("Getting activations for SAE2...")
    sae2_activations = get_sae_activations(sae2, gelu_model, num_samples=num_samples)
    
    # Compare the target feature with all features from SAE2
    print(f"Comparing feature {target_feature_idx} from SAE1 with all features from SAE2...")
    similarities = compare_feature_to_sae(
        target_feature_idx,
        sae1_activations,
        sae2_activations,
        threshold=threshold,
        min_density=min_density
    )
    
    # Create DataFrame for output
    results_df = pd.DataFrame(similarities, columns=['Feature_Index', 'Similarity'])
    
    # Print top K similar features
    print(f"\nTop {top_k} features in SAE2 most similar to feature {target_feature_idx} in SAE1:")
    for i, (idx, sim) in enumerate(similarities[:top_k]):
        print(f"{i+1}. Feature {idx}: similarity = {sim:.6f}")
    
    # Plot top similarities
    plot_top_similarities(
        similarities, 
        top_k=top_k, 
        title=f"Top {top_k} Features in SAE2 Similar to Feature {target_feature_idx} in SAE1"
    )
    
    return results_df

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare a specific feature between two SAEs")
    parser.add_argument("--sae1_path", type=str, required=True, help="Path to first SAE model")
    parser.add_argument("--sae2_path", type=str, required=True, help="Path to second SAE model")
    parser.add_argument("--feature_idx", type=int, default=269, help="Index of feature from SAE1 to compare")
    parser.add_argument("--top_k", type=int, default=50, help="Number of top similar features to display")
    parser.add_argument("--threshold", type=float, default=1e-8, help="Minimum activation value")
    parser.add_argument("--min_density", type=float, default=0.001, help="Minimum firing rate")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of token samples to process")
    parser.add_argument("--output_csv", type=str, help="Path to save results CSV")
    
    args = parser.parse_args()
    
    # Load SAE models
    print(f"Loading SAE1 from {args.sae1_path}...")
    sae1 = torch.load(args.sae1_path)
    
    print(f"Loading SAE2 from {args.sae2_path}...")
    sae2 = torch.load(args.sae2_path)
    
    # Run comparison
    results_df = main_comparison(
        sae1, 
        sae2, 
        target_feature_idx=args.feature_idx,
        top_k=args.top_k,
        threshold=args.threshold,
        min_density=args.min_density,
        num_samples=args.num_samples
    )
    
    # Save results if requested
    if args.output_csv:
        results_df.to_csv(args.output_csv, index=False)
        print(f"Results saved to {args.output_csv}")