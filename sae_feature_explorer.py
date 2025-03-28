#!/usr/bin/env python
"""
Simple SAE/VSAE Feature Explorer

A minimal tool to explore features in sparse autoencoders.

Usage:
    python simple_sae_explorer.py --model MODEL_PATH [--start_index INDEX]
"""

import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformer_lens import HookedTransformer, utils
except ImportError:
    print("Error: transformer_lens package is required.")
    print("Please install with: pip install transformer_lens")
    sys.exit(1)


class SimpleAutoEncoder(nn.Module):
    """A simple autoencoder compatible with both SAE and VSAE models."""
    
    def __init__(self, cfg):
        super().__init__()
        # Extract basic dimensions
        self.d_input = cfg.get("d_input", cfg.get("d_mlp", 2048))
        self.d_hidden = cfg.get("d_hidden", self.d_input * cfg.get("dict_mult", 8))
        
        # Initialize weights and biases
        self.W_enc = nn.Parameter(torch.zeros(self.d_input, self.d_hidden))
        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden))
        self.W_dec = nn.Parameter(torch.zeros(self.d_hidden, self.d_input))
        self.b_dec = nn.Parameter(torch.zeros(self.d_input))
        
        # Used for regularization value
        self.reg_coeff = cfg.get("l1_coeff", cfg.get("kl_coeff", 3e-4))
    
    def forward(self, x):
        # Ensure input is on the same device
        x = x.to(self.W_enc.device)
        
        # Center input
        x_centered = x - self.b_dec
        
        # Encode and apply ReLU
        acts = F.relu(x_centered @ self.W_enc + self.b_enc)
        
        # Decode
        x_recon = acts @ self.W_dec + self.b_dec
        
        return acts, x_recon
    
    @classmethod
    def load_from_file(cls, model_path):
        """Load model from file, handling different formats."""
        print(f"Loading model from {model_path}...")
        
        # Load state dict
        state_dict_obj = torch.load(model_path, map_location="cpu")
        
        # Extract config if available
        cfg = None
        if isinstance(state_dict_obj, dict) and not all(isinstance(v, torch.Tensor) for v in state_dict_obj.values()):
            if 'config' in state_dict_obj:
                cfg = state_dict_obj['config']
                print("Using config from state dict")
            elif 'cfg' in state_dict_obj:
                cfg = state_dict_obj['cfg']
                print("Using cfg from state dict")
            
            # Extract state dict
            if 'state_dict' in state_dict_obj:
                state_dict = state_dict_obj['state_dict']
            elif 'model_state_dict' in state_dict_obj:
                state_dict = state_dict_obj['model_state_dict']
            else:
                # Try to find tensor values
                for k, v in state_dict_obj.items():
                    if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values()):
                        state_dict = v
                        break
                else:
                    state_dict = state_dict_obj  # Use as-is if nothing better found
        else:
            # Direct state dict
            state_dict = state_dict_obj
            
            # Infer dimensions from shapes
            enc_key = next((k for k in state_dict.keys() if 'enc' in k.lower() and 'weight' in k.lower()), None)
            if enc_key and isinstance(state_dict[enc_key], torch.Tensor):
                shape = state_dict[enc_key].shape
                d_input = shape[0] if len(shape) >= 2 else state_dict[enc_key].numel()
                d_hidden = shape[1] if len(shape) >= 2 else state_dict[enc_key].numel()
                cfg = {"d_input": d_input, "d_hidden": d_hidden}
            else:
                # Fallback to defaults
                cfg = {"d_input": 2048, "d_hidden": 16384}
        
        # Create model instance
        instance = cls(cfg)
        
        # Try to find and map parameters
        param_mapping = {
            'W_enc': ['W_enc', 'encoder.weight'],
            'b_enc': ['b_enc', 'encoder.bias'],
            'W_dec': ['W_dec', 'decoder.weight'],
            'b_dec': ['b_dec', 'decoder.bias']
        }
        
        # Map parameters
        for param_name, possible_keys in param_mapping.items():
            for key in possible_keys:
                if key in state_dict:
                    value = state_dict[key].to("cpu")
                    if value.shape != getattr(instance, param_name).shape:
                        if param_name in ['W_enc', 'W_dec'] and len(value.shape) == 2:
                            if value.shape[1] == getattr(instance, param_name).shape[0] and value.shape[0] == getattr(instance, param_name).shape[1]:
                                value = value.t()  # Transpose if dimensions are swapped
                    
                    if value.shape == getattr(instance, param_name).shape:
                        getattr(instance, param_name).data.copy_(value)
                        break
        
        print(f"Model loaded with {instance.d_input} input features and {instance.d_hidden} hidden features")
        return instance


class FeatureExplorer:
    """Simple explorer for SAE/VSAE features."""
    
    def __init__(self, model_path, start_index=0):
        self.model_path = model_path
        self.current_idx = start_index
        self.model = None
        self.transformer = None
        
    def load_models(self):
        """Load autoencoder and transformer models."""
        # Load autoencoder
        self.model = SimpleAutoEncoder.load_from_file(self.model_path)
        
        # Load transformer
        print("Loading transformer model (gelu-1l)...")
        self.transformer = HookedTransformer.from_pretrained("gelu-1l")
        
        # Move both to the same device (use CUDA if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.transformer.to(device)
        
        print(f"Models loaded. Using device: {device}")
    
    def explore_feature(self, idx):
        """Explore a specific feature of the autoencoder."""
        if self.model is None:
            self.load_models()
        
        # Ensure index is in bounds
        idx = max(0, min(idx, self.model.d_hidden - 1))
        self.current_idx = idx
        
        # Print feature information
        print(f"\n{'='*80}")
        print(f"FEATURE {idx}/{self.model.d_hidden-1}")
        print(f"{'='*80}")
        
        # Get feature parameters
        feature_enc = self.model.W_enc[:, idx]
        feature_dec = self.model.W_dec[idx, :]
        feature_bias = self.model.b_enc[idx].item()
        
        # Show feature influence on vocabulary
        try:
            device = feature_dec.device
            W_out = self.transformer.W_out[0].to(device)
            W_U = self.transformer.W_U.to(device)
            
            logit_effect = feature_dec @ W_out @ W_U
            top_tokens_idx = torch.topk(logit_effect, 10).indices
            top_tokens = [self.transformer.to_string(idx.item()) for idx in top_tokens_idx]
            
            print("\nTop tokens influenced by this feature:")
            for i, token in enumerate(top_tokens):
                token_str = token.replace("\n", "↩").replace(" ", "·")
                print(f"  {i+1}. '{token_str}'")
        except Exception as e:
            print(f"Error calculating token influence: {e}")
        
        # Find example text that activates this feature
        self.show_example_activation(idx)
        
        # Show navigation help
        print("\nNavigation: n(ext) | b(ack) | [number] | q(uit)")
        
    def show_example_activation(self, idx):
        """Find example text that activates this feature."""
        # Example texts to test
        examples = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models can identify patterns in data.",
            "To be or not to be, that is the question.",
            "Neural networks have transformed artificial intelligence."
        ]
        
        print("\nSearching for examples that activate this feature...")
        
        device = self.model.W_enc.device
        
        # Initialize results
        activations = []
        
        # Process examples
        for text in examples:
            tokens = self.transformer.to_tokens(text, prepend_bos=True).to(device)
            
            with torch.no_grad():
                try:
                    # Get MLP activations
                    _, cache = self.transformer.run_with_cache(
                        tokens, 
                        stop_at_layer=1, 
                        names_filter=utils.get_act_name("post", 0)
                    )
                    mlp_acts = cache[utils.get_act_name("post", 0)]
                    seq_len = mlp_acts.shape[1]
                    
                    # Reshape and process
                    mlp_acts = mlp_acts.reshape(-1, self.model.d_input)
                    feature_in = self.model.W_enc[:, idx]
                    feature_bias = self.model.b_enc[idx]
                    
                    # Calculate activations
                    acts = F.relu((mlp_acts - self.model.b_dec) @ feature_in + feature_bias)
                    
                    # Find max activation
                    if acts.numel() > 0:
                        max_val = acts.max().item()
                        max_pos = acts.argmax().item() % seq_len
                        
                        # Get the activating token
                        text_tokens = self.transformer.to_str_tokens(tokens[0])
                        if max_pos < len(text_tokens) and max_val > 0:
                            token = text_tokens[max_pos]
                            activations.append((text, token, max_pos, max_val))
                except Exception as e:
                    print(f"Error processing example: {e}")
        
        # Sort by activation strength
        activations.sort(key=lambda x: x[3], reverse=True)
        
        # Display results
        if activations:
            print("\nExample text that activates this feature:")
            for i, (text, token, pos, val) in enumerate(activations[:3]):
                # Mark the activating token
                parts = list(self.transformer.to_str_tokens(self.transformer.to_tokens(text, prepend_bos=True)[0]))
                if pos < len(parts):
                    parts[pos] = f"[{parts[pos]}]"
                marked_text = "".join(parts)
                
                print(f"  {i+1}. '{marked_text}' (activation: {val:.4f})")
        else:
            print("No activations found in the example texts.")
    
    def run_interactive(self):
        """Run interactive exploration loop."""
        self.explore_feature(self.current_idx)
        
        while True:
            try:
                cmd = input("\nCommand: ").strip().lower()
                
                if cmd == 'q':
                    print("Exiting...")
                    break
                elif cmd == 'n':
                    self.explore_feature(self.current_idx + 1)
                elif cmd == 'b':
                    self.explore_feature(self.current_idx - 1)
                elif cmd.isdigit():
                    self.explore_feature(int(cmd))
                else:
                    print("Unknown command. Use n(ext), b(ack), [number], or q(uit)")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Parse arguments and run the feature explorer."""
    parser = argparse.ArgumentParser(description="Simple SAE Feature Explorer")
    parser.add_argument("--model", required=True, help="Path to model file (.pt)")
    parser.add_argument("--start_index", type=int, default=0, help="Starting feature index")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Run explorer
    explorer = FeatureExplorer(args.model, args.start_index)
    explorer.run_interactive()


if __name__ == "__main__":
    main()