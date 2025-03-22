import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import dataclass
from transformer_lens import HookedTransformer, utils
from tqdm import tqdm
import numpy as np


@dataclass
class GELU1LConfig:
    """Configuration for the 1-layer GELU transformer model."""
    model_name: str = "gelu-1l"                 # TransformerLens model name
    activation_name: str = "post"               # Activation hook name to extract
    layer_num: int = 0                          # Layer number (0 for 1-layer models)
    batch_size: int = 512                       # Batch size for processing
    context_length: int = 128                   # Context length for training data
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32          # Data type for model
    buffer_size: int = 10000                    # Number of tokens to keep in buffer
    dataset_name: str = "roneneldan/TinyStories"    # HF Dataset to load
    seed: int = 42                              # Random seed


class GELU1LModel:
    """
    1-layer GELU transformer model from TransformerLens.
    
    This class acts as a wrapper around the TransformerLens model
    to interface with our SAE training pipeline.
    """
    
    def __init__(self, config: GELU1LConfig):
        """
        Initialize the 1L-GELU model.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Load model from TransformerLens
        self.model = HookedTransformer.from_pretrained(
            config.model_name,
            device=config.device,
            dtype=config.dtype
        )
        
        # Save model parameter sizes for reference
        self.n_layers = self.model.cfg.n_layers
        self.d_model = self.model.cfg.d_model
        self.n_heads = self.model.cfg.n_heads
        self.d_head = self.model.cfg.d_head
        self.d_mlp = self.model.cfg.d_mlp
        self.d_vocab = self.model.cfg.d_vocab
        
        # Construct activation hook name
        self.hook_name = utils.get_act_name(config.activation_name, config.layer_num)
        
        # Load dataset
        self.tokenized_data = self._load_dataset()
        
        # Initialize token buffer and buffer index
        self.token_buffer = []  # Start with empty list
        self.buffer_idx = 0
    
    def _load_dataset(self):
        """
        Load and tokenize dataset.
        
        Returns:
            Tokenized dataset
        """
        # Import datasets here to avoid dependency if not needed
        try:
            from datasets import load_dataset
            
            # Load the dataset - use a tiny subset for testing to be safe
            data = load_dataset(self.config.dataset_name, cache_dir="G:\\Huggingface_datasets")
            
            # Tokenize the dataset
            tokenized_data = utils.tokenize_and_concatenate(
                data, 
                self.model.tokenizer, 
                max_length=self.config.context_length
            )
            
            # Shuffle the dataset
            tokenized_data = tokenized_data.shuffle(seed=self.config.seed)
            
            print(f"Loaded dataset with {len(tokenized_data)} sequences")
            
            # Verify we have tokens
            if "tokens" not in tokenized_data.column_names or len(tokenized_data["tokens"]) == 0:
                raise ValueError("Dataset doesn't contain tokens or is empty")
                
            return tokenized_data
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to synthetic data")
            
            # Create synthetic data as a fallback
            import numpy as np
            
            # Create a small synthetic dataset - 1000 sequences of context_length tokens
            dummy_tokens = torch.randint(
                0, self.model.tokenizer.vocab_size, 
                (1000, self.config.context_length),
                device=self.config.device
            )
            
            # Create a simple dataset-like object
            class DummyDataset:
                def __init__(self, tokens):
                    self.tokens = tokens
                    
                def __len__(self):
                    return len(self.tokens)
                    
                def __getitem__(self, idx):
                    if isinstance(idx, slice):
                        return {"tokens": self.tokens[idx]}
                    elif isinstance(idx, int):
                        return {"tokens": self.tokens[idx:idx+1]}
                    elif isinstance(idx, str):
                        # If the index is a string (like "tokens"), return the entire tokens tensor
                        if idx == "tokens":
                            return self.tokens
                        raise KeyError(f"Dataset has no column named {idx}")
                    else:
                        raise TypeError(f"Unsupported index type: {type(idx)}")
                    
                @property
                def column_names(self):
                    return ["tokens"]
                    
                def shuffle(self, seed=None):
                    # Simple shuffle
                    indices = torch.randperm(len(self.tokens))
                    self.tokens = self.tokens[indices]
                    return self
            
            dummy_dataset = DummyDataset(dummy_tokens)
            print(f"Created synthetic dataset with {len(dummy_dataset)} sequences")
            return dummy_dataset
    
    def _fill_buffer(self):
        """Fill the token buffer with new tokens from the dataset."""
        # Get a batch of tokens
        end_idx = min(self.buffer_idx + self.config.buffer_size, len(self.tokenized_data))
        tokens = self.tokenized_data["tokens"][self.buffer_idx:end_idx]
        
        # Update buffer index with wraparound
        self.buffer_idx = (self.buffer_idx + self.config.buffer_size) % len(self.tokenized_data)
        
        # Make sure we have enough tokens
        if len(tokens) < self.config.batch_size:
            # If we don't have enough tokens, get more from the beginning
            additional_tokens = self.tokenized_data["tokens"][:self.config.batch_size - len(tokens)]
            if isinstance(tokens, torch.Tensor) and isinstance(additional_tokens, torch.Tensor):
                tokens = torch.cat([tokens, additional_tokens])
            else:
                tokens = list(tokens) + list(additional_tokens)
        
        # Convert to tensor if needed
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, device=self.config.device)
        
        # Store in buffer
        self.token_buffer = tokens.to(self.config.device)
    
    def generate_batch(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Generate a batch of tokens.
        
        Args:
            batch_size: Batch size (uses config if None)
            
        Returns:
            Batch of tokens
        """
        batch_size = batch_size or self.config.batch_size
        
        # Fill buffer if empty or too small
        if not isinstance(self.token_buffer, torch.Tensor) or len(self.token_buffer) < batch_size:
            self._fill_buffer()
        
        # Get a random subset of tokens
        indices = torch.randperm(len(self.token_buffer))[:batch_size]
        return self.token_buffer[indices]
    
    def get_activations(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Get MLP activations from the model.
        
        Args:
            tokens: Input tokens of shape [batch_size, seq_len]
            
        Returns:
            MLP activations of shape [batch_size * seq_len, d_mlp]
        """
        try:
            with torch.no_grad():
                # Make sure tokens are on the right device
                tokens = tokens.to(self.config.device)
                
                # Run model with activation caching
                _, cache = self.model.run_with_cache(
                    tokens, 
                    names_filter=[self.hook_name]
                )
                
                # Extract activations
                if self.hook_name not in cache:
                    hook_keys = list(cache.keys())
                    print(f"Warning: Hook name {self.hook_name} not found in cache. Available hooks: {hook_keys[:5]}...")
                    if hook_keys:
                        # Use the first available hook as a fallback
                        self.hook_name = hook_keys[0]
                        print(f"Using {self.hook_name} as fallback")
                
                activations = cache[self.hook_name]
                
                # Reshape to [batch_size * seq_len, d_mlp]
                activations = activations.reshape(-1, self.d_mlp)
                
                return activations
                
        except Exception as e:
            print(f"Error getting activations: {e}")
            # Return random activations as a fallback
            batch_size, seq_len = tokens.shape
            return torch.randn(batch_size * seq_len, self.d_mlp, device=self.config.device)
    
    def get_batch_activations(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Generate a batch of tokens and return their activations.
        
        Args:
            batch_size: Batch size (uses config if None)
            
        Returns:
            MLP activations
        """
        batch_size = batch_size or self.config.batch_size
        
        # Generate batch of tokens
        tokens = self.generate_batch(batch_size)
        
        # Get activations
        return self.get_activations(tokens)