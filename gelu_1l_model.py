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
        
        # Initialize token buffer as None and buffer index
        self.token_buffer = None
        self.buffer_idx = 0
        
        # Initialize the tokenized data (will be lazy loaded)
        self.tokenized_data = None
    
    def _lazy_load_dataset(self):
        """
        Lazy load the dataset only when actually needed.
        """
        if self.tokenized_data is None:
            self.tokenized_data = self._load_dataset()
    
    def _load_dataset(self):
        """
        Load and tokenize dataset.
        
        Returns:
            Tokenized dataset
        
        Raises:
            Exception: If the dataset cannot be loaded or tokenized
        """
        # Import datasets here to avoid dependency if not needed
        from datasets import load_dataset
        
        # Load the dataset
        print(f"Loading dataset: {self.config.dataset_name}")
        data = load_dataset(self.config.dataset_name, split="train")
        
        # Print some info about the dataset for debugging
        print(f"Dataset info: {data}")
        if hasattr(data, 'column_names'):
            print(f"Dataset columns: {data.column_names}")
        
        # Handle different dataset structures
        if 'tokens' in data.column_names:
            print("Dataset already has 'tokens' column, skipping tokenization")
            tokenized_data = data
        elif 'text' in data.column_names:
            print("Dataset has 'text' column, performing tokenization")
            tokenized_data = utils.tokenize_and_concatenate(
                data, 
                self.model.tokenizer, 
                max_length=self.config.context_length
            )
        else:
            print(f"Dataset doesn't have recognized columns. Available columns: {data.column_names}")
            print("Attempting to locate a text-like column...")
            
            # Try to find a suitable column for tokenization
            text_column = None
            for col in data.column_names:
                if any(keyword in col.lower() for keyword in ['text', 'content', 'body', 'document']):
                    text_column = col
                    break
            
            if text_column:
                print(f"Using column '{text_column}' for tokenization")
                # Create a mapping for tokenize_and_concatenate
                data = data.rename_column(text_column, 'text')
                tokenized_data = utils.tokenize_and_concatenate(
                    data, 
                    self.model.tokenizer, 
                    max_length=self.config.context_length
                )
            else:
                raise ValueError(f"No suitable text column found in dataset. Available columns: {data.column_names}")
        
        # Shuffle the dataset
        tokenized_data = tokenized_data.shuffle(seed=self.config.seed)
        
        print(f"Loaded dataset with {len(tokenized_data)} sequences")
        print(f"Final dataset columns: {tokenized_data.column_names}")
        
        # Verify we have tokens
        if "tokens" not in tokenized_data.column_names:
            print(f"Warning: 'tokens' not found in final dataset. Available columns: {tokenized_data.column_names}")
            
            # If we don't have tokens, but we have input_ids or similar, map it to tokens
            for col in tokenized_data.column_names:
                if any(keyword in col.lower() for keyword in ['token', 'input_id', 'encoding']):
                    print(f"Using column '{col}' as tokens")
                    # Add a new column 'tokens' that's a copy of the found column
                    tokenized_data = tokenized_data.add_column('tokens', tokenized_data[col])
                    break
        
        if "tokens" not in tokenized_data.column_names or len(tokenized_data["tokens"]) == 0:
            raise ValueError("Dataset doesn't contain tokens or is empty")
                
        return tokenized_data
    
    def _fill_buffer(self):
        """
        Fill the token buffer with new tokens from the dataset.
        
        Raises:
            Exception: If the buffer cannot be filled
        """
        # Ensure dataset is loaded
        self._lazy_load_dataset()
        
        # Get a batch of tokens
        end_idx = min(self.buffer_idx + self.config.buffer_size, len(self.tokenized_data))
        
        # Debug info
        print(f"Filling buffer from index {self.buffer_idx} to {end_idx}, dataset size: {len(self.tokenized_data)}")
        
        # Get batch - handle different dataset types
        if isinstance(self.tokenized_data, dict):
            # Dictionary-like dataset
            tokens = self.tokenized_data["tokens"][self.buffer_idx:end_idx]
        else:
            # HuggingFace dataset or similar
            tokens_batch = self.tokenized_data[self.buffer_idx:end_idx]
            
            # Check if we're dealing with a batch of dictionaries or a single dictionary
            if isinstance(tokens_batch, dict):
                tokens = tokens_batch["tokens"]
            else:
                # Try to handle as a Dataset slice
                tokens = tokens_batch["tokens"]
        
        # Update buffer index with wraparound
        self.buffer_idx = (self.buffer_idx + self.config.buffer_size) % len(self.tokenized_data)
        
        # Make sure we have enough tokens
        if len(tokens) < self.config.batch_size:
            # If we don't have enough tokens, get more from the beginning
            print(f"Not enough tokens ({len(tokens)}), getting more from the beginning.")
            
            # Get additional tokens from the beginning
            if isinstance(self.tokenized_data, dict):
                additional_tokens = self.tokenized_data["tokens"][:self.config.batch_size - len(tokens)]
            else:
                additional_batch = self.tokenized_data[:self.config.batch_size - len(tokens)]
                if isinstance(additional_batch, dict):
                    additional_tokens = additional_batch["tokens"]
                else:
                    additional_tokens = additional_batch["tokens"]
            
            # Handle different types (list vs tensor)
            if isinstance(tokens, torch.Tensor) and isinstance(additional_tokens, torch.Tensor):
                tokens = torch.cat([tokens, additional_tokens])
            else:
                # Try to convert list to tensor if possible
                tokens = torch.cat([torch.tensor(tokens), torch.tensor(additional_tokens)])
        
        # Convert to tensor if needed
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, device=self.config.device)
        
        # Store in buffer
        self.token_buffer = tokens.to(self.config.device)
        print(f"Buffer filled with {len(self.token_buffer)} tokens with shape {self.token_buffer.shape}")
    
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
        if self.token_buffer is None or len(self.token_buffer) < batch_size:
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