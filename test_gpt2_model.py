import unittest
import torch
import numpy as np
from gpt2_model import GPT2Model, GPT2Config


class TestGPT2Model(unittest.TestCase):
    """Test cases for the GPT-2 model wrapper"""

    def setUp(self):
        """Set up test fixtures"""
        # Use a tiny configuration for testing
        self.config = GPT2Config(
            model_name="gpt2",
            activation_name="post",
            layer_num=0,
            batch_size=4,  # Small batch size for testing
            context_length=8,  # Small context length for testing
            device="cpu"  # Always use CPU for testing
        )

    def test_initialization(self):
        """Test model initialization with minimal operations"""
        # Skip this test by default - uncomment to run a real initialization test
        # It's skipped because it would download and load the actual model
        self.skipTest("Skipping initialization test to avoid downloading model")
        
        # Initialize model
        model = GPT2Model(self.config)
        
        # Check if model attributes are set correctly
        self.assertEqual(model.config.model_name, "gpt2")
        self.assertEqual(model.config.layer_num, 0)
        self.assertEqual(model.hook_name, utils.get_act_name("post", 0))
        
        # Verify model parameters
        self.assertEqual(model.n_layers, 12)  # GPT-2 small has 12 layers
        self.assertEqual(model.d_mlp, 3072)   # GPT-2 small has 3072 MLP dimension

    def test_buffer_management(self):
        """Test the token buffer management"""
        # Create a minimal model with synthetic data for testing
        model = self._create_minimal_test_model()
        
        # Test generate_batch
        batch = model.generate_batch(2)
        self.assertEqual(batch.shape, (2, self.config.context_length))
        
        # Test buffer refill
        model.buffer_idx = len(model.tokenized_data) - 1  # Force buffer refill
        model.token_buffer = []  # Clear buffer
        batch = model.generate_batch(2)
        self.assertEqual(batch.shape, (2, self.config.context_length))

    def test_activation_extraction(self):
        """Test activation extraction from the model"""
        # Create a minimal model with synthetic data for testing
        model = self._create_minimal_test_model()
        
        # Override the get_activations method with a mocked version
        def mocked_get_activations(tokens):
            batch_size, seq_len = tokens.shape
            return torch.randn(batch_size * seq_len, 16)  # Use small d_mlp for testing
            
        model.get_activations = mocked_get_activations
        
        # Test get_batch_activations
        batch_size = 2
        activations = model.get_batch_activations(batch_size)
        expected_shape = (batch_size * self.config.context_length, 16)
        self.assertEqual(activations.shape, expected_shape)

    def _create_minimal_test_model(self):
        """Create a minimal model for testing without loading the actual model"""
        model = GPT2Model.__new__(GPT2Model)
        model.config = self.config
        model.hook_name = f"blocks.0.{self.config.activation_name}"
        
        # Minimal mock of tokenized data
        class MockDataset:
            def __init__(self, length=100):
                self.length = length
                self.tokens = torch.randint(0, 1000, (length, 8))
                
            def __len__(self):
                return self.length
                
            def __getitem__(self, idx):
                if isinstance(idx, slice):
                    return {"tokens": self.tokens[idx]}
                return {"tokens": self.tokens[idx:idx+1]}
                
            @property
            def column_names(self):
                return ["tokens"]
                
            def shuffle(self, seed=None):
                return self
        
        model.tokenized_data = MockDataset()
        model.token_buffer = []
        model.buffer_idx = 0
        model.d_mlp = 16  # Small value for testing
        model.n_layers = 2
        model.d_model = 32
        
        return model


if __name__ == "__main__":
    unittest.main()
