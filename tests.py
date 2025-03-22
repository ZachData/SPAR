import torch
import unittest
import numpy as np
from typing import Tuple, Dict


from toy_model import ToyModel, ToyModelConfig, linear_lr, constant_lr, cosine_decay_lr
from gpt2_model import GPT2Model, GPT2Config

from vanilla_sae import VanillaSAE, SAEConfig
from vsae_iso import VSAEIsoGaussian, VSAEIsoConfig
from vsae_mix import VSAEMixGaussian, VSAEMixConfig
from vsae_multi import VSAEMultiGaussian, VSAEMultiConfig


class TestToyModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.config = ToyModelConfig(
            n_features=5,
            n_hidden=2,
            n_instances=3,
            batch_size=32,
            steps=10,
            device="cpu"  # Use CPU for testing
        )
        self.model = ToyModel(config=self.config)
        
    def test_initialization(self):
        """Test if the model initializes properly"""
        # Check if weights and biases have correct shape
        self.assertEqual(self.model.W.shape, 
                         (self.config.n_instances, self.config.n_hidden, self.config.n_features))
        self.assertEqual(self.model.b_final.shape, 
                         (self.config.n_instances, self.config.n_features))
        
    def test_forward_pass(self):
        """Test if forward pass works correctly"""
        batch_size = 16
        batch = torch.rand((batch_size, self.config.n_instances, self.config.n_features))
        output = self.model(batch)
        
        # Check output shape
        self.assertEqual(output.shape, batch.shape)
        
        # Check that output is non-negative (due to ReLU)
        self.assertTrue((output >= 0).all())
        
    def test_batch_generation(self):
        """Test if batch generation works"""
        batch_size = 16
        
        # Test simple batch
        batch = self.model.generate_batch(batch_size)
        self.assertEqual(batch.shape, 
                         (batch_size, self.config.n_instances, self.config.n_features))
        
        # Values should be in [0, 1]
        self.assertTrue((batch >= 0).all() and (batch <= 1).all())
        
    def test_loss_calculation(self):
        """Test if loss calculation works"""
        batch_size = 16
        batch = torch.rand((batch_size, self.config.n_instances, self.config.n_features))
        output = self.model(batch)
        loss = self.model.calculate_loss(output, batch)
        
        # Check that loss is a scalar
        self.assertEqual(loss.dim(), 0)
        # Check that loss is non-negative (squared error)
        self.assertTrue(loss >= 0)
        
    def test_correlated_features(self):
        """Test generation of correlated features"""
        # Create a toy model config with 2 correlated feature pairs
        config = ToyModelConfig(
            n_features=8,
            n_hidden=2,
            n_instances=2,
            n_correlated_pairs=2,
            batch_size=32,
            device="cpu"
        )
    
        # Initialize feature_probability as a proper 2D tensor
        feature_probability = torch.full((config.n_instances, 1), 0.5, device="cpu")
        
        # Create the model with explicit feature_probability
        model = ToyModel(
            config=config,
            feature_probability=feature_probability,
        )
        
        # Generate batch with correlated features
        batch_size = 100
        batch = model.generate_batch(batch_size)
        
        # Check shape
        self.assertEqual(batch.shape, (batch_size, config.n_instances, config.n_features))
        
        # For correlated pairs, they should either both be active or both inactive
        # Check first correlated pair (features 0 and 1)
        for instance in range(config.n_instances):
            feature_0_active = batch[:, instance, 0] > 0
            feature_1_active = batch[:, instance, 1] > 0
            
            # Calculate how often both features match in their active/inactive state
            matching_states = (feature_0_active == feature_1_active).float().mean()
            
            # They should match at least 95% of the time (allowing for minor numerical issues)
            self.assertGreater(matching_states.item(), 0.95)
            
            # If one is active, the other should be active too
            both_active = (feature_0_active & feature_1_active).float().sum()
            either_active = (feature_0_active | feature_1_active).float().sum()
            
            # If either is active, both should be active (nearly always)
            if either_active > 0:
                agreement_rate = both_active / either_active
                self.assertGreater(agreement_rate.item(), 0.95)
        
    def test_anticorrelated_features(self):
        """Test generation of anticorrelated features"""
        # Create a toy model config with 2 anticorrelated feature pairs
        config = ToyModelConfig(
            n_features=8,
            n_hidden=2,
            n_instances=2,
            n_anticorrelated_pairs=2,
            batch_size=32,
            device="cpu"
        )
        
        # Initialize feature_probability as a proper 2D tensor
        feature_probability = torch.full((config.n_instances, 1), 0.5, device="cpu")
        
        # Create the model with explicit feature_probability
        model = ToyModel(
            config=config,
            feature_probability=feature_probability
        )
                
        # Generate batch with anticorrelated features
        batch_size = 100
        batch = model.generate_batch(batch_size)
        
        # Check shape
        self.assertEqual(batch.shape, (batch_size, config.n_instances, config.n_features))
        
        # For anticorrelated pairs, they should not both be active at the same time
        # Check first anticorrelated pair (features 0 and 1)
        for instance in range(config.n_instances):
            feature_0_active = batch[:, instance, 0] > 0
            feature_1_active = batch[:, instance, 1] > 0
            
            # Calculate how often both features are active simultaneously
            both_active = (feature_0_active & feature_1_active).float().mean()
            
            # Should almost never be both active
            self.assertLess(both_active.item(), 0.05)
            
            # Check if they are mutually exclusive when the pair is present
            either_active = (feature_0_active | feature_1_active).float().sum()
            if either_active > 0:
                mutual_exclusion = 1.0 - ((feature_0_active & feature_1_active).float().sum() / either_active)
                self.assertGreater(mutual_exclusion.item(), 0.95)

        
    def test_matrix_correlation(self):
        """Test generation with correlation matrix"""
        n_features = 5
        corr_matrix = torch.eye(n_features, dtype=torch.float32)
        corr_matrix[0, 1] = corr_matrix[1, 0] = 0.8  # Strong correlation between features 0 and 1
        
        config = ToyModelConfig(
            n_features=n_features,
            n_hidden=2,
            n_instances=1,
            corr_matrix=corr_matrix,
            batch_size=32,
            device="cpu"
        )
        feature_probability = torch.ones((config.n_instances, 1), device="cpu")
        model = ToyModel(config=config, feature_probability=feature_probability)
        
        # Generate batch with correlated features
        batch_size = 100
        batch = model.generate_batch(batch_size)
        
        # Check shape
        self.assertEqual(batch.shape, (batch_size, config.n_instances, config.n_features))

    def test_optimize(self):
        """Test if optimization works without errors"""
        self.model.optimize(batch_size=8, steps=5)  # Use small values for testing

class TestGPT2Integration(unittest.TestCase):
    """Integration tests between GPT-2 Model and SAE/VSAE"""
    
    def setUp(self):
        """
        Set up test fixtures by creating a mock GPT2Model
        that doesn't actually load the real model
        """
        self.config = GPT2Config(
            model_name="gpt2",
            activation_name="post",
            layer_num=0,
            batch_size=4,
            context_length=8,
            device="cpu"
        )
        
        # Create a minimal mock model
        self.gpt2_model = self._create_minimal_mock_model()
        
        # Create SAE configuration
        self.sae_config = SAEConfig(
            d_input=16,  # Mock d_mlp size
            d_hidden=32,  # Small hidden dimension for tests
            dict_mult=2,
            l1_coeff=0.001,
            device="cpu"
        )
        
        # Create VSAE configuration
        self.vsae_config = VSAEIsoConfig(
            d_input=16,  # Mock d_mlp size
            d_hidden=32,  # Small hidden dimension for tests
            dict_mult=2,
            kl_coeff=0.001,
            var_flag=0,
            device="cpu"
        )
    
    def _create_minimal_mock_model(self):
        """Create a minimal mock GPT2Model for testing"""
        model = GPT2Model.__new__(GPT2Model)
        model.config = self.config
        model.hook_name = f"blocks.0.{self.config.activation_name}"
        model.d_mlp = 16  # Small value for testing
        model.n_layers = 2
        model.d_model = 32
        
        # Create a fake get_batch_activations method that returns random data
        def mock_get_batch_activations(batch_size=None):
            batch_size = batch_size or self.config.batch_size
            return torch.rand(batch_size * self.config.context_length, model.d_mlp)
        
        model.get_batch_activations = mock_get_batch_activations
        return model
    
    def test_sae_with_gpt2(self):
        """Test training SAE on GPT-2 activations"""
        # Create SAE
        sae = VanillaSAE(self.sae_config)
        
        # Get batch of activations
        batch_size = 4
        activations = self.gpt2_model.get_batch_activations(batch_size)
        
        # Check activation shape
        expected_shape = (batch_size * self.config.context_length, self.gpt2_model.d_mlp)
        self.assertEqual(activations.shape, expected_shape)
        
        # Create optimizer
        optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)
        
        # Train for a few steps
        for step in range(3):
            metrics = sae.train_step(activations, optimizer, step)
            
            # Check that metrics are produced
            self.assertIn("loss", metrics)
            self.assertIn("l2_loss", metrics)
            self.assertIn("l1_loss", metrics)
            
        # Test encoding and decoding
        with torch.no_grad():
            # Encode
            acts = sae.encode(activations)
            self.assertEqual(acts.shape, (activations.shape[0], self.sae_config.d_hidden))
            
            # Decode
            reconstructed = sae.decode(acts)
            self.assertEqual(reconstructed.shape, activations.shape)
    
    def test_vsae_with_gpt2(self):
        """Test training VSAE on GPT-2 activations"""
        # Create VSAE
        vsae = VSAEIsoGaussian(self.vsae_config)
        
        # Get batch of activations
        batch_size = 4
        activations = self.gpt2_model.get_batch_activations(batch_size)
        
        # Check activation shape
        expected_shape = (batch_size * self.config.context_length, self.gpt2_model.d_mlp)
        self.assertEqual(activations.shape, expected_shape)
        
        # Create optimizer
        optimizer = torch.optim.Adam(vsae.parameters(), lr=0.001)
        
        # Train for a few steps
        for step in range(3):
            metrics = vsae.train_step(activations, optimizer, step)
            
            # Check that metrics are produced
            self.assertIn("loss", metrics)
            self.assertIn("l2_loss", metrics)
            self.assertIn("kl_loss", metrics)
            
        # Test encoding and decoding
        with torch.no_grad():
            # Deterministic encode
            mu = vsae.encode(activations, deterministic=True)
            self.assertEqual(mu.shape, (activations.shape[0], self.vsae_config.d_hidden))
            
            # Stochastic encode
            z = vsae.encode(activations, deterministic=False)
            self.assertEqual(z.shape, (activations.shape[0], self.vsae_config.d_hidden))
            
            # Decode
            reconstructed = vsae.decode(z)
            self.assertEqual(reconstructed.shape, activations.shape)

class TestVanillaSAE(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.config = SAEConfig(
            d_input=10,
            d_hidden=30,
            dict_mult=3,
            l1_coeff=0.001,
            device="cpu"  # Use CPU for testing
        )
        self.sae = VanillaSAE(config=self.config)
        
    def test_initialization(self):
        """Test if the SAE initializes properly"""
        # Check if weights and biases have correct shape
        self.assertEqual(self.sae.W_enc.shape, (self.config.d_input, self.config.d_hidden))
        self.assertEqual(self.sae.W_dec.shape, (self.config.d_hidden, self.config.d_input))
        self.assertEqual(self.sae.b_enc.shape, (self.config.d_hidden,))
        self.assertEqual(self.sae.b_dec.shape, (self.config.d_input,))
        
    def test_normalize_decoder(self):
        """Test if decoder normalization works"""
        self.sae.normalize_decoder()
        
        # Check that decoder weights have unit norm
        norms = self.sae.W_dec.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))
        
    def test_forward_pass(self):
        """Test if forward pass works correctly"""
        batch_size = 8
        x = torch.rand((batch_size, self.config.d_input))
        outputs = self.sae(x)
        
        # Check that all expected outputs are present
        required_keys = ["loss", "l2_loss", "l1_loss", "acts", "x_recon", "frac_active"]
        for key in required_keys:
            self.assertIn(key, outputs)
            
        # Check shapes
        self.assertEqual(outputs["acts"].shape, (batch_size, self.config.d_hidden))
        self.assertEqual(outputs["x_recon"].shape, (batch_size, self.config.d_input))
        
        # Check that activations are non-negative (due to ReLU)
        self.assertTrue((outputs["acts"] >= 0).all())
        
    def test_encode_decode(self):
        """Test encode and decode methods"""
        batch_size = 8
        x = torch.rand((batch_size, self.config.d_input))
        
        # Test encode
        acts = self.sae.encode(x)
        self.assertEqual(acts.shape, (batch_size, self.config.d_hidden))
        self.assertTrue((acts >= 0).all())  # ReLU activations
        
        # Test decode
        x_recon = self.sae.decode(acts)
        self.assertEqual(x_recon.shape, (batch_size, self.config.d_input))
        
    def test_dead_neuron_detection(self):
        """Test dead neuron detection"""
        # Create fake activation history with some dead neurons
        batch_size = 8
        x = torch.zeros((batch_size, self.config.d_input))
        
        # First pass to initialize
        _ = self.sae(x)
        
        # Create more history with specific pattern
        for i in range(10):
            acts = torch.zeros((batch_size, self.config.d_hidden))
            # Make first half of neurons active sometimes
            if i % 2 == 0:
                acts[:, :self.config.d_hidden//2] = 1.0
            self.sae.activation_history.append((acts > 0).float().mean(0))
        
        # Detect dead neurons
        dead_neurons = self.sae.detect_dead_neurons()
        
        # Second half of neurons should be dead
        self.assertTrue((dead_neurons[self.config.d_hidden//2:]).all())
        self.assertFalse((dead_neurons[:self.config.d_hidden//2]).any())
        
    def test_train_step(self):
        """Test if training step works"""
        batch_size = 8
        x = torch.rand((batch_size, self.config.d_input))
        optimizer = torch.optim.Adam(self.sae.parameters(), lr=0.001)
        
        # Run a training step
        metrics = self.sae.train_step(x, optimizer, step=0)
        
        # Check that metrics are returned
        self.assertIn("loss", metrics)
        self.assertIn("l2_loss", metrics)
        self.assertIn("l1_loss", metrics)
        
    def test_resample_dead_neurons(self):
        """Test resampling of dead neurons"""
        # Force some neurons to be dead
        self.sae.activation_history = []
        
        batch_size = 16
        x = torch.rand((batch_size, self.config.d_input), device=self.sae.W_enc.device)
        
        # Run forward pass to initialize
        _ = self.sae(x)
        
        # Clear the activation history
        self.sae.activation_history = []
        
        # Explicitly create activation history with dead neurons
        # Make sure we have enough entries to satisfy any window requirements
        window_size = max(5, self.sae.config.dead_neuron_window)
        for i in range(window_size):
            frac_active = torch.ones(self.config.d_hidden, device=self.sae.W_enc.device)
            # Mark last 5 neurons as explicitly dead
            frac_active[-5:] = 0.0
            self.sae.activation_history.append(frac_active)
        
        # Verify that dead neurons are detected correctly
        dead_mask = self.sae.detect_dead_neurons()
        self.assertEqual(dead_mask[-5:].sum().item(), 5)
        
        # Resample the dead neurons
        dead_indices, n_dead = self.sae.resample_dead_neurons(x)
        
        # Check that 5 neurons were detected as dead
        self.assertEqual(n_dead, 5)
        self.assertEqual(len(dead_indices), 5)
        
        # The dead neurons should be the last 5
        for i in range(5):
            self.assertIn(self.config.d_hidden - i - 1, dead_indices)

class TestVSAEMix(unittest.TestCase):
    """Test cases for the Gaussian Mixture VSAE implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = VSAEMixConfig(
            d_input=10,
            d_hidden=30,
            dict_mult=3,
            kl_coeff=0.001,
            n_correlated_pairs=2,
            n_anticorrelated_pairs=2,
            device="cpu"  # Use CPU for testing
        )
        self.vsae = VSAEMixGaussian(config=self.config)
    
    def test_initialization(self):
        """Test if the VSAE initializes properly"""
        # Check if weights and biases have correct shape
        self.assertEqual(self.vsae.W_enc.shape, (self.config.d_input, self.config.d_hidden))
        self.assertEqual(self.vsae.W_dec.shape, (self.config.d_hidden, self.config.d_input))
        self.assertEqual(self.vsae.b_enc.shape, (self.config.d_hidden,))
        self.assertEqual(self.vsae.b_dec.shape, (self.config.d_input,))
        
        # Check prior means initialization
        self.assertEqual(self.vsae.prior_means.shape, (self.config.d_hidden,))
        
        # Check that correlated pairs have the same sign
        for i in range(self.config.n_correlated_pairs):
            self.assertEqual(self.vsae.prior_means[2*i].sign(), self.vsae.prior_means[2*i + 1].sign())
        
        # Check that anticorrelated pairs have opposite signs
        offset = 2 * self.config.n_correlated_pairs
        for i in range(self.config.n_anticorrelated_pairs):
            self.assertEqual(self.vsae.prior_means[offset + 2*i].sign(), -self.vsae.prior_means[offset + 2*i + 1].sign())
        
        # Check that remaining features have zero mean
        used_neurons = 2 * (self.config.n_correlated_pairs + self.config.n_anticorrelated_pairs)
        self.assertTrue((self.vsae.prior_means[used_neurons:] == 0).all())
    
    def test_forward_pass(self):
        """Test if forward pass works correctly"""
        batch_size = 8
        x = torch.rand((batch_size, self.config.d_input))
        outputs = self.vsae(x)
        
        # Check that all expected outputs are present
        required_keys = ["loss", "l2_loss", "kl_loss", "mu", "log_var", "z", "x_recon", "frac_active"]
        for key in required_keys:
            self.assertIn(key, outputs)
            
        # Check shapes
        self.assertEqual(outputs["mu"].shape, (batch_size, self.config.d_hidden))
        self.assertEqual(outputs["log_var"].shape, (batch_size, self.config.d_hidden))
        self.assertEqual(outputs["z"].shape, (batch_size, self.config.d_hidden))
        self.assertEqual(outputs["x_recon"].shape, (batch_size, self.config.d_input))
        
        # Check that activations are non-negative (due to ReLU)
        self.assertTrue((outputs["mu"] >= 0).all())
    
    def test_kl_divergence(self):
        """Test KL divergence calculation with mixture prior"""
        batch_size = 8
        
        # Create some test means and log variances
        mu = torch.zeros((batch_size, self.config.d_hidden))
        log_var = torch.zeros((batch_size, self.config.d_hidden))
        
        # Compute KL divergence
        kl = self.vsae.compute_kl_divergence(mu, log_var)
        
        # The KL divergence should be non-negative
        self.assertTrue(kl >= 0)
        
        # For zero mean and unit variance, the KL divergence should be:
        # 0.5 * sum((prior_means)^2) since we're comparing N(0,1) with N(prior_means, 1)
        expected_kl = 0.5 * (self.vsae.prior_means ** 2).sum()
        self.assertTrue(torch.isclose(kl, expected_kl, rtol=1e-4))
    
    def test_reparameterize(self):
        """Test reparameterization trick"""
        batch_size = 8
        mu = torch.ones((batch_size, self.config.d_hidden))
        log_var = torch.zeros((batch_size, self.config.d_hidden))  # log(1) = 0
        
        # With fixed random seed, outputs should be deterministic
        torch.manual_seed(42)
        z1 = self.vsae.reparameterize(mu, log_var)
        
        torch.manual_seed(42)
        z2 = self.vsae.reparameterize(mu, log_var)
        
        # Check that z matches with same seed
        self.assertTrue(torch.allclose(z1, z2))
        
        # With unit variance, z should be mu + standard normal
        torch.manual_seed(42)
        epsilon = torch.randn_like(mu)
        expected_z = mu + epsilon
        
        torch.manual_seed(42)
        z = self.vsae.reparameterize(mu, log_var)
        
        self.assertTrue(torch.allclose(z, expected_z))
    
    def test_encode_decode(self):
        """Test encode and decode methods"""
        batch_size = 8
        x = torch.rand((batch_size, self.config.d_input))
        
        # Test deterministic encode
        mu = self.vsae.encode(x, deterministic=True)
        self.assertEqual(mu.shape, (batch_size, self.config.d_hidden))
        self.assertTrue((mu >= 0).all())  # ReLU activations
        
        # Test stochastic encode
        z = self.vsae.encode(x, deterministic=False)
        self.assertEqual(z.shape, (batch_size, self.config.d_hidden))
        
        # Test decode
        x_recon = self.vsae.decode(mu)
        self.assertEqual(x_recon.shape, (batch_size, self.config.d_input))
    
    def test_train_step(self):
        """Test if training step works"""
        batch_size = 8
        x = torch.rand((batch_size, self.config.d_input))
        optimizer = torch.optim.Adam(self.vsae.parameters(), lr=0.001)
        
        # Run a training step
        metrics = self.vsae.train_step(x, optimizer, step=0)
        
        # Check that metrics are returned
        self.assertIn("loss", metrics)
        self.assertIn("l2_loss", metrics)
        self.assertIn("kl_loss", metrics)
        self.assertIn("sparsity", metrics)
    
    def test_integration_with_toy_model(self):
        """Test integration with toy model"""
        # Create a toy model with correlated and anti-correlated features
        toy_config = ToyModelConfig(
            n_features=8,
            n_hidden=5,
            n_instances=1,
            n_correlated_pairs=1,
            n_anticorrelated_pairs=1,
            batch_size=32,
            steps=10,
            device="cpu"
        )
        toy_model = ToyModel(config=toy_config)
        
        # Run a few optimization steps
        toy_model.optimize(steps=5)
        
        # Create VSAE with matching dimensions and correlation structure
        vsae_config = VSAEMixConfig(
            d_input=toy_config.n_hidden,
            d_hidden=20,
            n_correlated_pairs=toy_config.n_correlated_pairs,
            n_anticorrelated_pairs=toy_config.n_anticorrelated_pairs,
            device="cpu"
        )
        vsae = VSAEMixGaussian(config=vsae_config)
        
        # Generate toy model activations
        batch = toy_model.generate_batch(32)
        with torch.no_grad():
            hidden = torch.einsum("bif,ihf->bih", batch, toy_model.W)[:, 0, :]
        
        # Train VSAE for a few steps
        optimizer = torch.optim.Adam(vsae.parameters(), lr=0.001)
        for i in range(5):
            metrics = vsae.train_step(hidden, optimizer, step=i)
        
        # Test reconstruction
        with torch.no_grad():
            z = vsae.encode(hidden)
            hidden_recon = vsae.decode(z)
        
        # Shapes should match
        self.assertEqual(z.shape, (32, vsae_config.d_hidden))
        self.assertEqual(hidden_recon.shape, (32, toy_config.n_hidden))

class TestVSAEMulti(unittest.TestCase):
    """Test cases for the Multivariate Gaussian VSAE implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a custom correlation matrix
        d_hidden = 30
        corr_matrix = torch.eye(d_hidden)
        # Add correlation between first 10 features
        corr_matrix[:10, :10] = 0.5
        torch.diagonal(corr_matrix)[:] = 1.0
        
        self.config = VSAEMultiConfig(
            d_input=10,
            d_hidden=d_hidden,
            dict_mult=3,
            kl_coeff=0.001,
            corr_matrix=corr_matrix,
            device="cpu"  # Use CPU for testing
        )
        self.vsae = VSAEMultiGaussian(config=self.config)
    
    def test_initialization(self):
        """Test if the VSAE initializes properly"""
        # Check if weights and biases have correct shape
        self.assertEqual(self.vsae.W_enc.shape, (self.config.d_input, self.config.d_hidden))
        self.assertEqual(self.vsae.W_dec.shape, (self.config.d_hidden, self.config.d_input))
        self.assertEqual(self.vsae.b_enc.shape, (self.config.d_hidden,))
        self.assertEqual(self.vsae.b_dec.shape, (self.config.d_input,))
        
        # Check prior covariance initialization
        self.assertEqual(self.vsae.prior_covariance.shape, (self.config.d_hidden, self.config.d_hidden))
        
        # Check precision matrix
        self.assertEqual(self.vsae.prior_precision.shape, (self.config.d_hidden, self.config.d_hidden))
        
        # Check that precision * covariance ≈ identity
        identity_approx = self.vsae.prior_precision @ self.vsae.prior_covariance
        identity = torch.eye(self.config.d_hidden, device=self.config.device)
        self.assertTrue(torch.allclose(identity_approx, identity, atol=1e-5))
    
    def test_forward_pass(self):
        """Test if forward pass works correctly"""
        batch_size = 8
        x = torch.rand((batch_size, self.config.d_input))
        outputs = self.vsae(x)
        
        # Check that all expected outputs are present
        required_keys = ["loss", "l2_loss", "kl_loss", "mu", "log_var", "z", "x_recon", "frac_active"]
        for key in required_keys:
            self.assertIn(key, outputs)
            
        # Check shapes
        self.assertEqual(outputs["mu"].shape, (batch_size, self.config.d_hidden))
        self.assertEqual(outputs["log_var"].shape, (batch_size, self.config.d_hidden))
        self.assertEqual(outputs["z"].shape, (batch_size, self.config.d_hidden))
        self.assertEqual(outputs["x_recon"].shape, (batch_size, self.config.d_input))
        
        # Check that activations are non-negative (due to ReLU)
        self.assertTrue((outputs["mu"] >= 0).all())
    
    def test_kl_divergence(self):
        """Test KL divergence calculation with multivariate Gaussian prior"""
        batch_size = 8
        
        # Create some test means and log variances
        mu = torch.zeros((batch_size, self.config.d_hidden))
        log_var = torch.zeros((batch_size, self.config.d_hidden))
        
        # Compute KL divergence
        kl = self.vsae.compute_kl_divergence(mu, log_var)
        
        # The KL divergence should be non-negative
        self.assertTrue(kl >= 0)
    
    def test_reparameterize(self):
        """Test reparameterization trick"""
        batch_size = 8
        mu = torch.ones((batch_size, self.config.d_hidden))
        log_var = torch.zeros((batch_size, self.config.d_hidden))  # log(1) = 0
        
        # With fixed random seed, outputs should be deterministic
        torch.manual_seed(42)
        z1 = self.vsae.reparameterize(mu, log_var)
        
        torch.manual_seed(42)
        z2 = self.vsae.reparameterize(mu, log_var)
        
        # Check that z matches with same seed
        self.assertTrue(torch.allclose(z1, z2))
        
        # With unit variance, z should be mu + standard normal
        torch.manual_seed(42)
        epsilon = torch.randn_like(mu)
        expected_z = mu + epsilon
        
        torch.manual_seed(42)
        z = self.vsae.reparameterize(mu, log_var)
        
        self.assertTrue(torch.allclose(z, expected_z))
    
    def test_encode_decode(self):
        """Test encode and decode methods"""
        batch_size = 8
        x = torch.rand((batch_size, self.config.d_input))
        
        # Test deterministic encode
        mu = self.vsae.encode(x, deterministic=True)
        self.assertEqual(mu.shape, (batch_size, self.config.d_hidden))
        self.assertTrue((mu >= 0).all())  # ReLU activations
        
        # Test stochastic encode
        z = self.vsae.encode(x, deterministic=False)
        self.assertEqual(z.shape, (batch_size, self.config.d_hidden))
        
        # Test decode
        x_recon = self.vsae.decode(mu)
        self.assertEqual(x_recon.shape, (batch_size, self.config.d_input))
    
    def test_train_step(self):
        """Test if training step works"""
        batch_size = 8
        x = torch.rand((batch_size, self.config.d_input))
        optimizer = torch.optim.Adam(self.vsae.parameters(), lr=0.001)
        
        # Run a training step
        metrics = self.vsae.train_step(x, optimizer, step=0)
        
        # Check that metrics are returned
        self.assertIn("loss", metrics)
        self.assertIn("l2_loss", metrics)
        self.assertIn("kl_loss", metrics)
        self.assertIn("sparsity", metrics)
    
    def test_integration_with_toy_model(self):
        """Test integration with toy model"""
        # Create a toy model with correlation matrix
        toy_config = ToyModelConfig(
            n_features=8,
            n_hidden=5,
            n_instances=1,
            corr_rate=0.3,
            device="cpu"
        )
        toy_model = ToyModel(config=toy_config)
        
        # Run a few optimization steps
        toy_model.optimize(steps=5)
        
        # Create VSAE with matching dimensions and correlation structure
        vsae_config = VSAEMultiConfig(
            d_input=toy_config.n_hidden,
            d_hidden=20,
            corr_rate=toy_config.corr_rate,
            device="cpu"
        )
        vsae = VSAEMultiGaussian(config=vsae_config)
        
        # Generate toy model activations
        batch = toy_model.generate_batch(32)
        with torch.no_grad():
            hidden = torch.einsum("bif,ihf->bih", batch, toy_model.W)[:, 0, :]
        
        # Train VSAE for a few steps
        optimizer = torch.optim.Adam(vsae.parameters(), lr=0.001)
        for i in range(5):
            metrics = vsae.train_step(hidden, optimizer, step=i)
        
        # Test reconstruction
        with torch.no_grad():
            z = vsae.encode(hidden)
            hidden_recon = vsae.decode(z)
        
        # Shapes should match
        self.assertEqual(z.shape, (32, vsae_config.d_hidden))
        self.assertEqual(hidden_recon.shape, (32, toy_config.n_hidden))

class TestIntegration(unittest.TestCase):
    """Integration tests between Toy Model and SAE/VSAE"""

    def test_toy_model_to_sae(self):
        """Test training SAE on toy model activations"""
        # Create and train a small toy model
        toy_config = ToyModelConfig(
            n_features=8,
            n_hidden=3,
            n_instances=1,
            batch_size=32,
            steps=10,
            device="cpu"
        )
        toy_model = ToyModel(config=toy_config)
        toy_model.optimize(steps=10, batch_size=32)
        
        # Create SAE with matching dimensions
        sae_config = SAEConfig(
            d_input=toy_config.n_hidden,
            d_hidden=toy_config.n_features * 2,  # Overcomplete
            device="cpu"
        )
        sae = VanillaSAE(config=sae_config)
        
        # Generate activations from toy model
        batch = toy_model.generate_batch(32)
        with torch.no_grad():
            hidden = torch.einsum("bif,ihf->bih", batch, toy_model.W)
        
        # Feed to SAE
        optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)
        for i in range(5):  # Just a few steps
            metrics = sae.train_step(hidden.squeeze(1), optimizer, step=i)
            
        # Test encoding and reconstruction
        with torch.no_grad():
            acts = sae.encode(hidden.squeeze(1))
            hidden_recon = sae.decode(acts)
            
        # Check shapes
        self.assertEqual(acts.shape, (32, sae_config.d_hidden))
        self.assertEqual(hidden_recon.shape, (32, toy_config.n_hidden))
    
    def test_toy_model_to_vsae_iso(self):
        """Test training VSAE on toy model activations"""
        # Create and train a small toy model
        toy_config = ToyModelConfig(
            n_features=8,
            n_hidden=3,
            n_instances=1,
            batch_size=32,
            steps=10,
            device="cpu"
        )
        toy_model = ToyModel(config=toy_config)
        toy_model.optimize(steps=10, batch_size=32)
        
        # Create VSAE with matching dimensions
        vsae_config = VSAEIsoConfig(
            d_input=toy_config.n_hidden,
            d_hidden=toy_config.n_features * 2,  # Overcomplete
            var_flag=0,  # Fixed variance
            device="cpu"
        )
        vsae = VSAEIsoGaussian(config=vsae_config)
        
        # Generate activations from toy model
        batch = toy_model.generate_batch(32)
        with torch.no_grad():
            hidden = torch.einsum("bif,ihf->bih", batch, toy_model.W)
        
        # Feed to VSAE
        optimizer = torch.optim.Adam(vsae.parameters(), lr=0.001)
        for i in range(5):  # Just a few steps
            metrics = vsae.train_step(hidden.squeeze(1), optimizer, step=i)
            
        # Test encoding and reconstruction
        with torch.no_grad():
            # Test deterministic encoding
            mu = vsae.encode(hidden.squeeze(1), deterministic=True)
            self.assertEqual(mu.shape, (32, vsae_config.d_hidden))
            
            # Test stochastic encoding
            z = vsae.encode(hidden.squeeze(1), deterministic=False)
            self.assertEqual(z.shape, (32, vsae_config.d_hidden))
            
            # Test decoding
            hidden_recon = vsae.decode(z)
            self.assertEqual(hidden_recon.shape, (32, toy_config.n_hidden))

if __name__ == "__main__":
    unittest.main()
