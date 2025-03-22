# VSAE - Variational Sparse Autoencoders

This repository implements Variational Sparse Autoencoders (VSAEs) for disentangling superposed features in neural networks, alongside traditional Sparse Autoencoders (SAEs) for comparison. GPT2-small is not functional yet.

## Project Overview

The goal of this project is to investigate improved methods for feature disentanglement in neural networks using variational approaches to sparse autoencoders. The implementation supports experimentation on:

- Toy models of superposition (based on the Anthropic "Toy Models of Superposition" paper)
- GELU-1L transformer model
- GPT-2 Small model

Three variational approaches are implemented:

1. **VSAE-Iso**: VSAE with isotropic Gaussian prior
2. **VSAE-Mix**: VSAE with Gaussian mixture prior (for correlated/anti-correlated features)
3. **VSAE-Multi**: VSAE with multivariate Gaussian prior (for general correlation structures)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/vsae.git
cd vsae
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training SAE/VSAE on Toy Models

Train a vanilla SAE on a toy model with correlated features:

```bash
python main.py --vanilla_sae --toy_model --n_features 100 --n_hidden 20 --n_correlated_pairs 2 \
  --n_anticorrelated_pairs 1 --dict_mult 8 --l1_coeff 3e-4 --steps 10000 \
  --lr 1e-4 --lr_decay cosine --use_tensorboard --save_model
```

Train a VSAE with Gaussian mixture prior on toy model:

```bash
python main.py --vsae_mix --toy_model --n_features 100 --n_hidden 20 --n_correlated_pairs 2 \
  --n_anticorrelated_pairs 1 --dict_mult 8 --kl_coeff 3e-4 --var_flag 0 \
  --steps 10000 --lr 1e-4 --lr_decay cosine --use_tensorboard --save_model
```

Train a VSAE with multivariate Gaussian prior on toy model:

```bash
python main.py --vsae_multi --toy_model --n_features 100 --n_hidden 20 --corr_rate 0.5 \
  --dict_mult 8 --kl_coeff 3e-4 --steps 10000 --lr 1e-4 --lr_decay cosine \
  --use_tensorboard --save_model
```

### Training on GELU-1L Model

Train a vanilla SAE on GELU-1L model:

```bash
python test_gelu_1l.py --dict_mult 8 --l1_coeff 3e-4 --batch_size 512 \
  --steps 10000 --lr 1e-4 --save_model
```

### Training on GPT-2 Small Model

Train a vanilla SAE on GPT-2 Small:

```bash
python test_gpt2_small.py --layer 0 --act_name post --dict_mult 8 --l1_coeff 3e-4 \
  --batch_size 128 --steps 10000 --lr 1e-4 --lr_decay cosine --save_model
```

Train a VSAE on GPT-2 Small:

```bash
python test_gpt2_small.py --use_vsae --var_flag 1 --layer 0 --act_name post \
  --dict_mult 8 --kl_coeff 3e-4 --batch_size 128 --steps 10000 \
  --lr 1e-4 --lr_decay cosine --save_model
```

### Analysis

Analyze a trained SAE/VSAE:

```bash
python analyze_gpt2_sae.py ./results/sae_gpt2small_l0_post_20250322-120000.pt \
  --output_dir ./analysis_results --n_samples 1000 --feature_sparsity \
  --reconstruction_quality --dictionary_similarity --neuron_activations --latent_space
```

## Project Structure

- `main.py` - Main script for training SAEs/VSAEs on different models
- `vanilla_sae.py` - Vanilla SAE implementation
- `vsae_iso.py` - VSAE with isotropic Gaussian prior
- `vsae_mix.py` - VSAE with Gaussian mixture prior
- `vsae_multi.py` - VSAE with multivariate Gaussian prior
- `toy_model.py` - Toy model implementation for superposition studies
- `gelu_1l_model.py` - Wrapper for GELU-1L transformer model
- `gpt2_model.py` - Wrapper for GPT-2 small model
- `analyze_gpt2_sae.py` - Analysis tools for trained models
- `tests.py` - Unit tests

## TensorBoard Visualization

Monitor training with TensorBoard:

```bash
tensorboard --logdir=./runs
```

## References

This implementation is made by:
- Yuxioa Li's "Refining SAE for Correlated Features Extraction" Colab
- Anthropic's ["Toy Models of Superposition"](https://arxiv.org/abs/2209.10652) paper and [Colab](https://colab.research.google.com/github/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb) 
- Anthropic's ["Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"](https://arxiv.org/abs/2402.14705) paper
- Neel Nanda's ["Open Source Replication & Commentary on Anthropic's Dictionary Learning"](Open Source Replication & Commentary on Anthropic's Dictionary Learning Paper) paper
  and [Colab](https://colab.research.google.com/drive/1u8larhpxy8w4mMsJiSBddNOzFGj7_RTn?usp=sharing)

