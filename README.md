# Federated Learning with Privacy Mechanisms

This repository implements a federated learning system with various privacy-preserving approaches:

1. **Baseline FL**: Standard federated learning with no privacy mechanisms
2. **FL with Homomorphic Encryption (HE)**: Uses encryption to secure model updates
3. **FL with Differential Privacy (DP)**: Adds noise to model updates for privacy
4. **FL with HE+DP**: Combines both approaches

## Latest Improvements

We've made several improvements to the codebase:

- **Fixed DP Implementation**: Enhanced noise addition and privacy accounting
- **Improved MIA Evaluation**: More sophisticated membership inference attacks with AUC and additional metrics
- **Non-IID Data Support**: Added Dirichlet-based non-IID data distribution with configurable heterogeneity
- **Enhanced Visualization**: Better result plots with more metrics
- **Type Compatibility**: Fixed buffer dtype mismatch between HE and tensor operations
- **Additional Metrics**: More comprehensive evaluation metrics

## Setup and Installation

### Using Docker (Recommended)

```bash
   docker build -t fl_research .
```

### Running Experiments

Run experiments using Docker:

```bash
docker run -it --rm -v $(pwd):/app fl_research python main.py [OPTIONS]
```

Available options:

```
--experiment {baseline,he,dp,hybrid,all}  # Which experiment to run
--num_clients INTEGER                    # Number of clients to use
--num_rounds INTEGER                     # Number of federated learning rounds
--iid                                    # Use IID data distribution (default)
--non_iid                                # Use non-IID data distribution
--alpha FLOAT                            # Dirichlet concentration parameter (lower = more heterogeneity)
--local_epochs INTEGER                   # Number of local training epochs per round
--batch_size INTEGER                     # Batch size for training

# DP parameters
--target_epsilon FLOAT                   # Target epsilon for differential privacy
--target_delta FLOAT                     # Target delta for differential privacy
--max_grad_norm FLOAT                    # Maximum gradient norm for DP
--noise_multiplier FLOAT                 # Fixed noise multiplier (overrides epsilon if provided)

# HE parameters
--poly_modulus_degree INTEGER            # Polynomial modulus degree for HE
--scale INTEGER                          # Scale for fixed-point encoding in HE
--quantize_bits INTEGER                  # Number of bits for quantization in HE

# MIA parameters
--mia_attack_type {threshold,advanced}   # Type of MIA attack to use
```

## Example Commands

### Run all experiments with default settings:

```bash
docker run -it --rm -v $(pwd):/app fl_research python main.py --experiment all
```

### Run DP experiment with non-IID data:

```bash
docker run -it --rm -v $(pwd):/app fl_research python main.py --experiment dp --non_iid --alpha 0.1
```

### Run HE experiment with stronger encryption:

```bash
docker run -it --rm -v $(pwd):/app fl_research python main.py --experiment he --poly_modulus_degree 8192
```

### Run hybrid experiment with custom DP settings:

```bash
docker run -it --rm -v $(pwd):/app fl_research python main.py --experiment hybrid --target_epsilon 0.5 --noise_multiplier 1.1
```

## Results and Evaluation

After running experiments, you'll find several output files:

- `results_summary.csv`: A summary of metrics for all experiments
- `results_comparison.png`: Comparative visualization of different approaches
- Individual result plots for each approach

## Implementation Details

### Data

- CIFAR-10 dataset
- Support for both IID and non-IID data distributions

### Models

- Simple CNN architecture optimized for CIFAR-10
- Smaller CNN for HE experiments

### Privacy Mechanisms

- **Differential Privacy**: Implemented using Opacus with gradient clipping and noise addition
- **Homomorphic Encryption**: Implemented using Pyfhel with CKKS scheme

### Privacy Evaluation

- Membership Inference Attacks (MIA) using threshold-based and learning-based approaches
- Metrics include accuracy, precision, recall, F1 score, and AUC

## Interpreting Results

- **Accuracy**: Higher is better, represents model utility
- **MIA Success Rate**: Lower is better, represents better privacy protection
- **Communication Overhead**: Lower is better, represents efficiency
- **Computation Time**: Lower is better, represents efficiency

## License

MIT License
