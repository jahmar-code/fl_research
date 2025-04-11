# Federated Learning with Privacy Mechanisms

This repository implements a federated learning system with various privacy-preserving approaches:

1. **Baseline FL**: Standard federated learning with no privacy mechanisms
2. **FL with Homomorphic Encryption (HE)**: Uses encryption to secure model updates
3. **FL with Differential Privacy (DP)**: Adds noise to model updates for privacy
4. **FL with Standard Hybrid (HE+DP)**: Applies both mechanisms simultaneously
5. **FL with Sequential Hybrid (HE+DP)**: Our novel approach applying DP during training and HE during aggregation

## Latest Improvements

We've made several improvements to the codebase:

- **Sequential Hybrid Implementation**: Added novel sequential application of DP and HE that improves stability and heterogeneity resilience
- **Comprehensive Heterogeneity Analysis**: Enhanced support for non-IID data distributions with detailed analysis under varying degrees of heterogeneity
- **Improved Visualization**: Added extensive plots to visualize privacy-utility-heterogeneity tradeoffs and approach comparisons
- **Enhanced MIA Evaluation**: More sophisticated membership inference attacks with AUC and additional metrics
- **Resource Measurement**: Added precise communication and computation overhead tracking
- **Type Compatibility**: Fixed buffer dtype mismatch between HE and tensor operations

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
--experiment {baseline,he,dp,standard_hybrid,sequential_hybrid,all}  # Which experiment to run
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
docker run -it --rm -v $(pwd):/app fl_research python main.py --experiment dp --non_iid --alpha 0.5
```

### Run HE experiment with stronger encryption:

```bash
docker run -it --rm -v $(pwd):/app fl_research python main.py --experiment he --poly_modulus_degree 8192
```

### Run standard hybrid experiment with custom DP settings:

```bash
docker run -it --rm -v $(pwd):/app fl_research python main.py --experiment standard_hybrid --target_epsilon 0.5 --noise_multiplier 1.1
```

### Run sequential hybrid experiment:

```bash
docker run -it --rm -v $(pwd):/app fl_research python main.py --experiment sequential_hybrid --noise_multiplier 0.3 --quantize_bits 24
```

## Results and Evaluation

After running experiments, you'll find several output files:

- `results_summary.csv`: A summary of metrics for all experiments
- `analysis_results/`: Directory containing all visualizations
- Individual result plots for each approach

### Visualization

To generate all visualizations from experimental results:

```bash
docker run -it --rm -v $(pwd):/app fl_research python /app/generate_plots.py
```

## Implementation Details

### Data

- CIFAR-10 dataset
- Support for both IID and non-IID data distributions (Dirichlet allocation with configurable α)

### Models

- SimpleCNN: Three convolutional layers, used for Baseline and DP
- SmallCNN: Two convolutional layers, used for HE and Hybrid approaches

### Privacy Mechanisms

- **Differential Privacy**: Implemented using Opacus with gradient clipping and noise addition
- **Homomorphic Encryption**: Implemented using Pyfhel with CKKS scheme
- **Standard Hybrid**: Applies both DP and HE simultaneously
- **Sequential Hybrid**: Applies DP during training and HE during aggregation

### Privacy Evaluation

- Membership Inference Attacks (MIA) using threshold-based and learning-based approaches
- Metrics include accuracy, precision, recall, F1 score, and AUC

## Interpreting Results

- **Accuracy**: Higher is better, represents model utility
- **MIA Success Rate (AUC)**: Lower is better (closer to 0.5), represents better privacy protection
- **Communication Overhead**: Lower is better, represents efficiency
- **Computation Time**: Lower is better, represents efficiency
- **Accuracy Retention**: Higher is better, represents resilience to data heterogeneity

## License

MIT License
