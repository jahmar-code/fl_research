# Federated Learning with Privacy Mechanisms

This repository implements a federated learning system with various privacy-preserving approaches:

1. **Baseline FL**: Standard federated learning with no privacy mechanisms
2. **FL with Homomorphic Encryption (HE)**: Uses encryption to secure model updates
3. **FL with Differential Privacy (DP)**: Adds noise to model updates for privacy
4. **FL with Standard Hybrid (HE+DP)**: Applies both mechanisms simultaneously
5. **FL with Sequential Hybrid (HE+DP)**: Our novel approach applying DP during training and HE during aggregation

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

---

# Federated Learning: Theory and Implementation

### Theory

- federated learning (FL) is a machine learning approach where multiple devices (clients) collaboratively train a model while keeping their data localized, the process follows these steps:
  1.  initialization: a central server initializes a global model.
  2.  distribution: the server distributes the model to selected clients.
  3.  local training: each client trains the model on their local data.
  4.  aggregation: the server aggregates the model updates (not the data itself).
  5.  iteration: the process repeats for multiple rounds.
- the key advantages are:
  - data privacy (raw data never leaves devices).
  - reduced communication costs (only model updates are transmitted).
  - ability to leverage distributed data sources.
- the most common aggregation method is federated averaging (FedAvg), which computes a weighted average of client updates based on their dataset sizes.

### Implementation

- in the codebase, federated learning is implemented through:
  - FederatedClient class (federated/client.py): handles local training on client data.
  - FederatedServer class (federated/server.py): manages the global model and aggregation.
- the implementation uses PyTorch for model training and NumPy for efficient manipulation of model parameters.

---

# Homomorphic Encryption: Theory and Implementation

### Theory

- homomorphic encryption (HE): allows computations to be performed on encrypted data without decrypting it.
- the result, when decrypted, matches the result of performing the same operations on the plaintext.
- for federated learning, HE enables:
  - clients to encrypt model updates before sending them to the server.
  - the server to aggregate encrypted updates.
  - only clients have decryption keys, preventing the server from accessing raw updates.
- the main challenges of HE are:
  - computational overhead (encrypted operations are orders of magnitude slower).
  - limited operations (many schemes only support addition and multiplication).
  - noise growth (errors accumulate during computation, eventually corrupting results).

### Implementation

- the codebase uses CKKS scheme (the mathematical algorithm used for performing HE operations on data) via Pyfhel, which supports approximate arithmetic on real numbers:
  - HEFederatedClient class (privacy/he.py): extends the base client with encryption.
  - HEFederatedServer class (privacy/he.py): handles encrypted aggregation.
- key implementation details:
  - quantization: converting floating-point to integers.
  - chunking: breaking parameters into manageable pieces.
  - CKKS configuration: setting encryption parameters.

---

# Differential Privacy: Theory and Implementation

### Theory

- differential privacy (DP) provides mathematical guarantees about the privacy of individual data points in a dataset.
- it works by adding calibrated noise to the learning process, ensuring that the presence or absence of any single data point has a limited impact on the final model.
- key concepts:
  - ε (Epsilon): Privacy budget - lower values provide stronger privacy.
  - δ (Delta): Probability of privacy failure.
  - sensitivity: maximum impact a single record can have on the output.
  - noise addition: typically using gaussian or laplacian mechanisms.
- for federated learning, DP can be applied at different levels:
  - local DP: each client adds noise before sending updates.
  - central DP: the server adds noise during aggregation.

### Implementation

- the codebase uses Opacus, a PyTorch library for DP training, focusing on local DP:
  - DPFederatedClient class (privacy/dp.py): implements DP training.
- key implementation details:
  - gradient clipping: bounding gradient sensitivity.
  - noise addition: automatically handled by Opacus.
  - privacy accounting: tracking privacy budget.
  - memory management: handling computational overhead.

---

# Hybrid Approach: Theory and Implementation

### Theory

- the hybrid approach combine HE and DP to leverage their complementary strengths:
  - HE provides cryptographic security during aggregation.
  - DP provides formal privacy guarantees for the learning process.
- the goal is to achieve stronger privacy protection while mitigating the weaknesses of each approach.
  - HE has high computational overhead but doesn't degrade model utility.
  - DP adds noise that can reduce model accuracy but has lower computational cost.

### Implementation

- the codebase implements two hybrid approaches:
  - combined HE + DP (privacy/hybrid.py): applies both mechanisms in parallel.
  - sequential HE + DP (privacy/sequential_hybrid.py): applies mechanisms in sequence.
- the key difference is how the privacy mechanisms interact.
  - in the combined approach, DP is applied during training and HE during communication.
  - in the sequential approach, they're applied one after another in a specific order.

---

# Non-IID Data Distribution: Theory and Implementation

### Theory

- in federated learning, non-IID (non-Independent and Identically Distributed) data is an important challenge.
- in real-world scenarios, data distribution across clients is often heterogeneous, which can:
  - slow down convergence.
  - lead to biased global models.
  - create fairness issues.
- common types of non-IID dataL
  - label skew: different class distributions across clients.
  - feature skew: different feature distributions across clients.
  - quantity skew: different amounts of data across clients.

### Implementation

- the codebase uses a Dirichlet distribution to create non-IID data partitions.
- the alpha parameter controls heterogeneity:
  - lower alpha values (e.g. 0.1): create more heterogeneous distributions (more non-IID).
  - higher alpha values (e.g. 10): create more homogeneous distributions (closer to IID).
- this implementation allows the research to evaluate how different privacy mechanisms perform under varying levels of data heterogeneity.

---

# Membership Inference Attacks: Theory and Implementation

### Theory

- membership inference attacks (MIA) attempt to determine whether a specific data point was used to train a model.
- these attacks exploit the fact that models often behave differently on training data vs unseen data.
- MIAs serve as a practical way to evaluate the privacy protection of a model:
  - higher attack success rates indicate more privacy leakage.
  - lower success rates suggest better privacy protection.
- common attack approaches:
  - threshold-based: using simple statistics (e.g., loss values) to infer membership.
  - shadow model-based: training multiple "shadow models" to create training data for an attack model.
  - meta-classifier: building a machine learning model to distinguish members from non-members.

### Implementation

- the codebase implements two types of MIA in attacks/mia.py:
  - threshold attack: a simple approach using loss values.
  - advanced attack: a learning based approach.
- performance is measured with metrics like accuracy, precision, recall, F1 score, and AUC.

## License

MIT License
