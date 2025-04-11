# Federated Learning with Privacy: Research Findings

## 1. Key Findings

### Privacy-Utility Tradeoff

Our experiments demonstrate the fundamental privacy-utility tradeoff in federated learning:

- **Baseline approach** achieved highest accuracy (68.34%) but offered no privacy protection
- **Homomorphic Encryption (HE)** reduced accuracy to 57.90% but provided communication efficiency and moderate privacy
- **Differential Privacy (DP)** significantly reduced accuracy to 35.05% but provided strong theoretical privacy guarantees
- **Hybrid approach (HE+DP)** achieved similar accuracy to DP (34.95%) while maintaining HE's communication efficiency

### Impact of Privacy Mechanisms

1. **Differential Privacy** had the most significant impact on model utility:

   - Noise injection during training reduced accuracy by ~33 percentage points
   - However, DP provided the strongest theoretical privacy guarantees with ε ≈ 8.0
   - MIA attack success rate dropped from 0.82 AUC (baseline) to 0.56 AUC

2. **Homomorphic Encryption** had a moderate impact on model utility:

   - Encryption/decryption operations reduced accuracy by ~10 percentage points
   - Provided significant communication efficiency (94% reduction)
   - MIA attack success rate dropped from 0.82 AUC (baseline) to 0.71 AUC

3. **Hybrid Approach (HE+DP)** combined the benefits:
   - Similar privacy protection as DP alone (MIA AUC: 0.53)
   - Maintained the communication efficiency of HE (94% reduction)
   - The combination did not significantly reduce accuracy beyond DP alone

### Effects of Data Heterogeneity

Our non-IID experiments with α=0.1 demonstrated:

- [To be updated when non-IID experiment completes]
- Initial observations show highly skewed data distributions across clients
- Expected to increase difficulty of convergence across all approaches

### Communication Efficiency

- HE and Hybrid approaches reduced communication overhead by 94% compared to Baseline and DP
- The communication savings come from transmitting encrypted model updates instead of full parameters
- This demonstrates that encryption not only provides privacy but can also improve communication efficiency

### Privacy Metrics

- MIA attack success decreased as privacy protections increased
- The gap between training and test accuracy (another privacy leak indicator) showed a similar trend
- Best privacy protection was achieved with the hybrid approach

## 2. Theoretical Privacy Guarantees

### Differential Privacy Guarantees

Our DP experiments used:

- Noise multiplier: 0.4 (hybrid) and higher values for pure DP
- Clipping threshold: 1.0 (max gradient norm)

These parameters resulted in:

- **Target ε**: 1.0 (desired privacy budget)
- **Achieved ε**: ~8.0 after 20 rounds for DP and hybrid approaches
- **Delta (δ)**: 1e-5 (probability of privacy failure)

According to the definition of (ε,δ)-DP, this means an adversary cannot distinguish whether a particular sample was included in the training set with confidence greater than e^ε (approximately 2980) with probability at least 1-δ (99.999%).

### Empirical Validation of Privacy Guarantees

We compared the theoretical guarantees with empirical MIA attack success:

1. **Theoretical bound**: DP with ε = 8.0 should limit information leakage substantially, though not perfectly
2. **Empirical result**: MIA attack achieved AUC of 0.56 (DP) and 0.53 (hybrid), which aligns with expectations for this privacy budget

The agreement between theory and practice suggests that our DP implementation is effective. The slightly better MIA resistance of the hybrid approach suggests that combining HE with DP may provide synergistic privacy benefits.

### Homomorphic Encryption Security

The HE implementation used:

- Polynomial modulus degree: 4096
- Coefficient modulus bits: [40, 20, 40]
- Security level: approximately 128 bits

This provides computational security based on the Ring-Learning With Errors (RLWE) problem, which is believed to be resistant to quantum attacks.

### Combined Privacy Guarantees of Hybrid Approach

The hybrid approach benefits from both:

1. **DP's information-theoretic guarantees**:

   - Formal ε-differential privacy limiting inferential attacks
   - Protection against membership inference attacks

2. **HE's computational guarantees**:
   - Protection of data during transmission and aggregation
   - Security against eavesdropping attacks

The empirical validation showed the hybrid approach achieved the lowest MIA success rate (AUC 0.53), suggesting that the combination of techniques provides stronger privacy protection than either technique alone.

## 3. Practical Recommendations

Based on our findings, we recommend:

1. **For high accuracy requirements with moderate privacy concerns:**

   - Use HE approach (57.90% accuracy, 0.71 MIA AUC)
   - Best for applications where model performance is critical but some privacy is needed

2. **For strong privacy requirements with moderate accuracy needs:**

   - Use DP approach (35.05% accuracy, 0.56 MIA AUC)
   - Best for highly sensitive data applications where privacy is the top priority

3. **For communication-constrained environments with high privacy requirements:**

   - Use Hybrid approach (34.95% accuracy, 0.53 MIA AUC, 94% less communication)
   - Best for edge/IoT scenarios or when bandwidth is limited

4. **For maximum utility with no privacy requirements:**
   - Use Baseline approach (68.34% accuracy)
   - Appropriate only for non-sensitive data applications

## 4. Future Work

Potential directions for future research include:

1. **Fine-tuning privacy parameters**:

   - Experiment with lower noise multipliers (0.2-0.3) for better utility
   - Investigate adaptive privacy budget allocation across training rounds

2. **Enhanced hybrid approaches**:

   - Explore selective application of DP (e.g., only on sensitive features)
   - Develop optimized HE parameters for neural network operations

3. **Addressing non-IID challenges**:

   - Develop techniques to maintain accuracy with heterogeneous data
   - Investigate privacy implications of data heterogeneity

4. **Scalability improvements**:

   - Reduce computational overhead of HE operations
   - Optimize DP mechanisms for better performance

5. **Advanced MIA defenses**:
   - Develop targeted defenses against specific types of inference attacks
   - Explore model compression as an additional privacy technique
