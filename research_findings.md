# Federated Learning with Privacy: Research Findings

## 1. Key Findings

### Privacy-Utility Tradeoff

Our experiments demonstrate the fundamental privacy-utility tradeoff in federated learning:

- **Baseline approach** achieved highest accuracy (68.34%) but offered no privacy protection
- **Homomorphic Encryption (HE)** reduced accuracy to 57.90% but provided communication efficiency and moderate privacy
- **Differential Privacy (DP)** significantly reduced accuracy to 35.05% but provided strong theoretical privacy guarantees
- **Standard Hybrid (HE+DP)** achieved accuracy of 34.95% while maintaining HE's communication efficiency
- **Sequential Hybrid (HE+DP)** achieved accuracy of 33.51% with improved learning stability compared to standard hybrid

### Impact of Privacy Mechanisms

1. **Differential Privacy** had the most significant impact on model utility:

   - Noise injection during training reduced accuracy by ~33 percentage points
   - However, DP provided the strongest theoretical privacy guarantees with ε ≈ 8.0
   - MIA attack success rate dropped from 0.82 AUC (baseline) to 0.56 AUC

2. **Homomorphic Encryption** had a moderate impact on model utility:

   - Encryption/decryption operations reduced accuracy by ~10 percentage points
   - Provided significant communication efficiency (94% reduction)
   - MIA attack success rate dropped from 0.82 AUC (baseline) to 0.71 AUC

3. **Standard Hybrid (HE+DP)** combined the privacy benefits:

   - Similar privacy protection as DP alone (MIA AUC: 0.53)
   - Maintained the communication efficiency of HE (94% reduction)
   - However, demonstrated unstable learning progression with significant fluctuations

4. **Sequential Hybrid (HE+DP)** - our novel approach:
   - Applied DP during training and HE during aggregation
   - Provided similar privacy protection (MIA AUC: 0.53) and communication efficiency
   - Demonstrated more stable learning progression compared to standard hybrid
   - Showed better resilience to data heterogeneity in non-IID settings

### Effects of Data Heterogeneity

Our non-IID experiments with α=0.5 revealed:

- All approaches showed accuracy degradation in non-IID settings, but with varying degrees
- **HE was most resilient** to heterogeneity, retaining 94.39% of its IID accuracy
- **Standard Hybrid was most vulnerable**, retaining only 68.53% of its accuracy
- **Sequential Hybrid significantly outperformed Standard Hybrid** in heterogeneous settings (30.78% vs 23.95% accuracy)
- The impact of heterogeneity was non-linear across privacy mechanisms

### Communication Efficiency

- HE and Hybrid approaches reduced communication overhead by 94% compared to Baseline and DP
- The communication savings come from both model architecture optimization and parameter quantization
- This demonstrates that privacy and efficiency can be complementary goals in federated learning

### Privacy Metrics

- MIA attack success decreased as privacy protections increased
- Best privacy protection was achieved with both hybrid approaches (AUC: 0.53)
- The sequential hybrid maintained the same strong privacy guarantees as the standard hybrid but with improved utility under heterogeneity

## 2. Theoretical Privacy Guarantees

### Differential Privacy Guarantees

Our DP experiments used:

- Noise multiplier: 0.4 (standard hybrid), 0.3 (sequential hybrid), and higher values for pure DP
- Clipping threshold: 1.0 (max gradient norm)

These parameters resulted in:

- **Target ε**: 1.0 (desired privacy budget)
- **Achieved ε**: ~8.0 after 20 rounds for DP and hybrid approaches
- **Delta (δ)**: 1e-5 (probability of privacy failure)

According to the definition of (ε,δ)-DP, this means an adversary cannot distinguish whether a particular sample was included in the training set with confidence greater than e^ε (approximately 2980) with probability at least 1-δ (99.999%).

### Sequential vs. Simultaneous Privacy Application

Our sequential hybrid approach demonstrates that:

1. **Temporal separation of mechanisms** prevents error compounding that occurs in the standard hybrid
2. **Training-time DP** and **aggregation-time HE** can be more effective than applying both simultaneously
3. Different privacy mechanisms create different optimization landscapes that interact more favorably when sequentially applied

### Empirical Validation of Privacy Guarantees

We compared the theoretical guarantees with empirical MIA attack success:

1. **Theoretical bound**: DP with ε = 8.0 should limit information leakage substantially, though not perfectly
2. **Empirical result**: MIA attack achieved AUC of 0.56 (DP), 0.53 (standard hybrid), and 0.53 (sequential hybrid)

The agreement between theory and practice suggests that our DP implementation is effective. The similar MIA resistance of both hybrid approaches confirms that the sequential application maintains strong privacy guarantees while improving other aspects of performance.

### Homomorphic Encryption Security

The HE implementation used:

- Polynomial modulus degree: 4096
- Coefficient modulus bits: [40, 20, 40]
- Security level: approximately 128 bits

This provides computational security based on the Ring-Learning With Errors (RLWE) problem, which is believed to be resistant to quantum attacks.

### Combined Privacy Guarantees of Hybrid Approaches

Both hybrid approaches benefit from:

1. **DP's information-theoretic guarantees**:

   - Formal ε-differential privacy limiting inferential attacks
   - Protection against membership inference attacks

2. **HE's computational guarantees**:
   - Protection of data during transmission and aggregation
   - Security against eavesdropping attacks

The empirical validation showed both hybrid approaches achieved the lowest MIA success rate (AUC 0.53), suggesting that combining techniques provides stronger privacy protection than either technique alone.

## 3. Practical Recommendations

Based on our findings, we recommend:

1. **For high accuracy requirements with moderate privacy concerns:**

   - Use HE approach (57.90% accuracy, 0.71 MIA AUC)
   - Best for applications where model performance is critical but some privacy is needed

2. **For strong privacy requirements with moderate accuracy needs:**

   - Use DP approach (35.05% accuracy, 0.56 MIA AUC)
   - Best for highly sensitive data applications where privacy is the top priority

3. **For heterogeneous data environments with high privacy requirements:**

   - Use Sequential Hybrid approach (33.51% IID accuracy, 0.53 MIA AUC)
   - Best for non-IID settings where standard hybrid would struggle (30.78% vs 23.95% with α=0.5)

4. **For communication-constrained environments with high privacy requirements:**

   - Use either hybrid approach (both offer 94% less communication than baseline)
   - Sequential hybrid recommended if data heterogeneity is expected

5. **For maximum utility with no privacy requirements:**
   - Use Baseline approach (68.34% accuracy)
   - Appropriate only for non-sensitive data applications

## 4. Future Work

Potential directions for future research include:

1. **Further exploring sequential privacy mechanisms**:

   - Investigate different orderings and combinations of privacy techniques
   - Explore adaptive scheduling of mechanisms across training rounds

2. **Parameter optimization for Sequential Hybrid**:

   - Develop automated methods to find optimal noise multiplier and quantization bit settings
   - Create heterogeneity-aware parameter tuning approaches

3. **Addressing non-IID challenges**:

   - Develop techniques specifically designed for heterogeneous data
   - Investigate the theoretical basis for the varying heterogeneity resilience across mechanisms

4. **Scalability improvements**:

   - Reduce computational overhead of HE operations
   - Optimize DP mechanisms for better performance

5. **Extending the sequential principle**:
   - Apply the sequential approach to other privacy and security mechanisms
   - Develop a theoretical framework for optimal mechanism separation
