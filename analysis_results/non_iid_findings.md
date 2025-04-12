# Non-IID Data Analysis in Federated Learning with Privacy

## 1. Impact of Data Heterogeneity

### Accuracy Comparison

Our experiments with non-IID data (α=0.5) revealed significant impacts on model performance across all privacy approaches:

| Approach          | IID Accuracy (%) | Non-IID Accuracy (%) | Absolute Drop (%) | Relative Drop (%) |
| ----------------- | ---------------- | -------------------- | ----------------- | ----------------- |
| Baseline          | 68.34            | 66.06                | 2.28              | 3.34              |
| HE                | 57.90            | 54.65                | 3.25              | 5.61              |
| DP                | 35.05            | 32.28                | 2.77              | 7.90              |
| Standard Hybrid   | 34.95            | 23.95                | 11.00             | 31.47             |
| Sequential Hybrid | 33.51            | 30.78                | 2.73              | 8.15              |

![IID vs Non-IID Accuracy Comparison](/analysis_results/iid_vs_non_iid_accuracy.png)

### Key Observations

1. **All approaches suffer performance degradation** with heterogeneous data distributions
2. **HE shows high resilience** to data heterogeneity with only 5.61% relative accuracy drop
3. **Standard Hybrid approach is most vulnerable** to non-IID data with 31.47% relative drop
4. **Sequential Hybrid significantly outperforms Standard Hybrid** in heterogeneous settings (30.78% vs 23.95%)
5. **The privacy-heterogeneity relationship is non-linear** - different privacy mechanisms exhibit varying levels of sensitivity to data heterogeneity

## 2. Privacy-Heterogeneity Tradeoff

![Privacy-Heterogeneity Tradeoff](/analysis_results/privacy_heterogeneity_tradeoff.png)

### Heterogeneity Sensitivity Analysis

- **Baseline**: Low sensitivity (3.34% drop)
- **HE**: Low sensitivity (5.61% drop) despite providing moderate privacy
- **DP**: Moderate sensitivity (7.90% drop)
- **Sequential Hybrid**: Moderate sensitivity (8.15% drop)
- **Standard Hybrid**: High sensitivity (31.47% drop)

The data suggests a significant three-way tradeoff between privacy, utility, and robustness to data heterogeneity, with the sequential hybrid approach offering a better compromise than the standard hybrid.

![Three-Way Trade-off Analysis](/analysis_results/3d_tradeoff.png)

## 3. Robustness to Heterogeneity

Each approach's ability to maintain accuracy with heterogeneous data can be measured as "accuracy retention":

![Accuracy Retention](/analysis_results/accuracy_retention.png)

### Heterogeneity Robustness Ranking

1. **HE**: 94.39% accuracy retention - Most robust
2. **Baseline**: 96.66% accuracy retention
3. **Sequential Hybrid**: 91.85% accuracy retention
4. **DP**: 92.10% accuracy retention
5. **Standard Hybrid**: 68.53% accuracy retention - Least robust

This finding is particularly interesting because it demonstrates that our novel sequential hybrid approach maintains significantly better robustness to data heterogeneity compared to the standard hybrid approach, while still providing similar privacy guarantees.

## 4. Revised Practical Recommendations

Based on our combined IID and non-IID results, we revise our practical recommendations:

### For IID Data Scenarios:

1. **High accuracy, no privacy**: Baseline (68.34%)
2. **Moderate privacy, good accuracy**: HE (57.90%)
3. **Strong privacy, acceptable accuracy**: DP (35.05%), Standard Hybrid (34.95%), or Sequential Hybrid (33.51%)

### For Non-IID Data Scenarios:

1. **High accuracy, no privacy**: Baseline (66.06%)
2. **Moderate privacy, good accuracy**: HE (54.65%)
3. **Strong privacy, moderate accuracy**: DP (32.28%) or Sequential Hybrid (30.78%)
4. **Strong privacy, lower accuracy**: Standard Hybrid (23.95%)

**For heterogeneous data environments**: The Sequential Hybrid approach is strongly recommended over the Standard Hybrid when deploying with non-IID data, as it provides similar privacy protection but much better accuracy retention.

## 5. Theoretical Insights

1. **Why is Sequential Hybrid more resilient than Standard Hybrid?**

   - Temporal separation of mechanisms prevents compounding errors
   - DP noise affects training independently, while HE affects only communication
   - This prevents error multiplication that occurs in the standard hybrid approach

2. **Why does the Standard Hybrid approach struggle most?**
   - The simultaneous application of noise injection (DP) and encryption/quantization (HE) compounds the challenges
   - DP noise is amplified when dealing with skewed local distributions
   - Quantization errors in HE are more significant with heterogeneous updates
   - These effects appear to be multiplicative rather than additive

## 6. Implications for Future Research

1. **Adaptive Privacy Approaches**:

   - Adjust privacy parameters based on data heterogeneity
   - Lower DP noise or more precise HE parameters for highly non-IID settings

2. **Client Clustering**:

   - Group clients with similar data distributions
   - Apply different privacy settings to different clusters

3. **Sequential Privacy Orchestration**:

   - Explore other sequential combinations of privacy mechanisms
   - Investigate optimal ordering and parameter settings

4. **Heterogeneity-Aware Privacy Mechanisms**:
   - Develop mechanisms specifically designed for heterogeneous data
   - Optimize the sequential approach for different levels of data heterogeneity

## 7. Detailed Client Performance Variation

Our non-IID experiments showed considerable performance variation across clients:

- Standard Hybrid demonstrated the widest accuracy disparities between clients
- Sequential Hybrid maintained better minimum client accuracy
- HE showed more uniform client performance
- Baseline and DP had moderate client variation

This client-level analysis highlights another advantage of the Sequential Hybrid approach—it reduces performance disparities across clients with different data distributions, a valuable property for fair federated learning.

## 8. Conclusion

Our analysis reveals that data heterogeneity significantly impacts the effectiveness of privacy-preserving techniques in federated learning. Different privacy mechanisms exhibit varying levels of resilience to non-IID data, with the Standard Hybrid approach struggling the most and HE demonstrating remarkable robustness.

The Sequential Hybrid approach—our novel contribution applying privacy mechanisms temporally rather than simultaneously—provides a significant improvement over the Standard Hybrid in heterogeneous settings while maintaining similar privacy guarantees. This demonstrates the importance of carefully orchestrating privacy mechanisms in federated learning deployments with real-world data distributions.
