# Non-IID Data Analysis in Federated Learning with Privacy

## 1. Impact of Data Heterogeneity

### Accuracy Comparison

Our experiments with non-IID data (α=0.1) revealed significant impacts on model performance across all privacy approaches:

| Approach | IID Accuracy | Non-IID Accuracy | Absolute Drop | Relative Drop |
| -------- | ------------ | ---------------- | ------------- | ------------- |
| Baseline | 68.34%       | 58.17%           | 10.17%        | 14.88%        |
| HE       | 57.90%       | 53.32%           | 4.58%         | 7.91%         |
| DP       | 35.05%       | 28.51%           | 6.54%         | 18.66%        |
| HE+DP    | 34.95%       | 19.52%           | 15.43%        | 44.15%        |

![IID vs Non-IID Accuracy Comparison](/analysis_results/comparison/iid_vs_non_iid_accuracy.png)

### Key Observations

1. **All approaches suffer performance degradation** with heterogeneous data distributions
2. **HE shows the highest resilience** to data heterogeneity with only 7.91% relative accuracy drop
3. **Hybrid approach (HE+DP) is most vulnerable** to non-IID data with 44.15% relative drop
4. **The privacy-heterogeneity relationship is non-linear** - approaches with stronger privacy protections generally suffer more from data heterogeneity

## 2. Privacy-Heterogeneity Tradeoff

![Privacy-Heterogeneity Tradeoff](/analysis_results/comparison/privacy_heterogeneity_tradeoff.png)

### Heterogeneity Sensitivity Analysis

- **Baseline**: Moderate sensitivity (14.88% drop)
- **HE**: Low sensitivity (7.91% drop) despite providing moderate privacy
- **DP**: Medium-high sensitivity (18.66% drop)
- **HE+DP**: Very high sensitivity (44.15% drop)

The data suggests a significant three-way tradeoff between privacy, utility, and robustness to data heterogeneity.

![Three-way Tradeoff](/analysis_results/comparison/three_way_tradeoff.png)

## 3. Robustness to Heterogeneity

Each approach's ability to maintain accuracy with heterogeneous data can be measured as "accuracy retention":

![Accuracy Retention](/analysis_results/comparison/accuracy_retention.png)

### Heterogeneity Robustness Ranking

1. **HE**: 92.1% accuracy retention - Most robust
2. **Baseline**: 85.1% accuracy retention
3. **DP**: 81.3% accuracy retention
4. **HE+DP**: 55.9% accuracy retention - Least robust

This finding is particularly interesting because it suggests that homomorphic encryption not only provides privacy but also improves model robustness to data heterogeneity compared to the baseline.

## 4. Revised Practical Recommendations

Based on our combined IID and non-IID results, we revise our practical recommendations:

### For IID Data Scenarios:

1. **High accuracy, no privacy**: Baseline (68.34%)
2. **Moderate privacy, good accuracy**: HE (57.90%)
3. **Strong privacy, acceptable accuracy**: DP (35.05%) or HE+DP (34.95%)

### For Non-IID Data Scenarios:

1. **High accuracy, no privacy**: Baseline (58.17%)
2. **Moderate privacy, good accuracy**: HE (53.32%)
3. **Strong privacy, moderate accuracy**: DP (28.51%)
4. **Very strong privacy, lower accuracy**: HE+DP (19.52%)

**For highly heterogeneous data**: HE is strongly recommended as it provides the best balance of privacy, utility, and robustness to heterogeneity.

## 5. Theoretical Insights

1. **Why is HE more resilient to heterogeneity?**

   - HE operates on locally trained models without modifying the training process
   - Privacy is provided through encryption of model updates, not training perturbation
   - This allows models to learn local patterns effectively despite heterogeneity

2. **Why does the hybrid approach struggle most?**
   - The combination of noise injection (DP) and encryption/quantization (HE) compounds the challenges
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

3. **Personalization Techniques**:

   - Implement personalization layers to handle local distribution shifts
   - Keep shared base model with private personalization heads

4. **Improved Hybrid Approaches**:
   - Develop heterogeneity-aware hybrid privacy techniques
   - Consider sequential application rather than simultaneous application of HE and DP

## 7. Detailed Client Data Analysis

The extreme skew in our non-IID experiment (α=0.1) led to highly imbalanced client data:

- Some clients had over 80% of their samples from a single class
- Many clients had no samples from certain classes
- Example: Client 5 had 97% of samples from class 4

This level of heterogeneity is representative of real-world federated learning scenarios, where clients have very different data distributions based on user behavior and demographics.

## 8. Conclusion

Our analysis reveals that data heterogeneity significantly impacts the effectiveness of privacy-preserving techniques in federated learning. While all approaches experience accuracy degradation with non-IID data, HE demonstrates remarkable resilience, making it particularly suitable for real-world federated learning deployments with heterogeneous data.

The hybrid approach, which provides the strongest privacy guarantees, struggles most with heterogeneous data, suggesting that there is significant room for improvement in developing privacy mechanisms that are robust to data heterogeneity.
