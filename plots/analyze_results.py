import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os

# Create a folder for analysis results
os.makedirs('analysis_results', exist_ok=True)

# Function to load results from a JSON file (if you saved the results)
def load_results(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Original IID results
iid_results = {
    'Baseline': {
        'accuracy': 68.34,
        'privacy_metrics': {
            'test_auc': 0.82,  # Replace with your actual values
            'test_accuracy': 0.76,
            'test_precision': 0.75,
            'test_recall': 0.74,
            'test_f1': 0.74
        },
        'communication_overhead': 1750.89,
        'epsilon': 'N/A',  # No DP in baseline
    },
    'HE': {
        'accuracy': 57.90,
        'privacy_metrics': {
            'test_auc': 0.71,  # Replace with your actual values
            'test_accuracy': 0.68,
            'test_precision': 0.67,
            'test_recall': 0.66,
            'test_f1': 0.66
        },
        'communication_overhead': 104.43,
        'epsilon': 'N/A',  # No DP in HE
    },
    'DP': {
        'accuracy': 35.05,
        'privacy_metrics': {
            'test_auc': 0.56,  # Replace with your actual values
            'test_accuracy': 0.52,
            'test_precision': 0.51,
            'test_recall': 0.50,
            'test_f1': 0.50
        },
        'communication_overhead': 1750.89,
        'epsilon': 8.0,  # Replace with your actual epsilon value
    },
    'HE+DP': {
        'accuracy': 34.95,
        'privacy_metrics': {
            'test_auc': 0.53,  # Replace with your actual values
            'test_accuracy': 0.51,
            'test_precision': 0.49,
            'test_recall': 0.48,
            'test_f1': 0.48
        },
        'communication_overhead': 104.43,
        'epsilon': 8.0,  # Replace with your actual epsilon value
    },
    'Sequential HE+DP': {
        'accuracy': 33.51,  # Update with your actual value from the experiment
        'privacy_metrics': {
            'test_auc': 0.53,  # Update with your actual value from the experiment
            'test_accuracy': 0.51,
            'test_precision': 0.49,
            'test_recall': 0.48,
            'test_f1': 0.48
        },
        'communication_overhead': 104.43,
        'epsilon': 8.0,
    }
}

# New non-IID results with alpha=0.1
non_iid_results = {
    'Baseline': {
        'accuracy': 58.17,
        'privacy_metrics': {
            'test_auc': 0.79,  # Estimated, replace with actual values if available
            'test_accuracy': 0.73,
            'test_precision': 0.72,
            'test_recall': 0.71,
            'test_f1': 0.71
        },
        'communication_overhead': 1750.89,
        'epsilon': 'N/A',
    },
    'HE': {
        'accuracy': 53.32,
        'privacy_metrics': {
            'test_auc': 0.68,  # Estimated, replace with actual values if available
            'test_accuracy': 0.65,
            'test_precision': 0.64,
            'test_recall': 0.63,
            'test_f1': 0.63
        },
        'communication_overhead': 104.43,
        'epsilon': 'N/A',
    },
    'DP': {
        'accuracy': 28.51,
        'privacy_metrics': {
            'test_auc': 0.54,  # Estimated, replace with actual values if available
            'test_accuracy': 0.50,
            'test_precision': 0.49,
            'test_recall': 0.48,
            'test_f1': 0.48
        },
        'communication_overhead': 1750.89,
        'epsilon': 8.0,
    },
    'HE+DP': {
        'accuracy': 19.52,
        'privacy_metrics': {
            'test_auc': 0.51,  # Estimated, replace with actual values if available
            'test_accuracy': 0.49,
            'test_precision': 0.47,
            'test_recall': 0.46,
            'test_f1': 0.46
        },
        'communication_overhead': 104.43,
        'epsilon': 8.0,
    },
    'Sequential HE+DP': {
        'accuracy': 22.25,  # Update with your actual value from the experiment
        'privacy_metrics': {
            'test_auc': 0.50,  # Update with your actual value from the experiment
            'test_accuracy': 0.49,
            'test_precision': 0.47,
            'test_recall': 0.46,
            'test_f1': 0.46
        },
        'communication_overhead': 104.43,
        'epsilon': 8.0,
    }
}

# Function to generate all original visualizations
def generate_original_visualizations(results, prefix=""):
    # Visualization 1: Privacy-Utility Tradeoff
    plt.figure(figsize=(12, 8))
    for approach, data in results.items():
        auc = data['privacy_metrics']['test_auc']
        accuracy = data['accuracy']
        plt.scatter(auc, accuracy, s=100, label=approach)

    # Connecting the points to show a frontier
    approaches = ['HE+DP', 'DP', 'HE', 'Baseline']
    x = [results[a]['privacy_metrics']['test_auc'] for a in approaches]
    y = [results[a]['accuracy'] for a in approaches]
    plt.plot(x, y, 'r--')

    plt.xlabel('Vulnerability to MIA (AUC)', fontsize=14)
    plt.ylabel('Model Accuracy (%)', fontsize=14)
    plt.title('Privacy-Utility Tradeoff in Federated Learning', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig(f'analysis_results/{prefix}privacy_utility_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 2: Detailed Privacy Metrics Comparison
    metrics = ['test_auc', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    labels = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
    approaches = list(results.keys())

    x = np.arange(len(metrics))
    width = 0.2
    multiplier = 0

    fig, ax = plt.subplots(figsize=(14, 8))

    for approach, data in results.items():
        offset = width * multiplier
        privacy_values = [data['privacy_metrics'][m] for m in metrics]
        rects = ax.bar(x + offset, privacy_values, width, label=approach)
        multiplier += 1

    ax.set_ylabel('MIA Attack Performance', fontsize=14)
    ax.set_title('Privacy Metrics Across Approaches', fontsize=16)
    ax.set_xticks(x + width, labels)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'analysis_results/{prefix}privacy_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 3: Composite Metrics (Efficiency vs Privacy vs Accuracy)
    plt.figure(figsize=(10, 8))

    for approach, data in results.items():
        auc = data['privacy_metrics']['test_auc']
        accuracy = data['accuracy']
        comm = data['communication_overhead']
        plt.scatter(auc, accuracy, s=comm/10, alpha=0.7, label=f"{approach} (Comm: {comm:.1f} MB)")

    plt.xlabel('Vulnerability to MIA (AUC)', fontsize=14)
    plt.ylabel('Model Accuracy (%)', fontsize=14)
    plt.title('3D Trade-off: Privacy vs Utility vs Communication', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.savefig(f'analysis_results/{prefix}3d_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create comprehensive comparison visualizations
    plt.figure(figsize=(14, 10))

    # 1. Bar chart comparing final accuracy
    plt.subplot(2, 2, 1)
    approaches = list(results.keys())
    accuracies = [results[a]['accuracy'] for a in approaches]
    plt.bar(approaches, accuracies, color=['blue', 'green', 'orange', 'red', 'purple'])
    plt.ylabel('Final Accuracy (%)', fontsize=12)
    plt.title('Final Model Accuracy', fontsize=14)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)

    # 2. Bar chart comparing MIA success rate (AUC)
    plt.subplot(2, 2, 2)
    mia_rates = [results[a]['privacy_metrics']['test_auc'] for a in approaches]
    plt.bar(approaches, mia_rates, color=['blue', 'green', 'orange', 'red', 'purple'])
    plt.ylabel('MIA Success Rate (AUC)', fontsize=12)
    plt.title('Vulnerability to Membership Inference', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)

    # 3. Bar chart comparing communication overhead
    plt.subplot(2, 2, 3)
    comm_overhead = [results[a]['communication_overhead'] for a in approaches]
    plt.bar(approaches, comm_overhead, color=['blue', 'green', 'orange', 'red', 'purple'])
    plt.ylabel('Communication Overhead (MB)', fontsize=12)
    plt.title('Communication Efficiency', fontsize=14)
    plt.grid(axis='y', alpha=0.3)

    # 4. Composite score (normalized weighted sum)
    plt.subplot(2, 2, 4)
    # Normalize each metric
    norm_acc = [a/max(accuracies) for a in accuracies]  # Higher is better
    norm_privacy = [1 - (p/max(mia_rates)) for p in mia_rates]  # Lower MIA is better
    norm_comm = [1 - (c/max(comm_overhead)) for c in comm_overhead]  # Lower comm is better

    # Equal weighting (0.33 each)
    composite_scores = [0.33*a + 0.33*p + 0.33*c for a, p, c in zip(norm_acc, norm_privacy, norm_comm)]
    plt.bar(approaches, composite_scores, color=['blue', 'green', 'orange', 'red', 'purple'])
    plt.ylabel('Composite Score', fontsize=12)
    plt.title('Overall Performance (Equally Weighted)', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'analysis_results/{prefix}comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate new IID vs non-IID comparison visualizations
def generate_comparison_visualizations():
    # Create a folder for IID vs non-IID comparisons
    os.makedirs('analysis_results/comparison', exist_ok=True)
    
    # 1. IID vs non-IID Accuracy Comparison
    plt.figure(figsize=(12, 8))
    approaches = ['Baseline', 'HE', 'DP', 'HE+DP', 'Sequential HE+DP']
    x = np.arange(len(approaches))
    width = 0.35
    
    # Get accuracies
    iid_accuracies = [iid_results[a]['accuracy'] for a in approaches]
    non_iid_accuracies = [non_iid_results[a]['accuracy'] for a in approaches]
    
    # Calculate accuracy drops
    accuracy_drops = [iid - non_iid for iid, non_iid in zip(iid_accuracies, non_iid_accuracies)]
    
    # Plot bars
    plt.bar(x - width/2, iid_accuracies, width, label='IID', color='royalblue')
    plt.bar(x + width/2, non_iid_accuracies, width, label='Non-IID (α=0.1)', color='crimson')
    
    # Add labels and formatting
    plt.xlabel('Approach', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('IID vs Non-IID Performance Comparison', fontsize=16)
    plt.xticks(x, approaches)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add accuracy drop annotations
    for i, drop in enumerate(accuracy_drops):
        plt.annotate(f'Drop: {drop:.2f}%', 
                     xy=(x[i], non_iid_accuracies[i] - 5), 
                     ha='center', 
                     va='top',
                     fontsize=10, 
                     color='black',
                     bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
    
    plt.savefig('analysis_results/comparison/iid_vs_non_iid_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Privacy-Heterogeneity Tradeoff
    plt.figure(figsize=(12, 8))
    
    # Calculate percentage accuracy loss from IID to non-IID
    percentage_drop = [(iid - non_iid) / iid * 100 for iid, non_iid in zip(iid_accuracies, non_iid_accuracies)]
    mia_protection = [1 - iid_results[a]['privacy_metrics']['test_auc'] for a in approaches]
    
    # Plot points
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    plt.figure(figsize=(10, 8))
    
    for i, approach in enumerate(approaches):
        plt.scatter(mia_protection[i], percentage_drop[i], s=150, color=colors[i], label=approach)
    
    # Add labels and formatting
    plt.xlabel('Privacy Protection (1 - MIA AUC)', fontsize=14)
    plt.ylabel('Heterogeneity Sensitivity (% Accuracy Loss)', fontsize=14)
    plt.title('Privacy Protection vs Heterogeneity Sensitivity', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add annotations
    for i, approach in enumerate(approaches):
        plt.annotate(approach, 
                     xy=(mia_protection[i], percentage_drop[i]), 
                     xytext=(10, 10),
                     textcoords='offset points',
                     ha='center',
                     fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    plt.savefig('analysis_results/comparison/privacy_heterogeneity_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Detailed Performance Comparison Table
    performance_data = {
        'Approach': approaches,
        'IID Accuracy (%)': iid_accuracies,
        'Non-IID Accuracy (%)': non_iid_accuracies,
        'Accuracy Drop (%)': accuracy_drops,
        'Drop Percentage (%)': percentage_drop,
        'MIA AUC': [iid_results[a]['privacy_metrics']['test_auc'] for a in approaches],
        'Communication (MB)': [iid_results[a]['communication_overhead'] for a in approaches],
        'Privacy Budget (ε)': [iid_results[a]['epsilon'] for a in approaches]
    }
    
    df = pd.DataFrame(performance_data)
    df.to_csv('analysis_results/comparison/performance_comparison.csv', index=False)
    
    # 4. Normalized Accuracy Retention with Heterogeneity
    plt.figure(figsize=(10, 6))
    
    # Calculate normalized accuracy retention (higher is better)
    accuracy_retention = [non_iid / iid * 100 for iid, non_iid in zip(iid_accuracies, non_iid_accuracies)]
    
    # Plot bars
    barcolors = ['royalblue', 'green', 'orange', 'firebrick', 'purple']
    plt.bar(approaches, accuracy_retention, color=barcolors)
    
    # Add labels and formatting
    plt.xlabel('Approach', fontsize=14)
    plt.ylabel('Accuracy Retention (%)', fontsize=14)
    plt.title('Robustness to Heterogeneity (Higher is Better)', fontsize=16)
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 100)
    
    # Add percentage annotations
    for i, retention in enumerate(accuracy_retention):
        plt.annotate(f'{retention:.1f}%', 
                     xy=(i, retention + 1), 
                     ha='center', 
                     va='bottom',
                     fontsize=12)
    
    plt.savefig('analysis_results/comparison/accuracy_retention.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Create 3x3 matrix visualization for the three-way tradeoff
    # Privacy (x) vs Accuracy (y) vs Heterogeneity Robustness (color)
    plt.figure(figsize=(12, 8))
    
    privacy_budget = []
    for approach in approaches:
        if iid_results[approach]['epsilon'] == 'N/A':
            privacy_budget.append(0)  # No privacy for baseline and HE
        else:
            privacy_budget.append(float(iid_results[approach]['epsilon']))
    
    # Normalize for color mapping (higher retention = better robustness = darker color)
    norm_retention = [ret/100 for ret in accuracy_retention]
    
    # Plot
    plt.scatter(privacy_budget, iid_accuracies, s=200, c=norm_retention, cmap='YlGn', 
                alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label('Heterogeneity Robustness (Accuracy Retention %)', fontsize=12)
    
    # Labels and annotations
    plt.xlabel('Privacy Budget (ε, 0 = No DP)', fontsize=14)
    plt.ylabel('IID Accuracy (%)', fontsize=14)
    plt.title('Three-way Tradeoff: Privacy vs Accuracy vs Heterogeneity Robustness', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Add approach labels to points
    for i, approach in enumerate(approaches):
        plt.annotate(approach, 
                     xy=(privacy_budget[i], iid_accuracies[i]), 
                     xytext=(10, 0),
                     textcoords='offset points',
                     fontsize=12,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    
    plt.savefig('analysis_results/comparison/three_way_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()

# First, generate original visualizations for IID and non-IID separately
print("Generating original visualizations for IID data...")
generate_original_visualizations(iid_results, prefix="iid_")

print("Generating original visualizations for non-IID data...")
generate_original_visualizations(non_iid_results, prefix="non_iid_")

# Then, generate new comparative visualizations
print("Generating comparison visualizations between IID and non-IID data...")
generate_comparison_visualizations()

# Print summary tables
print("\nSummary of Federated Learning with IID Data:")
print("="*100)
print(f"{'Approach':<10} {'Accuracy':<10} {'MIA AUC':<10} {'Comm (MB)':<12} {'Epsilon':<10}")
print("-"*100)
for approach in iid_results.keys():
    data = iid_results[approach]
    print(f"{approach:<10} {data['accuracy']:<10.2f} {data['privacy_metrics']['test_auc']:<10.2f} {data['communication_overhead']:<12.2f} {data['epsilon']}")
print("="*100)

print("\nSummary of Federated Learning with Non-IID Data (α=0.1):")
print("="*100)
print(f"{'Approach':<10} {'Accuracy':<10} {'MIA AUC':<10} {'Comm (MB)':<12} {'Epsilon':<10}")
print("-"*100)
for approach in non_iid_results.keys():
    data = non_iid_results[approach]
    print(f"{approach:<10} {data['accuracy']:<10.2f} {data['privacy_metrics']['test_auc']:<10.2f} {data['communication_overhead']:<12.2f} {data['epsilon']}")
print("="*100)

# Calculate and print the accuracy drops
print("\nAccuracy Drops from IID to Non-IID:")
print("="*100)
print(f"{'Approach':<10} {'IID Acc':<10} {'Non-IID Acc':<12} {'Drop':<10} {'Drop %':<10}")
print("-"*100)
for approach in iid_results.keys():
    iid_acc = iid_results[approach]['accuracy']
    non_iid_acc = non_iid_results[approach]['accuracy']
    drop = iid_acc - non_iid_acc
    drop_percent = (drop / iid_acc) * 100
    print(f"{approach:<10} {iid_acc:<10.2f} {non_iid_acc:<12.2f} {drop:<10.2f} {drop_percent:<10.2f}%")
print("="*100)

print("\nAnalysis completed! All visualizations saved to the 'analysis_results' folder and comparison visualizations to 'analysis_results/comparison' subfolder.") 