import argparse
import logging
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from experiments.baseline_experiment import run_baseline_experiment
from experiments.he_experiment import run_he_experiment
from experiments.dp_experiment import run_dp_experiment
from experiments.hybrid_experiment import run_hybrid_experiment
from experiments.sequential_hybrid_experiment import run_sequential_hybrid_experiment

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Federated Learning with Privacy Experiments')
    
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['baseline', 'he', 'dp', 'hybrid', 'sequential_hybrid', 'all'],
                        help='Which experiment to run')
    
    parser.add_argument('--num_clients', type=int, default=10,
                        help='Number of clients to use')
    
    parser.add_argument('--num_rounds', type=int, default=20,
                        help='Number of federated learning rounds')
    
    parser.add_argument('--iid', action='store_true', default=True,
                        help='Use IID data distribution')
    
    parser.add_argument('--non_iid', action='store_true', default=False,
                        help='Use non-IID data distribution')
    
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet concentration parameter for non-IID data (lower = more heterogeneity)')
    
    parser.add_argument('--local_epochs', type=int, default=1,
                        help='Number of local training epochs per round')
    
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    
    # DP parameters
    parser.add_argument('--target_epsilon', type=float, default=1.0,
                        help='Target epsilon for differential privacy')
    
    parser.add_argument('--target_delta', type=float, default=1e-5,
                        help='Target delta for differential privacy')
    
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for DP')
    
    parser.add_argument('--noise_multiplier', type=float, default=None,
                        help='Fixed noise multiplier for DP (overrides epsilon-based calculation if provided)')
    
    # HE parameters
    parser.add_argument('--poly_modulus_degree', type=int, default=4096,
                        help='Polynomial modulus degree for HE')
    
    parser.add_argument('--scale', type=int, default=2**20,
                        help='Scale for fixed-point encoding in HE')
    
    parser.add_argument('--quantize_bits', type=int, default=16,
                        help='Number of bits for quantization in HE')
    
    # MIA parameters
    parser.add_argument('--mia_attack_type', type=str, default='advanced',
                      choices=['threshold', 'advanced'],
                      help='Type of MIA attack to use for privacy evaluation')
    
    return parser.parse_args()

def compare_results(results_dict):
    """
    Compare and visualize results from different approaches.
    
    Args:
        results_dict: Dictionary mapping approach names to their results
    """
    # Create figure for comparison
    plt.figure(figsize=(20, 15))
    
    # Define a consistent color scheme for all approaches
    color_map = {
        'Baseline': 'blue',
        'HE': 'green',
        'DP': 'orange',
        'HE+DP': 'red',
        'Sequential HE+DP': 'purple'
    }
    
    # Compare accuracy
    plt.subplot(2, 2, 1)
    for name, results in results_dict.items():
        plt.plot(results['accuracy'], label=name, color=color_map.get(name, 'gray'))
    plt.title('Accuracy Comparison')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Compare loss
    plt.subplot(2, 2, 2)
    for name, results in results_dict.items():
        plt.plot(results['loss'], label=name, color=color_map.get(name, 'gray'))
    plt.title('Loss Comparison')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Compare computation time
    plt.subplot(2, 2, 3)
    for name, results in results_dict.items():
        plt.plot(results['computation_time'], label=name, color=color_map.get(name, 'gray'))
    plt.title('Computation Time Comparison')
    plt.xlabel('Round')
    plt.ylabel('Seconds')
    plt.legend()
    plt.grid(True)
    
    # Compare MIA privacy metrics - Use AUC if available
    plt.subplot(2, 2, 4)
    names = []
    metrics_to_plot = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
    metrics_values = {metric: [] for metric in metrics_to_plot}
    
    for name, results in results_dict.items():
        metrics = results['privacy_metrics']
        names.append(name)
        for metric in metrics_to_plot:
            metrics_values[metric].append(metrics.get(metric, 0))
    
    x = np.arange(len(names))
    width = 0.15
    offsets = np.linspace(-0.3, 0.3, len(metrics_to_plot))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x + offsets[i], metrics_values[metric], width, label=metric.replace('test_', ''))
    
    plt.title('MIA Attack Performance Comparison')
    plt.xticks(x, names, rotation=15)  # Add rotation to handle longer names
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results_comparison.png')
    plt.close()
    
    # Create a summary table
    summary = {
        'Approach': [],
        'Final Accuracy (%)': [],
        'Final Loss': [],
        'Total Time (s)': [],
        'MIA Success Rate (AUC)': [],
        'MIA Success Rate (Acc)': [],
        'Communication Overhead (MB)': [],
        'Data Distribution': []
    }
    
    # Get experiment configuration
    iid_status = results_dict[list(results_dict.keys())[0]].get('iid', True)
    
    for name, results in results_dict.items():
        summary['Approach'].append(name)
        summary['Final Accuracy (%)'].append(results['accuracy'][-1])
        summary['Final Loss'].append(results['loss'][-1])
        summary['Total Time (s)'].append(results.get('total_time', 
                                                    sum(results['computation_time'])))
        
        # Check which metrics are available
        metrics = results['privacy_metrics']
        summary['MIA Success Rate (AUC)'].append(
            metrics.get('test_auc', 0) * 100 if 'test_auc' in metrics else 0)
        summary['MIA Success Rate (Acc)'].append(
            metrics.get('test_accuracy', 0) * 100)
        
        # Convert bytes to MB
        total_comm = sum(results['communication_overhead']) / (1024 * 1024)
        summary['Communication Overhead (MB)'].append(total_comm)
        
        # Add data distribution information
        summary['Data Distribution'].append("IID" if iid_status else "Non-IID")
    
    # Create DataFrame and save to CSV
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('results_summary.csv', index=False)
    print("Summary of results:")
    print(summary_df)
    
    return summary_df

def main():
    """Main function to run experiments."""
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Federated Learning Privacy Experiments")
    
    # If both iid and non_iid flags are set, prefer non_iid
    if args.non_iid:
        args.iid = False
    
    results = {}
    
    if args.experiment == 'baseline' or args.experiment == 'all':
        logging.info("Running Baseline FL experiment")
        results['Baseline'] = run_baseline_experiment(
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            iid=args.iid,
            alpha=args.alpha if not args.iid else None,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            mia_attack_type=args.mia_attack_type
        )
    
    if args.experiment == 'he' or args.experiment == 'all':
        logging.info("Running FL with HE experiment")
        results['HE'] = run_he_experiment(
            num_clients=args.num_clients,
            num_rounds=min(args.num_rounds, 10),  # Fewer rounds for HE due to overhead
            iid=args.iid,
            alpha=args.alpha if not args.iid else None,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            poly_modulus_degree=args.poly_modulus_degree,
            scale=args.scale,
            quantize_bits=args.quantize_bits,
            mia_attack_type=args.mia_attack_type
        )
    
    if args.experiment == 'dp' or args.experiment == 'all':
        logging.info("Running FL with DP experiment")
        results['DP'] = run_dp_experiment(
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            iid=args.iid,
            alpha=args.alpha if not args.iid else None,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            max_grad_norm=args.max_grad_norm,
            noise_multiplier=args.noise_multiplier,
            mia_attack_type=args.mia_attack_type
        )
    
    if args.experiment == 'hybrid' or args.experiment == 'all':
        logging.info("Running FL with HE+DP experiment")
        # Use a lower noise_multiplier for hybrid by default (0.4)
        # Since HE already provides some privacy, we can use less noise for better accuracy
        hybrid_noise_multiplier = args.noise_multiplier if args.noise_multiplier is not None else 0.4
        
        results['HE+DP'] = run_hybrid_experiment(
            num_clients=args.num_clients,
            num_rounds=min(args.num_rounds, 10),  # Fewer rounds for hybrid due to overhead
            iid=args.iid,
            alpha=args.alpha if not args.iid else None,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            max_grad_norm=args.max_grad_norm,
            noise_multiplier=hybrid_noise_multiplier,
            poly_modulus_degree=args.poly_modulus_degree,
            scale=args.scale,
            quantize_bits=args.quantize_bits,
            mia_attack_type=args.mia_attack_type
        )
    
    if args.experiment == 'sequential_hybrid' or args.experiment == 'all':
        logging.info("Running FL with Sequential HE+DP experiment")
        # Use lower noise multiplier (0.3) and higher quantize bits (24) for better accuracy
        seq_hybrid_noise_multiplier = args.noise_multiplier if args.noise_multiplier is not None else 0.3
        seq_quantize_bits = 24  # Higher bit precision
        
        results['Sequential HE+DP'] = run_sequential_hybrid_experiment(
            num_clients=args.num_clients,
            num_rounds=min(args.num_rounds, 10),  # Fewer rounds due to overhead
            iid=args.iid,
            alpha=args.alpha if not args.iid else None,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            max_grad_norm=args.max_grad_norm,
            noise_multiplier=seq_hybrid_noise_multiplier,
            poly_modulus_degree=args.poly_modulus_degree,
            scale=args.scale,
            quantize_bits=seq_quantize_bits,
            mia_attack_type=args.mia_attack_type
        )
    
    # If more than one experiment was run, compare results
    if len(results) > 1:
        logging.info("Comparing results from different approaches")
        compare_results(results)

if __name__ == '__main__':
    main()
