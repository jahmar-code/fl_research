import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.simple_cnn import SimpleCNN
from federated.server import FederatedServer
from privacy.dp import DPFederatedClient
from data.dataset import load_cifar10, create_client_data, get_dataloaders
from attacks.mia import MembershipInferenceAttack
from experiments.baseline_experiment import plot_results

def run_dp_experiment(num_clients=10, num_rounds=20, iid=True, alpha=0.5, local_epochs=1, batch_size=64,
                      target_epsilon=1.0, target_delta=1e-5, max_grad_norm=1.0, noise_multiplier=None,
                      mia_attack_type='advanced'):
    """
    Run a federated learning experiment with differential privacy.
    
    Args:
        num_clients: Number of clients to create
        num_rounds: Number of federated training rounds
        iid: Whether to distribute data in an IID manner
        alpha: Dirichlet alpha parameter for non-IID data distribution
        local_epochs: Number of local training epochs per round
        batch_size: Batch size for training
        target_epsilon: Target privacy budget
        target_delta: Target delta for (ε, δ)-DP
        max_grad_norm: Maximum L2 norm of per-sample gradients
        noise_multiplier: If provided, use this fixed noise multiplier instead of calculating from epsilon
        mia_attack_type: Type of MIA attack to use ('threshold' or 'advanced')
        
    Returns:
        Dict containing experiment results
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Federated Learning with Differential Privacy experiment")
    logging.info(f"Data distribution: {'IID' if iid else 'Non-IID (alpha=' + str(alpha) + ')'}")
    logging.info(f"DP parameters: epsilon={target_epsilon}, delta={target_delta}, max_grad_norm={max_grad_norm}")
    if noise_multiplier is not None:
        logging.info(f"Using fixed noise multiplier: {noise_multiplier}")
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load dataset
    logging.info("Loading CIFAR-10 dataset")
    train_dataset, test_dataset = load_cifar10()
    
    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Split training data among clients
    logging.info(f"Splitting data among {num_clients} clients (IID={iid})")
    client_data_indices = create_client_data(train_dataset, num_clients, iid=iid, alpha=alpha)
    client_dataloaders = get_dataloaders(train_dataset, client_data_indices, batch_size=batch_size)
    
    # Initialize global model
    global_model = SimpleCNN()
    
    # Initialize server
    server = FederatedServer(global_model, test_loader)
    
    # Initialize clients with DP capabilities
    clients = []
    for client_id in range(num_clients):
        client = DPFederatedClient(
            client_id=client_id,
            model=SimpleCNN(),
            dataloader=client_dataloaders[client_id],
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.SGD,
            optimizer_args={'lr': 0.01, 'momentum': 0.9},
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
            noise_multiplier=noise_multiplier
        )
        clients.append(client)
        server.add_client(client)
    
    # Train the model
    logging.info(f"Starting training for {num_rounds} rounds")
    start_time = time.time()
    history = []
    epsilon_history = []
    noise_multiplier_history = []
    
    for round_idx in range(num_rounds):
        logging.info(f"Round {round_idx+1}/{num_rounds}")
        
        # Select all clients to participate in this round
        selected_clients = server.select_clients()
        
        try:
            # Perform one round of federated averaging
            round_metrics = server.federated_averaging(selected_clients)
            
            # Track DP parameters across all clients
            client_stats = [client.train() for client in selected_clients]
            avg_epsilon = np.mean([stat.get('epsilon', 0) for stat in client_stats])
            avg_noise = np.mean([stat.get('noise_multiplier', 0) for stat in client_stats])
            
            epsilon_history.append(avg_epsilon)
            noise_multiplier_history.append(avg_noise)
            
            logging.info(f"Round {round_idx+1} completed. "
                        f"Accuracy: {round_metrics['accuracy']:.2f}%, "
                        f"Loss: {round_metrics['loss']:.4f}, "
                        f"Epsilon: {avg_epsilon:.4f}, "
                        f"Noise Multiplier: {avg_noise:.4f}, "
                        f"Time: {round_metrics['computation_time']:.2f}s")
            
            history.append(round_metrics)
        except Exception as e:
            logging.error(f"Error in round {round_idx+1}: {e}")
            break
    
    total_time = time.time() - start_time
    logging.info(f"Total training time: {total_time:.2f} seconds")
    logging.info(f"Final privacy budget spent (epsilon): {epsilon_history[-1]:.4f}")
    
    # Save a copy of the trained model for MIA evaluation
    trained_model = SimpleCNN()
    trained_model.load_state_dict(global_model.state_dict())
    
    # Evaluate privacy with MIA
    logging.info(f"Evaluating privacy with Membership Inference Attack (type: {mia_attack_type})")
    mia = MembershipInferenceAttack(trained_model)
    privacy_metrics = mia.evaluate_model_privacy(train_dataset, test_dataset, attack_type=mia_attack_type)
    
    # Output detailed privacy metrics
    logging.info("Privacy evaluation results:")
    for key, value in privacy_metrics.items():
        logging.info(f"  {key}: {value:.4f}")
    
    # Combine results
    results = {
        'accuracy': server.metrics['accuracy'],
        'loss': server.metrics['loss'],
        'communication_overhead': server.metrics['communication_overhead'],
        'computation_time': server.metrics['computation_time'],
        'privacy_metrics': privacy_metrics,
        'epsilon_history': epsilon_history,
        'noise_multiplier_history': noise_multiplier_history,
        'total_time': total_time,
        'iid': iid,
        'alpha': alpha
    }
    
    # Plot results including epsilon and noise multiplier
    plot_dp_results(results, "FL with DP")
    
    return results

def plot_dp_results(results, title):
    """
    Plot experiment results including privacy budget.
    
    Args:
        results: Dict containing experiment results
        title: Title for the plots
    """
    plt.figure(figsize=(20, 15))
    
    # Plot accuracy
    plt.subplot(2, 3, 1)
    plt.plot(results['accuracy'])
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    # Plot loss
    plt.subplot(2, 3, 2)
    plt.plot(results['loss'])
    plt.title(f'{title} - Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot epsilon (privacy budget spent)
    plt.subplot(2, 3, 3)
    plt.plot(results['epsilon_history'])
    plt.title(f'{title} - Privacy Budget (ε)')
    plt.xlabel('Round')
    plt.ylabel('Epsilon')
    plt.grid(True)
    
    # Plot noise multiplier
    plt.subplot(2, 3, 4)
    plt.plot(results['noise_multiplier_history'])
    plt.title(f'{title} - Noise Multiplier')
    plt.xlabel('Round')
    plt.ylabel('Noise Multiplier')
    plt.grid(True)
    
    # Plot computation time
    plt.subplot(2, 3, 5)
    plt.plot(results['computation_time'])
    plt.title(f'{title} - Computation Time')
    plt.xlabel('Round')
    plt.ylabel('Seconds')
    plt.grid(True)
    
    # Plot MIA metrics
    plt.subplot(2, 3, 6)
    metrics = results['privacy_metrics']
    
    # Get the available metrics (backward compatibility)
    available_metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    if 'test_auc' in metrics:
        available_metrics.append('test_auc')
    
    metric_values = [metrics.get(metric, 0) for metric in available_metrics]
    metric_labels = [metric.replace('test_', '') for metric in available_metrics]
    
    plt.bar(metric_labels, metric_values)
    plt.title(f'{title} - MIA Attack Metrics')
    plt.ylabel('Score')
    plt.grid(True)
    
    # Add data distribution info to title
    data_dist = 'IID' if results.get('iid', True) else f'Non-IID (α={results.get("alpha", 0.5)})'
    plt.suptitle(f'{title} - {data_dist}')
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}_results.png")
    plt.close()

if __name__ == "__main__":
    run_dp_experiment() 