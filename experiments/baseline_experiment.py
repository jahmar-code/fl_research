import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.simple_cnn import SimpleCNN
from federated.server import FederatedServer
from federated.client import FederatedClient
from data.dataset import load_cifar10, create_client_data, get_dataloaders
from attacks.mia import MembershipInferenceAttack

def run_baseline_experiment(num_clients=10, num_rounds=20, iid=True, alpha=0.5, local_epochs=1, batch_size=64, mia_attack_type='advanced'):
    """
    Run a baseline federated learning experiment.
    
    Args:
        num_clients: Number of clients to create
        num_rounds: Number of federated training rounds
        iid: Whether to distribute data in an IID manner
        alpha: Dirichlet alpha parameter for non-IID data distribution (only used if iid=False)
        local_epochs: Number of local training epochs per round
        batch_size: Batch size for training
        mia_attack_type: Type of MIA attack to use ('threshold' or 'advanced')
        
    Returns:
        Dict containing experiment results
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Baseline Federated Learning experiment")
    logging.info(f"Data distribution: {'IID' if iid else 'Non-IID (alpha=' + str(alpha) + ')'}")
    
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
    
    # Initialize clients
    clients = []
    for client_id in range(num_clients):
        client = FederatedClient(
            client_id=client_id,
            model=SimpleCNN(),
            dataloader=client_dataloaders[client_id],
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.SGD,
            optimizer_args={'lr': 0.01, 'momentum': 0.9}
        )
        clients.append(client)
        server.add_client(client)
    
    # Train the model
    logging.info(f"Starting training for {num_rounds} rounds")
    history = server.train(num_rounds=num_rounds)
    
    # Evaluate privacy with MIA
    logging.info(f"Evaluating privacy with Membership Inference Attack (type: {mia_attack_type})")
    mia = MembershipInferenceAttack(server.global_model)
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
        'iid': iid,
        'alpha': alpha
    }
    
    # Plot results
    plot_results(results, "Baseline FL")
    
    return results

def plot_results(results, title):
    """
    Plot experiment results.
    
    Args:
        results: Dict containing experiment results
        title: Title for the plots
    """
    plt.figure(figsize=(20, 15))
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(results['accuracy'])
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(results['loss'])
    plt.title(f'{title} - Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot communication overhead
    plt.subplot(2, 2, 3)
    plt.plot(results['communication_overhead'])
    plt.title(f'{title} - Communication Overhead')
    plt.xlabel('Round')
    plt.ylabel('Bytes')
    plt.grid(True)
    
    # Plot computation time
    plt.subplot(2, 2, 4)
    plt.plot(results['computation_time'])
    plt.title(f'{title} - Computation Time')
    plt.xlabel('Round')
    plt.ylabel('Seconds')
    plt.grid(True)
    
    # Add data distribution info to title
    data_dist = 'IID' if results.get('iid', True) else f'Non-IID (α={results.get("alpha", 0.5)})'
    plt.suptitle(f'{title} - {data_dist}')
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}_results.png")
    plt.close()

if __name__ == "__main__":
    run_baseline_experiment() 