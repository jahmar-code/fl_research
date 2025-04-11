import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.simple_cnn import SimpleCNN, SmallCNN
from federated.server import FederatedServer
from privacy.he import HEFederatedClient, HEFederatedServer
from data.dataset import load_cifar10, create_client_data, get_dataloaders
from attacks.mia import MembershipInferenceAttack
from experiments.baseline_experiment import plot_results

def run_he_experiment(num_clients=10, num_rounds=20, iid=True, alpha=0.5, local_epochs=1, batch_size=64,
                      poly_modulus_degree=4096, coeff_modulus_bits=[40, 20, 40],
                      scale=2**20, quantize_bits=16, mia_attack_type='advanced'):
    """
    Run a federated learning experiment with homomorphic encryption.
    
    Args:
        num_clients: Number of clients to create
        num_rounds: Number of federated training rounds
        iid: Whether to distribute data in an IID manner
        alpha: Dirichlet alpha parameter for non-IID data distribution (only used if iid=False)
        local_epochs: Number of local training epochs per round
        batch_size: Batch size for training
        poly_modulus_degree: Polynomial modulus degree for HE
        coeff_modulus_bits: Coefficient modulus bit-sizes for HE
        scale: Scale for fixed-point encoding in HE
        quantize_bits: Number of bits for quantization in HE
        mia_attack_type: Type of MIA attack to use ('threshold' or 'advanced')
        
    Returns:
        Dict containing experiment results
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Federated Learning with Homomorphic Encryption experiment")
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
    
    # Initialize global model - using SmallCNN for better HE compatibility
    global_model = SmallCNN()
    
    # Initialize server with HE capabilities
    server = HEFederatedServer(global_model, test_loader)
    
    # Initialize clients with HE capabilities
    clients = []
    for client_id in range(num_clients):
        client = HEFederatedClient(
            client_id=client_id,
            model=SmallCNN(),
            dataloader=client_dataloaders[client_id],
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.SGD,
            optimizer_args={'lr': 0.01, 'momentum': 0.9},
            poly_modulus_degree=poly_modulus_degree,
            coeff_modulus_bits=coeff_modulus_bits,
            scale=scale,
            quantize_bits=quantize_bits
        )
        clients.append(client)
        server.add_client(client)
    
    # Train the model
    logging.info(f"Starting training for {num_rounds} rounds")
    start_time = time.time()
    history = []
    
    for round_idx in range(num_rounds):
        logging.info(f"Round {round_idx+1}/{num_rounds}")
        
        # Select all clients to participate in this round
        selected_clients = server.select_clients()
        
        try:
            # Perform one round of federated averaging with HE
            round_metrics = server.federated_averaging(selected_clients)
            
            logging.info(f"Round {round_idx+1} completed. "
                        f"Accuracy: {round_metrics['accuracy']:.2f}%, "
                        f"Loss: {round_metrics['loss']:.4f}, "
                        f"Time: {round_metrics['computation_time']:.2f}s")
            
            history.append(round_metrics)
        except Exception as e:
            logging.error(f"Error in round {round_idx+1}: {e}")
            break
    
    total_time = time.time() - start_time
    logging.info(f"Total training time: {total_time:.2f} seconds")
    
    # Save a copy of the trained model for MIA evaluation
    trained_model = SmallCNN()
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
        'total_time': total_time,
        'iid': iid,
        'alpha': alpha
    }
    
    # Plot results
    plot_results(results, "FL with HE")
    
    return results

if __name__ == "__main__":
    run_he_experiment(num_rounds=10)  # Fewer rounds for HE due to computational overhead 