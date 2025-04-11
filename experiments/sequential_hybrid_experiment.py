import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.simple_cnn import SimpleCNN, SmallCNN
from federated.server import FederatedServer
from privacy.sequential_hybrid import SequentialHybridClient, SequentialHybridServer
from data.dataset import load_cifar10, create_client_data, get_dataloaders
from attacks.mia import MembershipInferenceAttack
from experiments.dp_experiment import plot_dp_results

def run_sequential_hybrid_experiment(num_clients=10, num_rounds=20, iid=True, alpha=0.5, local_epochs=1, batch_size=64,
                         # DP parameters
                         target_epsilon=1.0, target_delta=1e-5, max_grad_norm=1.0, noise_multiplier=None,
                         # HE parameters
                         poly_modulus_degree=4096, coeff_modulus_bits=[40, 20, 40],
                         scale=2**20, quantize_bits=24, mia_attack_type='advanced'):
    """
    Run a federated learning experiment with sequential application of DP and HE.
    
    Args:
        num_clients: Number of clients to create
        num_rounds: Number of federated training rounds
        iid: Whether to distribute data in an IID manner
        alpha: Dirichlet alpha parameter for non-IID data distribution (only used if iid=False)
        local_epochs: Number of local training epochs per round
        batch_size: Batch size for training
        
        # DP parameters
        target_epsilon: Target privacy budget
        target_delta: Target delta for (ε, δ)-DP
        max_grad_norm: Maximum L2 norm of per-sample gradients
        noise_multiplier: If provided, use this fixed noise multiplier instead of calculating from epsilon
        
        # HE parameters
        poly_modulus_degree: Polynomial modulus degree for HE
        coeff_modulus_bits: Coefficient modulus bit-sizes for HE
        scale: Scale for fixed-point encoding in HE
        quantize_bits: Number of bits for quantization in HE
        mia_attack_type: Type of MIA attack to use ('threshold' or 'advanced')
        
    Returns:
        Dict containing experiment results
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Federated Learning with Sequential HE+DP Privacy experiment")
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
    
    # Print the data distribution statistics for each client
    logging.info("Data distribution statistics:")
    for client_id, indices in enumerate(client_data_indices):
        labels = [train_dataset[idx][1] for idx in indices]
        label_counts = {}
        for label in labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        logging.info(f"Client {client_id}: {len(indices)} samples, label distribution: {label_counts}")
    
    # Initialize global model - using SmallCNN for better HE compatibility
    global_model = SmallCNN()
    
    # Initialize server with Sequential HE+DP capabilities
    server = SequentialHybridServer(global_model, test_loader)
    
    # Initialize clients with Sequential HE+DP capabilities
    clients = []
    for client_id in range(num_clients):
        client = SequentialHybridClient(
            client_id=client_id,
            model=SmallCNN(),
            dataloader=client_dataloaders[client_id],
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.SGD,
            optimizer_args={'lr': 0.01, 'momentum': 0.9},
            # DP parameters - using lower noise_multiplier for better accuracy
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
            noise_multiplier=noise_multiplier if noise_multiplier is not None else 0.3,
            # HE parameters - using higher quantize_bits for better precision
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
    epsilon_history = []
    noise_multiplier_history = []  # Track noise multiplier
    
    for round_idx in range(num_rounds):
        logging.info(f"--- Round {round_idx+1}/{num_rounds} ---")
        
        # Select all clients to participate in this round
        selected_clients = server.select_clients()
        
        try:
            # Perform one round of federated averaging with sequential privacy
            round_metrics = server.federated_averaging(selected_clients)
            
            # Extract epsilon values
            if 'avg_epsilon' in round_metrics:
                epsilon = round_metrics['avg_epsilon']
                epsilon_history.append(epsilon)
                
                # Extract noise multiplier values if available
                avg_noise_multiplier = 0.0
                noise_count = 0
                for client in selected_clients:
                    if hasattr(client, 'noise_multiplier') and client.noise_multiplier is not None:
                        avg_noise_multiplier += client.noise_multiplier
                        noise_count += 1
                
                if noise_count > 0:
                    avg_noise_multiplier /= noise_count
                    noise_multiplier_history.append(avg_noise_multiplier)
                    logging.info(f"Round {round_idx+1} completed. "
                                f"Accuracy: {round_metrics['accuracy']:.2f}%, "
                                f"Loss: {round_metrics['loss']:.4f}, "
                                f"Epsilon: {epsilon:.4f}, "
                                f"Noise: {avg_noise_multiplier:.4f}, "
                                f"Time: {round_metrics['computation_time']:.2f}s")
                else:
                    # If no noise multiplier is available, use a default or fixed value
                    noise_multiplier_history.append(noise_multiplier if noise_multiplier is not None else 0.3)
                    logging.info(f"Round {round_idx+1} completed. "
                                f"Accuracy: {round_metrics['accuracy']:.2f}%, "
                                f"Loss: {round_metrics['loss']:.4f}, "
                                f"Epsilon: {epsilon:.4f}, "
                                f"Time: {round_metrics['computation_time']:.2f}s")
            else:
                # If no epsilon data, still track a default noise multiplier
                noise_multiplier_history.append(noise_multiplier if noise_multiplier is not None else 0.3)
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
    
    # Report final privacy budget if available
    if epsilon_history:
        logging.info(f"Final privacy budget spent (epsilon): {epsilon_history[-1]:.4f}")
    else:
        logging.info("No privacy budget information available.")
    
    # Save a copy of the trained model for MIA evaluation
    trained_model = SmallCNN()
    trained_model.load_state_dict(global_model.state_dict())
    
    # Evaluate privacy with M
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
    
    # Plot results
    plot_dp_results(results, "FL with Sequential HE+DP")
    
    return results

if __name__ == "__main__":
    # Run with default parameters when executed directly
    run_sequential_hybrid_experiment() 