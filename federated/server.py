import numpy as np
import copy
import torch
import time
import logging
from collections import OrderedDict

class FederatedServer:
    """Base class for a federated learning server/aggregator."""
    
    def __init__(self, global_model, test_loader=None):
        """
        Initialize a federated learning server.
        
        Args:
            global_model: The global model to be trained
            test_loader: DataLoader for global test data (optional)
        """
        self.global_model = global_model
        self.test_loader = test_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.global_model.to(self.device)
        self.current_round = 0
        self.clients = []
        self.metrics = {
            'accuracy': [],
            'loss': [],
            'communication_overhead': [],
            'computation_time': []
        }
        
    def add_client(self, client):
        """Add a client to the federation."""
        self.clients.append(client)
        
    def select_clients(self, num_clients=None):
        """
        Select a subset of clients to participate in a training round.
        
        Args:
            num_clients: Number of clients to select (default: all clients)
            
        Returns:
            List of selected clients
        """
        if num_clients is None or num_clients >= len(self.clients):
            return self.clients
        
        # Randomly select clients without replacement
        selected_indices = np.random.choice(len(self.clients), num_clients, replace=False)
        return [self.clients[i] for i in selected_indices]
    
    def aggregate(self, client_updates, client_samples):
        """
        Aggregate client updates using weighted averaging (FedAvg).
        
        Args:
            client_updates: List of client model updates
            client_samples: List of number of samples per client
            
        Returns:
            Aggregated model parameters
        """
        # Calculate the total number of samples
        total_samples = sum(client_samples)
        
        # Initialize the aggregated parameters with zeros like the first update
        aggregated_params = [np.zeros_like(param) for param in client_updates[0]]
        
        # Perform weighted averaging
        for client_idx, update in enumerate(client_updates):
            weight = client_samples[client_idx] / total_samples
            for param_idx, param in enumerate(update):
                aggregated_params[param_idx] += param * weight
        
        return aggregated_params
    
    def federated_averaging(self, selected_clients):
        """
        Perform one round of federated averaging.
        
        Args:
            selected_clients: List of clients to participate in this round
            
        Returns:
            Dict of metrics for this round
        """
        start_time = time.time()
        
        # Get the current global model parameters
        global_params = [param.clone().detach() for param in self.global_model.parameters()]
        
        # Distribute global model parameters to selected clients
        for client in selected_clients:
            client.set_model_parameters(global_params)
        
        # Train on each client
        client_weights = []
        client_stats = []
        client_sample_sizes = []
        
        print(f"Starting training on {len(selected_clients)} selected clients...")
        for i, client in enumerate(selected_clients):
            print(f"Training client {i+1}/{len(selected_clients)} (ID: {client.client_id})...")
            
            # Client performs local training
            stats = client.train()
            client_stats.append(stats)
            
            # Collect model weights and sample sizes for weighted averaging
            weights = [param.clone().detach() for param in client.model.parameters()]
            client_weights.append(weights)
            client_sample_sizes.append(stats['samples'])
            
            print(f"Client {client.client_id} finished training. Accuracy: {stats['accuracy']:.2f}%, Loss: {stats['loss']:.4f}")
        
        print("Client training complete. Performing federated averaging...")
        
        # Perform weighted averaging of client models
        new_params = self.weighted_average(client_weights, client_sample_sizes)
        
        # Update global model with new parameters
        with torch.no_grad():
            for param, new_param in zip(self.global_model.parameters(), new_params):
                param.copy_(new_param)
        
        # Calculate metrics
        computation_time = time.time() - start_time
        
        # Estimate communication overhead
        param_size_bytes = sum(param.numel() * param.element_size() for param in self.global_model.parameters())
        communication_overhead = param_size_bytes * len(selected_clients) * 2  # Upload and download
        
        # Evaluate global model if test data is provided
        if self.test_loader is not None:
            accuracy, loss = self.evaluate()
            print(f"Global model evaluation - Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
        else:
            accuracy, loss = 0, 0
        
        # Store metrics
        self.metrics['accuracy'].append(accuracy)
        self.metrics['loss'].append(loss)
        self.metrics['computation_time'].append(computation_time)
        self.metrics['communication_overhead'].append(communication_overhead)
        
        print(f"Round completed in {computation_time:.2f} seconds")
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'computation_time': computation_time,
            'communication_overhead': communication_overhead
        }
    
    def train(self, num_rounds=10, clients_per_round=None):
        """
        Train the model for a number of rounds.
        
        Args:
            num_rounds: Number of training rounds
            clients_per_round: Number of clients to select per round (default: all)
            
        Returns:
            Dict containing training history
        """
        history = []
        
        print(f"Starting federated training for {num_rounds} rounds")
        for round_idx in range(num_rounds):
            print(f"\n--- Round {round_idx + 1}/{num_rounds} ---")
            
            # Select clients for this round
            if clients_per_round is None:
                selected_clients = self.clients  # Use all clients
            else:
                selected_clients = self.select_clients(clients_per_round)
            
            # Perform one round of federated averaging
            round_metrics = self.federated_averaging(selected_clients)
            round_metrics['round'] = round_idx + 1
            history.append(round_metrics)
            
            self.current_round += 1
            
            print(f"Round {round_idx + 1} completed. Accuracy: {round_metrics['accuracy']:.2f}%, Loss: {round_metrics['loss']:.4f}")
        
        print("Federated training complete!")
        return history
    
    def weighted_average(self, client_weights, client_sample_sizes):
        """
        Perform weighted averaging of client models based on their dataset sizes.
        
        Args:
            client_weights: List of model weights from each client
            client_sample_sizes: List of sample sizes from each client
            
        Returns:
            List of averaged parameters
        """
        # Calculate weighted average based on number of samples
        total_samples = sum(client_sample_sizes)
        
        # Initialize averaged parameters with zeros like the first client's parameters
        avg_params = [torch.zeros_like(param) for param in client_weights[0]]
        
        # Compute weighted average
        for i, weights in enumerate(client_weights):
            weight = client_sample_sizes[i] / total_samples
            for j, param in enumerate(weights):
                avg_params[j] += param * weight
                
        return avg_params
        
    def evaluate(self):
        """
        Evaluate the global model on the test dataset.
        
        Returns:
            Tuple of (accuracy, loss)
        """
        self.global_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.global_model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        loss = test_loss / len(self.test_loader)
        
        return accuracy, loss