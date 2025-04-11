import torch
import numpy as np
import copy
from collections import OrderedDict

class FederatedClient:
    """Base class for a federated learning client."""
    
    def __init__(self, client_id, model, dataloader, criterion, optimizer_class, optimizer_args):
        """
        Initialize a federated learning client.
        
        Args:
            client_id: Unique identifier for the client
            model: PyTorch model
            dataloader: DataLoader containing client's local data
            criterion: Loss function
            optimizer_class: Optimizer class (e.g., torch.optim.SGD)
            optimizer_args: Arguments for the optimizer
        """
        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_args)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def train(self):
        """
        Train the model on the client's local data.
        
        Returns:
            Dict containing training statistics
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        print(f"Client {self.client_id}: Starting local training with {len(self.dataloader)} batches...")
        batch_counter = 0
        
        for inputs, labels in self.dataloader:
            batch_counter += 1
            if batch_counter % 10 == 0:  # Print progress every 10 batches
                print(f"Client {self.client_id}: Processing batch {batch_counter}/{len(self.dataloader)}")
                
            # Move data to appropriate device
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Calculate statistics
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        # Calculate final statistics
        final_loss = total_loss / len(self.dataloader)
        accuracy = 100. * correct / total
        
        return {
            'loss': final_loss,
            'accuracy': accuracy,
            'client_id': self.client_id,
            'samples': total
        }
    
    def get_model_update(self):
        """
        Get model updates to send to the server.
        
        Returns:
            Model weights (parameters)
        """
        return self.model.get_weights()
    
    def set_model_parameters(self, parameters):
        """
        Update the client's model with parameters from the server.
        
        Args:
            parameters: List of model parameters
        """
        with torch.no_grad():
            for param, new_param in zip(self.model.parameters(), parameters):
                param.copy_(new_param)
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: DataLoader containing test data
            
        Returns:
            Dict containing evaluation statistics
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        stats = {
            'loss': test_loss / len(test_loader),
            'accuracy': accuracy,
            'samples': total
        }
        
        return stats