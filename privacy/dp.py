import torch
import numpy as np
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from federated.client import FederatedClient

class DPFederatedClient(FederatedClient):
    """Client with differential privacy capabilities."""
    
    def __init__(self, client_id, model, dataloader, criterion, optimizer_class, optimizer_args,
                 target_epsilon=1.0, target_delta=1e-5, max_grad_norm=1.0, noise_multiplier=None):
        """
        Initialize a federated learning client with differential privacy.
        
        Args:
            client_id: Unique identifier for the client
            model: PyTorch model
            dataloader: DataLoader containing client's local data
            criterion: Loss function
            optimizer_class: Optimizer class (e.g., torch.optim.SGD)
            optimizer_args: Arguments for the optimizer
            target_epsilon: Target privacy budget
            target_delta: Target delta for (ε, δ)-DP
            max_grad_norm: Maximum L2 norm of per-sample gradients
            noise_multiplier: If provided, use this fixed noise multiplier instead of calculating from epsilon
        """
        super().__init__(client_id, model, dataloader, criterion, optimizer_class, optimizer_args)
        
        # Initialize DP parameters
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        
        # Create privacy engine
        self.privacy_engine = PrivacyEngine()
        
        if noise_multiplier is not None:
            # Use a fixed noise multiplier for stronger privacy guarantees
            self.model, self.optimizer, self.dataloader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.dataloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )
        else:
            # Attach privacy engine to optimizer based on epsilon target
            self.model, self.optimizer, self.dataloader = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.dataloader,
                epochs=1,  # We'll reset this in the train method based on actual epochs
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=True,  # Enable Poisson sampling for better privacy guarantees
            )
    
    def train(self, epochs=1):
        """
        Train the model with differential privacy.
        
        Args:
            epochs: Number of local epochs
            
        Returns:
            Dict containing training statistics
        """
        # Update the privacy engine's expected epochs
        self.privacy_engine.steps_per_epoch = len(self.dataloader)
        self.privacy_engine.target_delta = self.target_delta
        self.privacy_engine.target_epsilon = self.target_epsilon
        self.privacy_engine.epochs = epochs
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            with BatchMemoryManager(
                data_loader=self.dataloader,
                max_physical_batch_size=64,  # Adjust based on memory constraints
                optimizer=self.optimizer
            ) as memory_safe_data_loader:
                for inputs, targets in memory_safe_data_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    
                    # Apply gradient clipping (handled by Opacus but adding this for clarity)
                    if not hasattr(self.optimizer, 'step_with_privacy'):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            total_loss += epoch_loss / len(self.dataloader)
        
        # Get the current privacy spent
        epsilon = self.privacy_engine.get_epsilon(self.target_delta)
        noise_mult = getattr(self.privacy_engine, 'noise_multiplier', 0.0)
        
        accuracy = 100. * correct / total
        stats = {
            'loss': total_loss / epochs,
            'accuracy': accuracy,
            'samples': total,
            'epsilon': epsilon,
            'noise_multiplier': noise_mult
        }
        
        return stats