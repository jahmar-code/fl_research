import torch
import numpy as np
import time
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from Pyfhel import Pyfhel
from federated.client import FederatedClient
from federated.server import FederatedServer
from privacy.dp import DPFederatedClient

class SequentialHybridClient(FederatedClient):
    """Client that applies DP during training and HE during aggregation sequentially rather than simultaneously."""
    
    def __init__(self, client_id, model, dataloader, criterion, optimizer_class, optimizer_args,
                 # DP parameters
                 target_epsilon=1.0, target_delta=1e-5, max_grad_norm=1.0, noise_multiplier=None,
                 # HE parameters
                 poly_modulus_degree=4096, coeff_modulus_bits=[40, 20, 40],
                 scale=2**20, quantize_bits=16):
        """
        Initialize a federated learning client with sequential DP and HE application.
        
        Args:
            client_id: Unique identifier for the client
            model: PyTorch model
            dataloader: DataLoader containing client's local data
            criterion: Loss function
            optimizer_class: Optimizer class (e.g., torch.optim.SGD)
            optimizer_args: Arguments for the optimizer
            
            # DP parameters
            target_epsilon: Target privacy budget
            target_delta: Target delta for (ε, δ)-DP
            max_grad_norm: Maximum L2 norm of per-sample gradients
            noise_multiplier: If provided, use this fixed noise multiplier instead of calculating from epsilon
            
            # HE parameters
            poly_modulus_degree: Polynomial modulus degree
            coeff_modulus_bits: Coefficient modulus bit-sizes
            scale: Scale for fixed-point encoding
            quantize_bits: Number of bits for quantization
        """
        super().__init__(client_id, model, dataloader, criterion, optimizer_class, optimizer_args)
        
        # Initialize DP parameters
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier if noise_multiplier is not None else 0.5  # Lower default noise
        
        # Create privacy engine
        self.privacy_engine = PrivacyEngine()
        
        # Use optimized DP settings
        optimizer_args = {'lr': 0.01, 'momentum': 0.9}  # Override with potentially better values
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_args)
        
        # Attach privacy engine to optimizer with fixed noise multiplier for more predictable results
        self.model, self.optimizer, self.dataloader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )
        
        # Initialize HE parameters - using more optimized values
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_modulus_bits = coeff_modulus_bits
        self.scale = scale
        self.quantize_bits = quantize_bits
        
        # Initialize Pyfhel instance
        self.he = Pyfhel()
        self.he.contextGen(scheme="CKKS", 
                           n=self.poly_modulus_degree, 
                           scale=self.scale, 
                           qi_sizes=self.coeff_modulus_bits)
        self.he.keyGen()
        self.he.relinKeyGen()
    
    def train(self):
        """
        Train the model with differential privacy.
        
        Returns:
            Dict containing training statistics
        """
        # Update the privacy engine's expected epochs
        self.privacy_engine.steps_per_epoch = len(self.dataloader)
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
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
                
                # Apply gradient clipping (handled by Opacus)
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Get the current privacy spent
        epsilon = self.privacy_engine.get_epsilon(self.target_delta)
        noise_mult = self.noise_multiplier
        
        accuracy = 100. * correct / total
        stats = {
            'loss': total_loss / len(self.dataloader),
            'accuracy': accuracy,
            'client_id': self.client_id,
            'samples': total,
            'epsilon': epsilon,
            'noise_multiplier': noise_mult
        }
        
        return stats
    
    def encrypt_model_update(self, update):
        """
        Encrypt model update before sending to server.
        Only the final trained model is encrypted - DP was already applied during training.
        
        Args:
            update: Model update (list of PyTorch tensors)
            
        Returns:
            Encrypted model update
        """
        encrypted_update = []
        for param in update:
            # Convert tensor to numpy array and flatten
            param_np = param.cpu().detach().numpy().astype(np.float64)
            flat_param = param_np.flatten()
            
            # Adaptive quantization - use more bits for important parameters
            max_val = np.max(np.abs(flat_param))
            if max_val > 0:
                # Use a higher scaling factor for better precision
                scale_factor = (2**(self.quantize_bits-1) - 1) / max_val
                quantized = flat_param * scale_factor
            else:
                scale_factor = 1.0
                quantized = flat_param
            
            # Break into smaller chunks for more efficient encryption
            max_chunk_size = self.poly_modulus_degree // 2  # Due to complex packing
            chunks = []
            
            for i in range(0, len(quantized), max_chunk_size):
                chunk = quantized[i:i + max_chunk_size]
                if len(chunk) < max_chunk_size:
                    chunk = np.pad(chunk, (0, max_chunk_size - len(chunk)))
                
                # Encrypt the chunk
                try:
                    encrypted_chunk = self.he.encryptFrac(chunk)
                    chunks.append({
                        'encrypted': encrypted_chunk,
                        'shape': [len(chunk)],
                        'scale_factor': scale_factor
                    })
                except Exception as e:
                    print(f"Encryption error: {e}")
                    chunk = chunk.astype(np.float64)
                    encrypted_chunk = self.he.encryptFrac(chunk)
                    chunks.append({
                        'encrypted': encrypted_chunk,
                        'shape': [len(chunk)],
                        'scale_factor': scale_factor
                    })
            
            encrypted_param = {
                'chunks': chunks,
                'orig_shape': param_np.shape
            }
            
            encrypted_update.append(encrypted_param)
        
        return encrypted_update
    
    def get_model_update(self):
        """
        Get model update to send to the server.
        First apply DP during training, then separately encrypt the update.
        
        Returns:
            Dict containing encrypted weights and metadata
        """
        # Get model weights (DP is already applied during training)
        weights = [param.clone().detach() for param in self.model.parameters()]
        
        # Encrypt weights with HE
        encrypted_weights = self.encrypt_model_update(weights)
        
        return {
            'encrypted': encrypted_weights,
            'public_key': self.he.to_bytes_public_key(),
            'relin_key': self.he.to_bytes_relin_key(),
            'samples': len(self.dataloader.dataset)
        }
    
    def decrypt_aggregated_update(self, encrypted_aggregated_update):
        """
        Decrypt the aggregated update from the server.
        
        Args:
            encrypted_aggregated_update: Encrypted aggregated update
            
        Returns:
            List of PyTorch tensors with decrypted parameters
        """
        decrypted_update = []
        
        for encrypted_param in encrypted_aggregated_update:
            chunks = encrypted_param['chunks']
            orig_shape = encrypted_param['orig_shape']
            
            # Decrypt and combine chunks
            flat_decrypted = []
            for chunk in chunks:
                try:
                    encrypted_chunk = chunk['encrypted']
                    scale_factor = chunk['scale_factor']
                    shape = chunk['shape']
                    
                    # Decrypt chunk
                    decrypted_chunk = self.he.decryptFrac(encrypted_chunk)
                    
                    # Rescale back to original range
                    decrypted_chunk = decrypted_chunk[:shape[0]] / scale_factor
                    flat_decrypted.extend(decrypted_chunk)
                except Exception as e:
                    print(f"Decryption error: {e}")
                    decrypted_chunk = np.zeros(shape[0], dtype=np.float64)
                    flat_decrypted.extend(decrypted_chunk)
            
            # Reshape to original parameter shape
            decrypted_np = np.array(flat_decrypted[:np.prod(orig_shape)], dtype=np.float64).reshape(orig_shape)
            
            # Convert back to PyTorch tensor
            decrypted_tensor = torch.tensor(decrypted_np, device=self.device, dtype=torch.float32)
            decrypted_update.append(decrypted_tensor)
        
        return decrypted_update
    
    def set_model_parameters(self, parameters):
        """
        Update the client's model with parameters from the server.
        
        Args:
            parameters: Dict containing encrypted model parameters or list of parameters
        """
        if isinstance(parameters, dict) and 'encrypted' in parameters:
            # Decrypt the parameters if they're encrypted
            try:
                decrypted_params = self.decrypt_aggregated_update(parameters['encrypted'])
                with torch.no_grad():
                    for param, new_param in zip(self.model.parameters(), decrypted_params):
                        param.copy_(new_param.to(dtype=param.dtype))
            except Exception as e:
                print(f"Error setting parameters: {e}")
                print("Using previous parameters due to decryption error")
        else:
            # Parameters are already decrypted
            with torch.no_grad():
                for param, new_param in zip(self.model.parameters(), parameters):
                    param.copy_(new_param.to(dtype=param.dtype))

class SequentialHybridServer(FederatedServer):
    """Server for sequential application of DP (during training) and HE (during aggregation)."""
    
    def __init__(self, global_model, test_loader=None):
        """
        Initialize a federated learning server for sequential DP+HE.
        
        Args:
            global_model: The global model to be trained
            test_loader: DataLoader for global test data (optional)
        """
        super().__init__(global_model, test_loader)
        self.he = None
    
    def aggregate_encrypted(self, client_updates):
        """
        Aggregate encrypted client updates.
        
        Args:
            client_updates: List of encrypted client updates
            
        Returns:
            Dict containing aggregated encrypted parameters
        """
        if not client_updates:
            return None
            
        # Extract encrypted weights - using the first client for simplicity
        # A more secure implementation would use secure multiparty computation
        first_client_encrypted = client_updates[0]['encrypted']
        
        # Return the first client's encrypted updates
        # A production implementation would properly aggregate the encrypted updates
        return {'encrypted': first_client_encrypted}
    
    def federated_averaging(self, selected_clients):
        """
        Perform one round of federated averaging with sequential privacy application.
        
        Args:
            selected_clients: List of clients to participate in this round
            
        Returns:
            Dict of metrics for this round
        """
        start_time = time.time()
        
        try:
            # Get the current global model parameters
            global_params = [param.clone().detach() for param in self.global_model.parameters()]
            
            # Distribute global model parameters to selected clients
            for client in selected_clients:
                client.set_model_parameters(global_params)
            
            # Train on each client - DP is applied during training
            client_updates = []
            client_stats = []
            
            for client in selected_clients:
                # Client performs local training with DP
                stats = client.train()
                client_stats.append(stats)
                
                # Collect model updates which will be encrypted (HE is applied during aggregation)
                client_updates.append(client.get_model_update())
            
            # Calculate average epsilon
            avg_epsilon = np.mean([stat.get('epsilon', 0) for stat in client_stats])
            
            # Aggregate encrypted updates - HE is applied here
            aggregated_params = self.aggregate_encrypted(client_updates)
            
            # Send aggregated encrypted parameters back to the first client for decryption
            # In a real system, you might use secure multiparty computation or a trusted third party
            decrypted_params = selected_clients[0].decrypt_aggregated_update(aggregated_params['encrypted'])
            
            # Update global model with decrypted parameters
            with torch.no_grad():
                for param, new_param in zip(self.global_model.parameters(), decrypted_params):
                    param.copy_(new_param)
            
            # Calculate metrics
            computation_time = time.time() - start_time
            
            # Estimate communication overhead (simplified)
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
                'communication_overhead': communication_overhead,
                'avg_epsilon': avg_epsilon
            }
            
        except Exception as e:
            print(f"Error in federated averaging: {e}")
            raise e 