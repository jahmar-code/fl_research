import numpy as np
import torch
from Pyfhel import Pyfhel
from federated.client import FederatedClient
from federated.server import FederatedServer

class HEFederatedClient(FederatedClient):
    """Client with homomorphic encryption capabilities."""
    
    def __init__(self, client_id, model, dataloader, criterion, optimizer_class, optimizer_args,
                 poly_modulus_degree=4096, coeff_modulus_bits=[40, 20, 40],
                 scale=2**20, quantize_bits=16):
        """
        Initialize a federated learning client with homomorphic encryption.
        
        Args:
            client_id: Unique identifier for the client
            model: PyTorch model
            dataloader: DataLoader containing client's local data
            criterion: Loss function
            optimizer_class: Optimizer class (e.g., torch.optim.SGD)
            optimizer_args: Arguments for the optimizer
            poly_modulus_degree: Polynomial modulus degree
            coeff_modulus_bits: Coefficient modulus bit-sizes
            scale: Scale for fixed-point encoding
            quantize_bits: Number of bits for quantization
        """
        super().__init__(client_id, model, dataloader, criterion, optimizer_class, optimizer_args)
        
        # Initialize HE parameters
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
        
    def encrypt_model_update(self, update):
        """
        Encrypt model update before sending to server.
        
        Args:
            update: Model update (list of numpy arrays)
            
        Returns:
            Encrypted model update
        """
        encrypted_update = []
        for param in update:
            # Flatten parameter
            flat_param = param.flatten()
            
            # Quantize to integers (CKKS works better with larger values)
            max_val = np.max(np.abs(flat_param))
            if max_val > 0:
                scale_factor = (2**(self.quantize_bits-1) - 1) / max_val
                quantized = flat_param * scale_factor
            else:
                scale_factor = 1.0
                quantized = flat_param
            
            # Reshape for encryption
            # Break into smaller chunks if necessary (CKKS has limits on vector size)
            max_chunk_size = self.poly_modulus_degree // 2  # Due to complex packing
            chunks = []
            
            for i in range(0, len(quantized), max_chunk_size):
                chunk = quantized[i:i + max_chunk_size]
                # Pad to ensure consistent size
                if len(chunk) < max_chunk_size:
                    chunk = np.pad(chunk, (0, max_chunk_size - len(chunk)))
                
                # Encrypt the chunk
                encrypted_chunk = self.he.encryptFrac(chunk)
                chunks.append({
                    'encrypted': encrypted_chunk,
                    'shape': [len(chunk)],
                    'scale_factor': scale_factor
                })
            
            # Store the original shape for decryption
            encrypted_param = {
                'chunks': chunks,
                'orig_shape': param.shape
            }
            
            encrypted_update.append(encrypted_param)
        
        return encrypted_update
    
    def get_model_update(self):
        """
        Get encrypted model update to send to the server.
        
        Returns:
            Encrypted model weights (parameters)
        """
        # Get model weights
        weights = super().get_model_update()
        
        # Encrypt weights
        encrypted_weights = self.encrypt_model_update(weights)
        
        return {
            'weights': encrypted_weights,
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
            Decrypted model parameters
        """
        decrypted_update = []
        
        for encrypted_param in encrypted_aggregated_update:
            chunks = encrypted_param['chunks']
            orig_shape = encrypted_param['orig_shape']
            
            # Decrypt and combine chunks
            flat_decrypted = []
            for chunk in chunks:
                encrypted_chunk = chunk['encrypted']
                scale_factor = chunk['scale_factor']
                shape = chunk['shape']
                
                # Decrypt chunk
                decrypted_chunk = self.he.decryptFrac(encrypted_chunk)
                
                # Rescale back to original range
                decrypted_chunk = decrypted_chunk[:shape[0]] / scale_factor
                flat_decrypted.extend(decrypted_chunk)
            
            # Reshape to original parameter shape
            decrypted_param = np.array(flat_decrypted[:np.prod(orig_shape)]).reshape(orig_shape)
            decrypted_update.append(decrypted_param)
        
        return decrypted_update
    
    def set_model_parameters(self, parameters):
        """
        Update local model with parameters from the server.
        
        Args:
            parameters: Encrypted model parameters
        """
        if isinstance(parameters, dict) and 'encrypted' in parameters:
            # Decrypt the parameters if they're encrypted
            decrypted_params = self.decrypt_aggregated_update(parameters['encrypted'])
            super().set_model_parameters(decrypted_params)
        else:
            # Parameters are already decrypted
            super().set_model_parameters(parameters)


class HEFederatedServer(FederatedServer):
    """Server with homomorphic encryption capabilities."""
    
    def __init__(self, global_model, test_loader=None):
        """
        Initialize a federated learning server with HE support.
        
        Args:
            global_model: The global model to be trained
            test_loader: DataLoader for global test data (optional)
        """
        super().__init__(global_model, test_loader)
        
    def aggregate_encrypted(self, client_updates):
        """
        Aggregate encrypted client updates using homomorphic encryption.
        
        Args:
            client_updates: List of encrypted client updates
            
        Returns:
            Aggregated encrypted model
        """
        # Extract encrypted weights and sample counts
        encrypted_weights_list = [update['weights'] for update in client_updates]
        samples_list = [update['samples'] for update in client_updates]
        total_samples = sum(samples_list)
        
        # Load the public key and relin key from the first client
        # (all clients should have compatible keys for this to work)
        he = Pyfhel()
        he.from_bytes_public_key(client_updates[0]['public_key'])
        he.from_bytes_relin_key(client_updates[0]['relin_key'])
        
        # Initialize aggregated weights
        aggregated_weights = []
        
        # For each parameter
        for param_idx in range(len(encrypted_weights_list[0])):
            # Get all client chunks for this parameter
            param_chunks = []
            for client_weights in encrypted_weights_list:
                param_chunks.append(client_weights[param_idx]['chunks'])
            
            # Initialize aggregated parameter
            orig_shape = encrypted_weights_list[0][param_idx]['orig_shape']
            aggregated_param = {
                'chunks': [],
                'orig_shape': orig_shape
            }
            
            # For each chunk
            for chunk_idx in range(len(param_chunks[0])):
                # Extract all client chunks at this index
                client_chunks = [client_param_chunks[chunk_idx] for client_param_chunks in param_chunks]