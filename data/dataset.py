import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms

def load_cifar10():
    """Load CIFAR-10 dataset with standard transformations."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

def create_client_data(dataset, num_clients, iid=True, alpha=0.5):
    """
    Split dataset among clients in IID or non-IID fashion.
    
    Args:
        dataset: The dataset to split
        num_clients: Number of clients
        iid: Whether to use IID (True) or non-IID (False) distribution
        alpha: Concentration parameter for Dirichlet distribution (lower = more non-IID)
              Only used when iid=False
    
    Returns:
        List of indices for each client
    """
    num_samples = len(dataset)
    indices = list(range(num_samples))
    
    # Get labels for dataset
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        # Try to extract targets from data
        if hasattr(dataset, 'data') and hasattr(dataset, 'targets'):
            labels = dataset.targets
        else:
            # Last resort: extract labels by iterating through dataset
            labels = []
            for _, label in dataset:
                labels.append(label)
            labels = torch.tensor(labels)
    
    if isinstance(labels, list):
        labels = torch.tensor(labels)
    
    # Split indices according to distribution
    if iid:
        # IID: randomly shuffle indices and split evenly
        np.random.shuffle(indices)
        chunks = np.array_split(indices, num_clients)
        client_indices = [chunk.tolist() for chunk in chunks]
    else:
        # Non-IID: use Dirichlet distribution
        num_classes = len(torch.unique(labels))
        client_indices = [[] for _ in range(num_clients)]
        
        # Sort indices by label
        label_indices = [[] for _ in range(num_classes)]
        for idx, label in enumerate(labels):
            label_indices[label].append(idx)
            
        # Distribute data according to Dirichlet distribution
        diri_dist = np.random.dirichlet(alpha=alpha * np.ones(num_clients), size=num_classes)
        
        # For each class, assign samples to clients according to distribution
        for c in range(num_classes):
            class_indices = label_indices[c]
            np.random.shuffle(class_indices)
            class_size = len(class_indices)
            
            # Calculate how many samples of this class to assign to each client
            client_sample_counts = (np.array(diri_dist[c]) * class_size).astype(int)
            # Adjust counts to ensure all samples are assigned
            diff = class_size - np.sum(client_sample_counts)
            # Add the remaining samples to random clients
            if diff > 0:
                client_sample_counts[np.random.choice(num_clients, diff, replace=False)] += 1
            
            # Distribute samples
            index = 0
            for client_id in range(num_clients):
                count = client_sample_counts[client_id]
                client_indices[client_id].extend(class_indices[index:index+count])
                index += count
    
    # Print data distribution statistics
    print("Data distribution statistics:")
    for client_id, indices in enumerate(client_indices):
        client_labels = [labels[i].item() for i in indices]
        unique_labels, counts = np.unique(client_labels, return_counts=True)
        distribution = {int(label): int(count) for label, count in zip(unique_labels, counts)}
        print(f"Client {client_id}: {len(indices)} samples, label distribution: {distribution}")
    
    return client_indices

def get_dataloaders(dataset, client_indices, batch_size=64):
    """
    Create DataLoader for each client.
    
    Args:
        dataset: The dataset to use
        client_indices: List of indices for each client
        batch_size: Batch size for training
    
    Returns:
        List of DataLoaders for each client
    """
    client_dataloaders = []
    
    for indices in client_indices:
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_dataloaders.append(loader)
        
    return client_dataloaders