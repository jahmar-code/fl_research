import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """A simple CNN for CIFAR-10 classification."""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_weights(self):
        """Get model weights as a list of numpy arrays."""
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy())
        return weights
    
    def set_weights(self, weights):
        """Set model weights from a list of numpy arrays."""
        for param, weight in zip(self.parameters(), weights):
            param.data = torch.from_numpy(weight).to(param.device)

class SmallCNN(nn.Module):
    """A smaller CNN for CIFAR-10 that works better with homomorphic encryption."""
    
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_weights(self):
        """Get model weights as a list of numpy arrays."""
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy())
        return weights
    
    def set_weights(self, weights):
        """Set model weights from a list of numpy arrays."""
        for param, weight in zip(self.parameters(), weights):
            param.data = torch.from_numpy(weight).to(param.device)