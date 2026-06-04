import torch
import torch.nn as nn

class VanillaMLP(nn.Module):
    """
    A vanilla Multi-Layer Perceptron (MLP) with ReLU activation.
    """
    def __init__(self, in_features, hid_features, out_features, num_layers=2):
        """
        Args:
            in_features (int): Number of input features.
            hid_features (int): Number of hidden features.
            out_features (int): Number of output features.
            num_layers (int): Total number of linear layers. Must be >= 2.
        """
        super(VanillaMLP, self).__init__()
        
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
            
        layers = []
        
        # Input layer
        layers.append(nn.Linear(in_features, hid_features))
        layers.append(nn.ReLU())
        
        # Intermediate hidden layers (if any)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hid_features, hid_features))
            layers.append(nn.ReLU())
            
        # Output layer
        layers.append(nn.Linear(hid_features, out_features))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
