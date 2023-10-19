import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv

from data_loader import DataGraph, train_loader, val_loader, test_loader

class SGConvModel(nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_dim,):
        super(SGConvModel, self).__init__()
        self.conv1 = SGConv(num_features, hidden_dim)
        self.conv2 = SGConv(hidden_dim, num_classes)

    def forward(self, 
                node_features: torch.Tensor, 
                edge_indices: torch.Tensor) -> torch.Tensor:
        """
        input: node features (|V|, F_in), edge indices (2, |V|)
        output: 
        """
        x = self.conv1(node_features, edge_indices)
        x = F.relu(x)
        x = self.conv2(x, edge_indices)
        return x