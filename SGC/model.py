import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv
from typing import List

from data_loader import DataGraph

class SGConvModel(nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_layer: List[int], dropout: float):
        super(SGConvModel, self).__init__()
        self.conv1 = SGConv(num_features, hidden_layer[0])
        self.conv2 = SGConv(hidden_layer[0], hidden_layer[1])
        self.linear = nn.Linear(hidden_layer[1], num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                node_features: torch.Tensor, 
                edge_indices: torch.Tensor) -> torch.Tensor:
        """
        input: node features (|V|, F_in), edge indices (2, |V|)
        output: 
        """
        x = self.conv1(node_features, edge_indices)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_indices)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x