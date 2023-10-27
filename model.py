import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv, SGConv

from typing import List, Dict, Union


####################
#        GCN        
####################


class GCNModel(nn.Module):
    def __init__(self, 
                 num_features: int, 
                 num_classes: int, 
                 hidden_layer: List[int], 
                 dropout: float
                 ) -> None:
        super(GCNModel, self).__init__()
        self.num_conv = len(hidden_layer)

        self.conv = []
        self.conv.append(GCNConv(num_features, hidden_layer[0]))
        for i in range(1, self.num_conv):
            self.conv.append(GCNConv(hidden_layer[i - 1], hidden_layer[i]))
        
        self.linear = nn.Linear(hidden_layer[-1], num_classes)

    def forward(self, 
                data: Data
                ) -> torch.Tensor:
        x, edge_indices = data.x, data.edge_index

        for i in range(self.num_conv):
            x = self.conv[i](x, edge_indices)
            x = F.relu(x)
        
        x = self.linear(x)
        return x
    
    def __str__(self):
        string = 'model GCN:\n'
        for i in range(self.num_conv):
            string += f'\t{str(self.conv[i])}\n'
        string += f'\t{self.linear}'
        return string
    


####################
#        GIN        
####################

class MLP(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 hidden_layer: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

class GINModel(nn.Module):
    def __init__(self, 
                 num_features: int, 
                 num_classes: int,
                 hidden_layers: List[int],
                 epsilon: float):
        super(GINModel, self).__init__()

        self.n = len(hidden_layers)
        assert self.n in [1, 3], f"len hidden layers must be 1 ou 3, not {self.n}"

        if self.n == 1:
            self.conv = GINConv(MLP(in_channels=num_features,
                                    out_channels=num_classes,
                                    hidden_layer=hidden_layers[0]),
                                    eps=epsilon)
        else:
            self.conv1 = GINConv(MLP(in_channels=num_features,
                                     out_channels=hidden_layers[1],
                                     hidden_layer=hidden_layers[0]),
                                     eps=epsilon)
            self.conv2 = GINConv(MLP(in_channels=hidden_layers[1],
                                     out_channels=num_classes,
                                     hidden_layer=hidden_layers[2]),
                                     eps=epsilon)
            self.relu = nn.ReLU()

    def forward(self, 
                data: Data
                ) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        if self.n == 1:
            x = self.conv(x, edge_index)
        else:
            x = self.conv1(x, edge_index)
            x = self.relu(x)
            x = self.conv2(x, edge_index)
        return x

    
    
####################
#        SGC        
####################

class SGCModel(nn.Module):
    def __init__(self, 
                 num_features: int, 
                 num_classes: int, 
                 hidden_layer: List[int],
                 K: int, 
                 dropout: float):
        super(SGCModel, self).__init__()
        self.num_conv = len(hidden_layer)
        self.conv1 = SGConv(num_features, hidden_layer[0], K=K)
        if self.num_conv == 2:
            self.conv2 = SGConv(hidden_layer[0], hidden_layer[1], K=K)
        self.linear = nn.Linear(hidden_layer[-1], num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, 
                data: Data
                ) -> torch.Tensor:
        node_features, edge_indices = data.x, data.edge_index
        x = self.conv1(node_features, edge_indices)
        x = self.dropout(x)
        if self.num_conv == 2:
            x = self.relu(x)
            x = self.conv2(x, edge_indices)
            x = self.dropout(x)
        x = self.linear(x)

        return x



####################
#     GetModel        
####################

def getModel(model_name: str,
             model_info: Dict[str, Union[int, List[int]]],
             num_features: int,
             num_classes: int
             ) -> nn.Module:
    
    if model_name == 'GCN':
        model = GCNModel(num_features=num_features,
                         num_classes=num_classes,
                         hidden_layer=model_info['hidden_layer'],
                         dropout=model_info['dropout'])
        
    elif model_name == 'SGC':
        model = SGCModel(num_features=num_features,
                         num_classes=num_classes,
                         hidden_layer=model_info['hidden_layer'],
                         dropout=model_info['dropout'],
                         K=model_info['K'])
        
    elif model_name == 'GIN':
        model = GINModel(num_features=num_features,
                         num_classes=num_classes,
                         hidden_layers=model_info['hidden_layer'], 
                         epsilon=float(model_info['epsilon']))
    
    return model
