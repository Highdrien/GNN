import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv, SGConv

from typing import List, Dict


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
        self.conv1 = GCNConv(num_features, hidden_layer[0])
        self.conv2 = GCNConv(hidden_layer[0], hidden_layer[1])
        self.linear = nn.Linear(hidden_layer[1], num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                data: Data
                ) -> torch.Tensor:
        node_features, edge_indices = data.x, data.edge_index
        x = self.conv1(node_features, edge_indices)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_indices)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x
    


####################
#        GIN        
####################

class NNModule(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 hidden_layer: int):
        super(NNModule, self).__init__()
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
                 my_nn_module: nn.Module, 
                 epsilon: float):
        super(GINModel, self).__init__()
        self.conv1 = GINConv(my_nn_module, eps=epsilon)

    def forward(self, 
                data: Data
                ) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return x
    
    
####################
#        SGC        
####################

class SGCModel(nn.Module):
    def __init__(self, 
                 num_features: int, 
                 num_classes: int, 
                 hidden_layer: List[int], 
                 dropout: float):
        super(SGCModel, self).__init__()
        self.conv1 = SGConv(num_features, hidden_layer[0])
        self.conv2 = SGConv(hidden_layer[0], hidden_layer[1])
        self.linear = nn.Linear(hidden_layer[1], num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                data: Data
                ) -> torch.Tensor:
        node_features, edge_indices = data.x, data.edge_index
        x = self.conv1(node_features, edge_indices)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_indices)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x



####################
#     GetModel        
####################

def getModel(MODELS: Dict[str, dict],
             model_name: str,
             num_features: int,
             num_classes: int
             ) -> nn.Module:
    
    assert model_name in MODELS.keys(), f"Error, model_name must be in {list(MODELS.keys())} but is {model_name}"
    
    if model_name == 'GCN':
        model = GCNModel(num_features=num_features,
                         num_classes=num_classes,
                         hidden_layer=MODELS['GCN']['hidden_layer'],
                         dropout=MODELS['GCN']['dropout'])
        
    elif model_name == 'SGC':
        model = SGCModel(num_features=num_features,
                         num_classes=num_classes,
                         hidden_layer=MODELS['SGC']['hidden_layer'],
                         dropout=MODELS['SGC']['dropout'])
        
    elif model_name == 'GIN':
        my_nn_module = NNModule(in_channels=num_features, 
                                out_channels=num_classes,
                                hidden_layer=MODELS['GIN']['hidden_layer'])
        model = GINModel(my_nn_module, 
                         epsilon=MODELS['GIN']['epsilon'])
    
    return model
