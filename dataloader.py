import networkx as nx
from icecream import ic
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data
import torch_geometric.utils as tf_utils
from torch_geometric.datasets import TUDataset
import networkx as nx


DATASET = {'MUTAG': {'graphs': 188, 'features': 7, 'classes': 2},
           'ENZYMES': {'graphs': 600, 'features': 3, 'classes': 6},
           'PROTEINS': {'graphs': 1113, 'features': 3, 'classes': 2}}


class DataGraph(Dataset):
    def __init__(self, 
                 dataset_name: str, 
                 mode: str
                 ) -> None:
        """ charge les données du dataset """
        
        ic(dataset_name)
        assert dataset_name in DATASET.keys(), f"datasetname must be {list(DATASET.keys())} but is {dataset_name} "
        assert mode in ['train', 'val', 'test'], f"mode must be train, val or test but mode is {mode}"

        self.mode = mode
        dataset = TUDataset(root='data', name=dataset_name)
        self.data = self.convert_to_networkx(dataset)
        self.split_data()
        
        self.num_features = DATASET[dataset_name]['features']
        self.num_classes = DATASET[dataset_name]['classes']

    
    def __len__(self) -> int:
        """ renvoie le nombre de données """
        return len(self.data)

    def __getitem__(self, 
                    index: int
                    ) -> Tuple[Data, torch.Tensor]:
        """ renvoie le graph[index] sous la forme Data(x=nodes_features, edge_index=edge_index)
            et le label y en one hot encoding """
        g = self.data[index]

        # edge index
        edge_index = torch.tensor(list(g.edges)).t().contiguous()

        # nodes features
        nodes_features = []
        for i in range(nx.number_of_nodes(g)):
            nodes_features.append(g.nodes[i]['x'])
        nodes_features = torch.tensor(nodes_features)

        y = torch.tensor(g.graph["y"][0])
        y = torch.nn.functional.one_hot(y, num_classes=self.num_classes)

        data = Data(x=nodes_features, edge_index=edge_index)

        return data, y
    

    def split_data(self) -> None:
        n = len(self.data)
        split1 = int(0.6 * n)
        split2 = int(0.8 * n)
        if self.mode == 'train':
            self.data = self.data[:split1]
        if self.mode == 'val':
            self.data = self.data[split1:split2]
        if self.mode == 'test':
            self.data = self.data[split2:]
    

    def convert_to_networkx(self, 
                            dataset: TUDataset
                            ) -> List[nx.classes.graph.Graph]:
        """
        Conversion des données PyTorch en graphes NetworkX
        :param dataset: le jeu de données PyTorch geometric
        :return: les graphes convertis
        """
        graphs = []
        for idx in range(len(dataset)):
            g = tf_utils.to_networkx(dataset[idx], 
                                     node_attrs=["x"],
                                     to_undirected=True, 
                                     graph_attrs=["y"])
            graphs.append(g)
        return graphs
    

if __name__ == '__main__':
    generator = DataGraph(dataset_name='ENZYMES', mode='train')
    ic(len(generator))
    data, y = generator.__getitem__(3)
    nodes_features = data.x
    edge_index = data.edge_index
    ic(nodes_features.shape)
    ic(edge_index.shape)
    ic(y)
    ic(y.dtype)
    ic(y.shape)
    ic(type(data))
