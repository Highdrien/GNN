from torch_geometric.datasets import TUDataset
import torch_geometric.utils as tf_utils
import networkx as nx
import torch

from icecream import ic


DATASET = {'MUTAG': {'graphs': 188, 'features': 7, 'classes': 2},
           'ENZYMES': {'graphs': 600, 'features': 3, 'classes': 6},
           'PROTEINS': {'graphs': 1113, 'features': 3, 'classes': 2}}


class DataGraph():
    def __init__(self, 
                 dataset_name: int, 
                 mode: int) -> None:
        
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
        return len(self.data)

    def __getitem__(self, index: int):
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
        y = y.to(torch.float32)

        return nodes_features, edge_index, y
    

    def split_data(self):
        n = len(self.data)
        split1 = int(0.6 * n)
        split2 = int(0.8 * n)
        if self.mode == 'train':
            self.data = self.data[:split1]
        if self.mode == 'val':
            self.data = self.data[split1:split2]
        if self.mode == 'test':
            self.data = self.data[split2:]
    
    def convert_to_networkx(self, dataset):
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
    nodes_features, edge_index, y = generator.__getitem__(3)
    ic(nodes_features.shape)
    ic(edge_index.shape)
    ic(y)
    ic(y.dtype)
    ic(y.shape)