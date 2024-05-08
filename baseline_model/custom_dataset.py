import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from scipy.sparse import coo_matrix

import dgl
from dgl.data import DGLDataset

class ProteinGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ProteinGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['all_features.npy'] 

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_dict = np.load(self.raw_paths[0], allow_pickle=True).item()
        
        data_list = []
        for key, sample_data in data_dict.items():
            adjacency_matrix = sample_data['adjacency_matrix']
            edge_index = coo_matrix(adjacency_matrix).nonzero()
            # edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_index = torch.tensor(np.array(edge_index), dtype=torch.long)

            node_features = sample_data['node_features']
            x = torch.tensor(node_features, dtype=torch.float)

            label = torch.tensor([int(sample_data['label'])], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, y=label)
            
            plm_features = sample_data['plm_features']
            
            data.plm_features = plm_features
            data.key = key
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class ProteinGraphDGLDataset(DGLDataset):
    def __init__(self, root):
        self.root = root
        super().__init__(name='protein_graph', url=None, raw_dir=self.root, save_dir=self.root)
        self.load()

    def process(self):
        pyg_dataset = ProteinGraphDataset(root=self.root) 

        self.graphs = []
        self.labels = []

        for data in pyg_dataset:
            g = dgl.graph((data.edge_index[0], data.edge_index[1]))
            g.ndata['feat'] = data.x
            self.labels.append(data.y)
            
            if isinstance(data.plm_features, np.ndarray):
                plm_features = torch.tensor(data.plm_features, dtype=torch.float)
            else:
                plm_features = data.plm_features

            g.ndata['plm_features'] = plm_features
            

            self.graphs.append(g)

        self.labels = torch.tensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)




# class ProteinGraphDataset(InMemoryDataset):
#     def __init__(self, root, transform=None, pre_transform=None):
#         super(ProteinGraphDataset, self).__init__(root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self):
#         return ['all_features.npy']

#     @property
#     def processed_file_names(self):
#         return ['data.pt']

#     def download(self):
#         pass

#     def process(self):
#         data_dict = np.load(self.raw_paths[0], allow_pickle=True).item()
#         data_list = []
#         for key, sample_data in data_dict.items():
#             adjacency_matrix = sample_data['adjacency_matrix']
#             edge_index = coo_matrix(adjacency_matrix).nonzero()
#             edge_index = torch.tensor(np.array(edge_index), dtype=torch.long)
#             node_features = sample_data['node_features']
#             x = torch.tensor(node_features, dtype=torch.float)
#             label = torch.tensor([int(sample_data['label'])], dtype=torch.long)
#             data = Data(x=x, edge_index=edge_index, y=label)
#             plm_features = sample_data['plm_features']
#             data.plm_features = plm_features
#             data.key = key
#             data_list.append(data)
#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])


# class ProteinGraphDGLDataset(DGLDataset):
#     def __init__(self, root):
#         self.root = root
#         super().__init__(name='protein_graph', url=None, raw_dir=self.root, save_dir=self.root)
#         self.load()

#     def process(self):
#         pyg_dataset = ProteinGraphDataset(root=self.root)
#         self.graphs = []
#         self.labels = []
#         self.graph_keys = []

#         for data in pyg_dataset:
#             g = dgl.graph((data.edge_index[0], data.edge_index[1]))
#             g.ndata['feat'] = data.x
#             if isinstance(data.plm_features, np.ndarray):
#                 plm_features = torch.tensor(data.plm_features, dtype=torch.float)
#             else:
#                 plm_features = data.plm_features
#             g.ndata['plm_features'] = plm_features
#             g.graph_key = data.key
#             self.graphs.append(g)
#             self.labels.append(data.y)
#             self.graph_keys.append(data.key)

#         self.labels = torch.tensor([label.item() for label in self.labels])

#     def __getitem__(self, i):
#         g = self.graphs[i]
#         label = self.labels[i]
#         key = self.graph_keys[i]
#         return g, label, key

#     def __len__(self):
#         return len(self.graphs)





