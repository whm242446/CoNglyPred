import os
import copy
import math
import numpy as np
import torch
import torch_geometric
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from scipy.sparse import coo_matrix


class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, output_dim=None):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(num_features, hidden_size))
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_size, hidden_size))

        if self.output_dim is not None:
            self.output_layer = nn.Linear(hidden_size, output_dim)

    def forward(self, x, edge_index):

        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)

        if self.output_dim is not None:
            x = self.output_layer(x)

        return x
   
class GATModel(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, output_dim=None, heads=1):
        super(GATModel, self).__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GATConv(num_features, hidden_size, heads=heads))
        for _ in range(num_layers - 1):
            self.conv_layers.append(GATConv(hidden_size * heads, hidden_size, heads=heads))

        if self.output_dim is not None:
            self.output_layer = nn.Linear(hidden_size * heads, output_dim)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)

        if self.output_dim is not None:
            x = self.output_layer(x)

        return x

