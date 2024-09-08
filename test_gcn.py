import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import networkx as nx

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GCNConv

import pandas as pd
import networkx as nx
from torch_geometric.data import Data
import torch
from torch_geometric.utils import train_test_split_edges
import pyarrow.parquet as pq

trips = pq.read_table('trip.parquet')
df = trips.to_pandas()

# Create a directed graph from the dataset
G = nx.from_pandas_edgelist(df, source='PULocationID', target='DOLocationID', create_using=nx.DiGraph())

print(G)

# Generate a mapping from node labels to consecutive integers
node_mapping = {node: i for i, node in enumerate(G.nodes)}
num_nodes = len(node_mapping)

# Create edge index for PyTorch Geometric
edge_index = torch.tensor([[
    node_mapping[source], node_mapping[target]] for source, target in G.edges()], dtype=torch.long).t().contiguous()

# Create a PyTorch Geometric graph data object
data = Data(edge_index=edge_index, num_nodes=num_nodes)
num_features = num_nodes

# Create node features using an identity matrix
node_features = torch.eye(num_nodes)

# Update the data object to include node features
data.x = node_features

# Assuming `data` is your PyTorch Geometric data object
data = train_test_split_edges(data)


class GCN(torch.nn.Module):
    def __init__(self, num_features = 1, num_classes = 1):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 30)
        self.conv2 = GCNConv(30, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

class GCNModel():
    def __init__(self):
        self.model = GCN()

    def fit(self, data, epochs = 100):
        # Assume `data` is your graph data object from Step 1
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.BCELoss()

        loss_list = []
        for epoch in range(epochs):  # Number of epochs
            self.model.train()
            optimizer.zero_grad()

            # Forward pass for positive samples
            pred_pos = self.model(data.x, data.train_pos_edge_index).squeeze()

            # Generate negative samples
            edge_index_neg = negative_sampling(edge_index, num_nodes=data.num_nodes, num_neg_samples=edge_index.size(1))

            # Predictions for negative samples
            pred_neg = self.model(data.x, edge_index_neg).squeeze()

            # Combine positive and negative predictions and create labels accordingly
            pred_all = torch.cat([pred_pos, pred_neg], dim=0)
            labels = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)

            # Calculate loss, perform backpropagation, and update model parameters
            loss = criterion(pred_all, labels)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
