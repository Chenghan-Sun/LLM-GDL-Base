#!/usr/bin/env python3

import torch
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, global_add_pool
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class CustomGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        # Perform convolution by applying kernel_size
        return torch.matmul(adj_t, x) * self.kernel_size

    def aggregate(self, inputs, index, ptr, dim_size):
        return global_mean_pool(inputs, index, ptr)

    def update(self, aggr_out):
        return aggr_out

    def propagate(self, edge_index, size=None, **kwargs):
        kwargs['edge_index'] = edge_index
        return super().propagate(**kwargs)


class GCN_Net2(nn.Module):
    def __init__(self):
        super(GCN_Net2, self).__init__()
        self.gconv1 = CustomGCNConv(1, 64)
        self.gconv2 = CustomGCNConv(64, 64)
        self.gconv3 = CustomGCNConv(64, 64)
        self.gconv4 = CustomGCNConv(64, 64)
        self.fc1 = nn.Linear(64, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(64)
        # self.dropout = nn.Dropout(p=0.9)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index 
        x = F.relu(self.bn1(self.gconv1(x, edge_index)))  # add BatchNorm1d 
        x = F.relu(self.bn2(self.gconv2(x, edge_index)))  # add BatchNorm1d 
        x = F.relu(self.bn3(self.gconv3(x, edge_index)))  # add BatchNorm1d 
        x = F.relu(self.bn4(self.gconv4(x, edge_index)))  # add BatchNorm1d 
        x = global_mean_pool(x, data.batch)
        # x = self.dropout(x)  # add dropout layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GCN_Net1(nn.Module):
    def __init__(self):
        super(GCN_Net1, self).__init__()
        self.gconv1 = GCNConv(1, 64)
        self.gconv2 = GCNConv(64, 64)
        self.gconv3 = GCNConv(64, 64)
        self.gconv4 = GCNConv(64, 64)
        self.fc1 = nn.Linear(64, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(64)
        # self.dropout = nn.Dropout(p=0.9)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index 
        x = F.relu(self.bn1(self.gconv1(x, edge_index)))  # add BatchNorm1d 
        x = F.relu(self.bn2(self.gconv2(x, edge_index)))  # add BatchNorm1d 
        x = F.relu(self.bn3(self.gconv3(x, edge_index)))  # add BatchNorm1d 
        x = F.relu(self.bn4(self.gconv4(x, edge_index)))  # add BatchNorm1d 
        x = global_mean_pool(x, data.batch)
        # x = self.dropout(x)  # add dropout layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# class GAT(nn.Module):
#     def __init__(self):
#         super(GAT, self).__init__():
#         self.gconv1 = 

