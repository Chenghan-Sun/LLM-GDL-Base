#!/usr/bin/env python3

import torch
from torch_geometric.nn import GCNConv, global_mean_pool, GINConv, global_add_pool, BatchNorm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GIN_Net(nn.Module):
    def __init__(self):
        super(GIN_Net, self).__init__()
        
        # Define MLP for GINConv layers
        nn1 = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64))
        nn2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
        nn3 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
        nn4 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
        
        # Define GINConv layers with MLPs
        self.gconv1 = GINConv(nn1)
        self.gconv2 = GINConv(nn2)
        self.gconv3 = GINConv(nn3)
        self.gconv4 = GINConv(nn4)
        
        self.bn1 = BatchNorm(64)
        self.bn2 = BatchNorm(64)
        self.bn3 = BatchNorm(64)
        self.bn4 = BatchNorm(64)
        
        self.fc1 = nn.Linear(64, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=True)
        
        # self.dropout = nn.Dropout(p=0.9)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply GINConv layers with BatchNorm and ReLU
        x = F.relu(self.bn1(self.gconv1(x, edge_index)))
        x = F.relu(self.bn2(self.gconv2(x, edge_index)))
        x = F.relu(self.bn3(self.gconv3(x, edge_index)))
        x = F.relu(self.bn4(self.gconv4(x, edge_index)))
        x = global_mean_pool(x, data.batch)
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class ModifiedGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(ModifiedGCNConv, self).__init__(in_channels, out_channels, **kwargs)
        self.stride = stride

    def forward(self, x, edge_index, batch):
        # print(f"  Input to ModifiedGCNConv - x: {x.shape}, edge_index: {edge_index.shape}, batch: {batch.shape}")

        x = super(ModifiedGCNConv, self).forward(x, edge_index)
        # print(f"  After GCNConv - x: {x.shape}")

        if self.stride > 1:
            original_num_nodes = x.size(0)
            num_nodes = (original_num_nodes + self.stride - 1) // self.stride  # Ceiling division
            pad_size = num_nodes * self.stride - original_num_nodes
            if pad_size > 0:
                pad_x = torch.zeros(pad_size, x.size(1)).to(x.device)
                x = torch.cat([x, pad_x], dim=0)
                pad_batch = torch.full((pad_size,), batch.max() + 1, dtype=torch.long).to(batch.device)
                batch = torch.cat([batch, pad_batch], dim=0)
                # print(f"  After stride padding - x: {x.shape}, batch: {batch.shape}")

            x = x.view(-1, self.stride, x.size(1)).mean(dim=1)  # Downsample nodes by stride
            batch = batch[::self.stride]  # Downsample batch tensor by stride
            num_nodes = x.size(0)

            mask = (edge_index[0] % self.stride == 0) & (edge_index[1] % self.stride == 0)
            edge_index = edge_index[:, mask]
            edge_index = edge_index // self.stride

            valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            edge_index = edge_index[:, valid_mask]

            # print(f"  After stride - x: {x.shape}, edge_index: {edge_index.shape}, batch: {batch.shape}")

        assert x.size(1) == self.out_channels, \
            f"  Output dimension mismatch: expected {self.out_channels}, got {x.size(1)}"
        # print(f"  Output from ModifiedGCNConv - x: {x.shape}")

        return x, edge_index, batch

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

class GCN_Net3(nn.Module):
    def __init__(self):
        super(GCN_Net3, self).__init__()
        self.gconv1 = ModifiedGCNConv(1, 64, stride=2)
        self.gconv2 = ModifiedGCNConv(64, 64, stride=2)
        self.gconv3 = ModifiedGCNConv(64, 64, stride=2)
        self.gconv4 = ModifiedGCNConv(64, 64, stride=2)
        self.fc1 = nn.Linear(64, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=True)
        self.bn1 = BatchNorm(64)
        self.bn2 = BatchNorm(64)
        self.bn3 = BatchNorm(64)
        self.bn4 = BatchNorm(64)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # print(f"Initial input - x: {x.shape}, edge_index: {edge_index.shape}, batch: {batch.shape}")

        x, edge_index, batch = self.gconv1(x, edge_index, batch)
        x = F.relu(self.bn1(x))
        # print(f"After gconv1 - x: {x.shape}, edge_index: {edge_index.shape}, batch: {batch.shape}")

        x, edge_index, batch = self.gconv2(x, edge_index, batch)
        x = F.relu(self.bn2(x))
        # print(f"After gconv2 - x: {x.shape}, edge_index: {edge_index.shape}, batch: {batch.shape}")

        x, edge_index, batch = self.gconv3(x, edge_index, batch)
        x = F.relu(self.bn3(x))
        # print(f"After gconv3 - x: {x.shape}, edge_index: {edge_index.shape}, batch: {batch.shape}")

        x, edge_index, batch = self.gconv4(x, edge_index, batch)
        x = F.relu(self.bn4(x))
        # print(f"After gconv4 - x: {x.shape}, edge_index: {edge_index.shape}, batch: {batch.shape}")

        assert x.size(1) == 64, f"Output dimension mismatch: expected 64, got {x.size(1)}"

        # Ensure unique and sequential batch indices
        unique_batch = torch.unique(batch)
        new_batch = torch.arange(unique_batch.size(0), device=batch.device).repeat_interleave(torch.bincount(batch))
        # print(f"After remapping batch indices - new_batch: {new_batch.shape}")

        # Perform global mean pooling
        x = global_mean_pool(x, new_batch)
        # print(f"After global mean pool - x: {x.shape}")

        x = F.relu(self.fc1(x))
        # print(f"After fc1 - x: {x.shape}")

        x = self.fc2(x)
        # print(f"After fc2 - x: {x.shape}")

        return x


class GCN_Net2(nn.Module):
    def __init__(self):
        super(GCN_Net2, self).__init__()
        self.gconv1 = CustomGCNConv(1, 64)
        self.gconv2 = CustomGCNConv(64, 64)
        self.gconv3 = CustomGCNConv(64, 64)
        self.gconv4 = CustomGCNConv(64, 64)
        self.fc1 = nn.Linear(64, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=True)
        self.bn1 = BatchNorm(64)
        self.bn2 = BatchNorm(64)
        self.bn3 = BatchNorm(64)
        self.bn4 = BatchNorm(64)
        # self.dropout = nn.Dropout(p=0.9)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index 
        x = F.relu(self.bn1(self.gconv1(x, edge_index)))  # add BatchNorm 
        x = F.relu(self.bn2(self.gconv2(x, edge_index)))  # add BatchNorm 
        x = F.relu(self.bn3(self.gconv3(x, edge_index)))  # add BatchNorm 
        x = F.relu(self.bn4(self.gconv4(x, edge_index)))  # add BatchNorm 
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
        self.bn1 = BatchNorm(64)
        self.bn2 = BatchNorm(64)
        self.bn3 = BatchNorm(64)
        self.bn4 = BatchNorm(64)
        # self.dropout = nn.Dropout(p=0.9)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index 
        x = F.relu(self.bn1(self.gconv1(x, edge_index)))  # add BatchNorm 
        x = F.relu(self.bn2(self.gconv2(x, edge_index)))  # add BatchNorm 
        x = F.relu(self.bn3(self.gconv3(x, edge_index)))  # add BatchNorm 
        x = F.relu(self.bn4(self.gconv4(x, edge_index)))  # add BatchNorm 
        x = global_mean_pool(x, data.batch)
        # x = self.dropout(x)  # add dropout layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# class GAT(nn.Module):
#     def __init__(self):
#         super(GAT, self).__init__():
#         self.gconv1 = 

