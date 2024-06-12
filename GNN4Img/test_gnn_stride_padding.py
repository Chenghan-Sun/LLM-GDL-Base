import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

class ModifiedGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(ModifiedGCNConv, self).__init__(in_channels, out_channels, **kwargs)
        self.stride = stride

    def forward(self, x, edge_index, batch):
        print(f"  Input to ModifiedGCNConv - x: {x.shape}, edge_index: {edge_index.shape}, batch: {batch.shape}")

        x = super(ModifiedGCNConv, self).forward(x, edge_index)
        print(f"  After GCNConv - x: {x.shape}")

        if self.stride > 1:
            original_num_nodes = x.size(0)
            num_nodes = (original_num_nodes + self.stride - 1) // self.stride  # Ceiling division
            pad_size = num_nodes * self.stride - original_num_nodes
            if pad_size > 0:
                pad_x = torch.zeros(pad_size, x.size(1)).to(x.device)
                x = torch.cat([x, pad_x], dim=0)
                pad_batch = torch.full((pad_size,), batch.max() + 1, dtype=torch.long).to(batch.device)
                batch = torch.cat([batch, pad_batch], dim=0)
                print(f"  After stride padding - x: {x.shape}, batch: {batch.shape}")

            x = x.view(-1, self.stride, x.size(1)).mean(dim=1)  # Downsample nodes by stride
            batch = batch[::self.stride]  # Downsample batch tensor by stride
            num_nodes = x.size(0)

            mask = (edge_index[0] % self.stride == 0) & (edge_index[1] % self.stride == 0)
            edge_index = edge_index[:, mask]
            edge_index = edge_index // self.stride

            valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            edge_index = edge_index[:, valid_mask]

            print(f"  After stride - x: {x.shape}, edge_index: {edge_index.shape}, batch: {batch.shape}")

        assert x.size(1) == self.out_channels, \
            f"  Output dimension mismatch: expected {self.out_channels}, got {x.size(1)}"
        print(f"  Output from ModifiedGCNConv - x: {x.shape}")

        return x, edge_index, batch

class GCN_Net3(nn.Module):
    def __init__(self):
        super(GCN_Net3, self).__init__()
        self.gconv1 = ModifiedGCNConv(1, 64, stride=2)
        self.gconv2 = ModifiedGCNConv(64, 64, stride=2)
        self.gconv3 = ModifiedGCNConv(64, 64, stride=3)
        self.gconv4 = ModifiedGCNConv(64, 64, stride=3)
        self.fc1 = nn.Linear(64, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(64)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        print(f"Initial input - x: {x.shape}, edge_index: {edge_index.shape}, batch: {batch.shape}")

        x, edge_index, batch = self.gconv1(x, edge_index, batch)
        x = F.relu(self.bn1(x))
        print(f"After gconv1 - x: {x.shape}, edge_index: {edge_index.shape}, batch: {batch.shape}")

        x, edge_index, batch = self.gconv2(x, edge_index, batch)
        x = F.relu(self.bn2(x))
        print(f"After gconv2 - x: {x.shape}, edge_index: {edge_index.shape}, batch: {batch.shape}")

        x, edge_index, batch = self.gconv3(x, edge_index, batch)
        x = F.relu(self.bn3(x))
        print(f"After gconv3 - x: {x.shape}, edge_index: {edge_index.shape}, batch: {batch.shape}")

        x, edge_index, batch = self.gconv4(x, edge_index, batch)
        x = F.relu(self.bn4(x))
        print(f"After gconv4 - x: {x.shape}, edge_index: {edge_index.shape}, batch: {batch.shape}")

        assert x.size(1) == 64, f"Output dimension mismatch: expected 64, got {x.size(1)}"

        # Add padding after the final ModifiedGCNConv layer
        final_num_nodes = data.x.size(0)  # Use the original number of nodes
        pad_size = final_num_nodes - x.size(0)
        if pad_size > 0:
            pad_x = torch.zeros(pad_size, x.size(1)).to(x.device)
            x = torch.cat([x, pad_x], dim=0)
            pad_batch = torch.full((pad_size,), batch.max() + 1, dtype=torch.long).to(batch.device)
            batch = torch.cat([batch, pad_batch], dim=0)
            print(f"After final padding - x: {x.shape}, batch: {batch.shape}")

        # Ensure the batch tensor matches the input size
        if batch.size(0) < final_num_nodes:
            pad_size = final_num_nodes - batch.size(0)
            pad_batch = torch.arange(batch.max() + 1, batch.max() + 1 + pad_size, device=batch.device)
            batch = torch.cat([batch, pad_batch], dim=0)
            print(f"After final batch padding - batch: {batch.shape}")

        # Ensure unique and sequential batch indices
        batch = torch.arange(final_num_nodes, device=batch.device)
        print(f"After remapping batch indices - batch: {batch.shape}")

        # Debugging batch before pooling
        unique_batches, counts = torch.unique(batch, return_counts=True)
        print(f"Unique batches before pooling: {unique_batches}, counts: {counts}")

        x = global_mean_pool(x, batch)
        print(f"After global mean pool - x: {x.shape}")

        x = F.relu(self.fc1(x))
        print(f"After fc1 - x: {x.shape}")

        x = self.fc2(x)
        print(f"After fc2 - x: {x.shape}")

        return x

if __name__=="__main__":
    # Example initialization
    model = GCN_Net3()

    # Create dummy data to test the model
    x = torch.randn((100, 1))
    edge_index = torch.randint(0, 100, (2, 300))
    batch = torch.arange(100)

    data = Data(x=x, edge_index=edge_index, batch=batch)

    output = model(data)
    print("Final output shape:", output.shape)


    # # Create dummy data to test the model
    # x = torch.randn((100, 1))
    # edge_index = torch.randint(0, 100, (2, 300))  # Random edge connections

    # # # Define two batches
    # # batch_1 = torch.zeros(64, dtype=torch.long)  # First 64 nodes in the first batch
    # # batch_2 = torch.ones(36, dtype=torch.long) * 1  # Remaining 36 nodes in the second batch

    # # # Combine the batches into a single tensor
    # # batch = torch.cat([batch_1, batch_2])

    # # Create the data object
    # data = Data(x=x, edge_index=edge_index)

    # # Create the DataLoader
    # data_loader = DataLoader([data], batch_size=64, shuffle=False)

    # # Example initialization
    # model = GCN_Net3()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # # Training loop
    # model.train()
    # for epoch in range(10):  # Example for 10 epochs
    #     for batch_data in data_loader:
    #         optimizer.zero_grad()
    #         out = model(batch_data)
    #         loss = F.cross_entropy(out, torch.randint(0, 10, (out.size(0),)))  # Dummy target
    #         loss.backward()
    #         optimizer.step()
    #     print(f'Epoch {epoch}, Loss: {loss.item()}')




