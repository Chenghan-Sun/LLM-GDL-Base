#!/usr/bin/env python3

from tqdm import tqdm, trange
import sys, os
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, RandomRotation, RandomResizedCrop
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
import torch.nn as nn
import torch.nn.functional as F

print(f"Cuda availability = {torch.cuda.is_available()}")
print(f"Number of GPUs = {torch.cuda.device_count()}")

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Data augmentation and normalization
transform = Compose([
    RandomRotation(10),
    RandomResizedCrop(28, scale=(0.8, 1.0)),
    ToTensor()
])

# CIFAR10
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# MNIST (1X28X28)
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=ToTensor())

def img2graph(image):
    """ Create graph data from the images. 
    Each image will be treated as a graph where each pixel is a node, and edges connect neighboring pixels.
        - added diagonal edges 
        - added self-loop edge 
    Total equivalent to 3X3 conv layer
    """
    image = image.squeeze()
    rows, cols = image.size()

    # node features     
    x = image.view(-1, 1)  # number of node features = 1 

    # edges features
    edge_index = []
    for r in range(rows):
        for c in range(cols):
            curr_node_idx = r * cols + c

            # add self-loop edge 
            edge_index.append((curr_node_idx, curr_node_idx))
            
            # add up/down/left/right edges
            if r > 0:  # Connect to the pixel above
                edge_index.append((curr_node_idx, (r-1) * cols + c))
                edge_index.append(((r-1) * cols + c, curr_node_idx))
            if r < rows - 1:  # Connect to the pixel below 
                edge_index.append((curr_node_idx, (r+1) * cols + c))
                edge_index.append(((r+1) * cols + c, curr_node_idx))
            if c > 0:  # Connect to the pixel left 
                edge_index.append((curr_node_idx, r * cols + (c-1)))
                edge_index.append((r * cols + (c-1), curr_node_idx))
            if c < cols - 1:  # Connect to the pixel right
                edge_index.append((curr_node_idx, r * cols + (c+1)))
                edge_index.append((r * cols + (c+1), curr_node_idx))

            # add diagonal edges 
            if r > 0 and c > 0:
                edge_index.append((curr_node_idx, (r-1) * cols + (c-1)))
                edge_index.append(((r-1) * cols + (c-1), curr_node_idx))
            if r > 0 and c < cols - 1:
                edge_index.append((curr_node_idx, (r-1) * cols + (c+1)))
                edge_index.append(((r-1) * cols + (c+1), curr_node_idx))
            if r < rows - 1 and c > 0:
                edge_index.append((curr_node_idx, (r+1) * cols + (c-1)))
                edge_index.append(((r+1) * cols + (c-1), curr_node_idx))
            if r < rows - 1 and c < cols - 1:
                edge_index.append((curr_node_idx, (r+1) * cols + (c+1)))
                edge_index.append(((r+1) * cols + (c+1), curr_node_idx))

    # stored in contiguous memory
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

print("Begin converting the images to graphs: ")
train_graphs = [img2graph(img_pair[0]) for img_pair in tqdm(trainset)]
test_graphs = [img2graph(img_pair[0]) for img_pair in tqdm(testset)]

print("Begin assigning the labels to graphs: ")
for i, g in enumerate(train_graphs):
    g.y = torch.tensor([trainset[i][1]], dtype=torch.long)

for i, g in enumerate(test_graphs):
    g.y = torch.tensor([testset[i][1]], dtype=torch.long)

print("Begin creating the PyG Dataloader: ")
train_loader = DataLoader(train_graphs, batch_size=128, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=128, shuffle=False) 

# GCN Arch -- equivalent to CNN Net1 
class GCN_Net1(nn.Module):
    def __init__(self):
        super(GCN_Net1, self).__init__()
        self.gconv1 = GCNConv(1, 64)
        self.gconv2 = GCNConv(64, 64)
        self.gconv3 = GCNConv(64, 64)
        self.fc1 = nn.Linear(64, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index 
        x = F.relu(self.bn1(self.gconv1(x, edge_index)))  # add BatchNorm1d 
        x = F.relu(self.bn2(self.gconv2(x, edge_index)))  # add BatchNorm1d 
        x = F.relu(self.bn3(self.gconv3(x, edge_index)))  # add BatchNorm1d 
        x = global_mean_pool(x, data.batch)
        x = self.dropout(x)  # add dropout layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# class GAT(nn.Module):
#     def __init__(self):
#         super(GAT, self).__init__():
#         self.gconv1 = 

def train_eval(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    for epoch in range(100):
        # Training
        model.train()
        train_total = 0
        train_correct = 0
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data.y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += data.y.size(0)
            train_correct += (predicted == data.y).sum().item()

        train_acc = train_correct / train_total
        scheduler.step()  # Step the learning rate scheduler

        # Inference 
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += data.y.size(0)
                correct += (predicted == data.y).sum().item()

        test_acc = correct / total

        print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')

print('Begin training: ')
model = GCN_Net1().to(device)
train_eval(model)

print('All training was done!')

