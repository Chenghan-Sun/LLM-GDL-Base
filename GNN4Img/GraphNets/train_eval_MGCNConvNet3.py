#!/usr/bin/env python3

from tqdm import tqdm, trange
import sys, os
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, RandomRotation, RandomResizedCrop
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.nn as nn

from gnn_models import GCN_Net1, GCN_Net2, GCN_Net3
from utils import plot_metrics, EarlyStopping

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
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=ToTensor())
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
    x = image.view(-1, 1).float()  # number of node features = 1 

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
# only use the first 600 images for training and the first 100 images for testing
# subset_trainset = [(trainset.data[i], trainset.targets[i]) for i in range(60)]
# subset_testset = [(testset.data[i], testset.targets[i]) for i in range(10)]

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


def train_eval(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # early_stopping = EarlyStopping(patience=30, delta=0.01)  # EarlyStopping instance
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(300):
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
        # val_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                outputs = model(data)
                # loss = criterion(outputs, data.y)
                # val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += data.y.size(0)
                correct += (predicted == data.y).sum().item()

        test_acc = correct / total
        # val_loss /= len(test_loader)

        print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc) 

        # Check early stopping condition
        # early_stopping(val_loss, model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     model.load_state_dict(torch.load('checkpoint.pt'))
        #     break
    
    # Plot metrics
    plot_metrics('expt_MGCNConvNet3_trial6', train_losses, train_accuracies, test_accuracies)


print('Begin training: ')
model = GCN_Net3().to(device)
train_eval(model)

print('All training was done!')

