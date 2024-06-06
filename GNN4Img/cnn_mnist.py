#!/usr/bin/env python3

from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.transforms import ToTensor

print(f"Cuda availability = {torch.cuda.is_available()}")
print(f"Number of GPUs = {torch.cuda.device_count()}")

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# CIFAR10
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# MNIST
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=ToTensor())
testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=ToTensor())

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Arch 
class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.gap(x)
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x)
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        shortcut = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(shortcut + x)
        x = self.gap(x)
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_eval(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # training
    model.train()
    for epoch in trange(100):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # inference
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'test accuracy:{correct/total}')


# VGG like
# model = Net1()
# model.to(device)
# train_eval(model)

# VGG-BN like
model = Net2()
model.to(device)
train_eval(model)

print('------------------')
print('------------------')
# ResNet like
model = Net3()
model.to(device)
train_eval(model)
