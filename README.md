# LLM GDL Base
Pytorchに基づいたプロジェクトを実践する

## I: GNN4Img
- Implemented Stride and Padding for [graph convolutional operator (GCNConv)](https://pytorch-geometric.readthedocs.io/en/2.5.1/generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv) layer in PyTorch Geometric.
- Dataset:
    - MNIST
- Baselines:
    - CNN Net1: Vanilla 3X3 CNN 
    - CNN Net2: CNN Net1 with BatchNorm 
    - CNN Net3: CNN Net2 with ResNet
- Results:

![Training after 300 epochs](GNN4Img/results/expt_MGCNConvNet3_trial5.png)