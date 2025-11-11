import time

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

"""
GCNConv - NodeToNodeLayer with degree normalization
"""

# Define a simple Graph Convolutional Network
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
    
    def forward(self, x, edge_index):
        # First graph convolution layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second graph convolution layer
        x = self.conv2(x, edge_index)
        return x

# Load the Cora dataset (citation network)
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]  # Get the first (and only) graph

print(f'Dataset: {dataset}')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'\nGraph:')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
print(f'Contains self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}')

# Initialize model
model = GCN(dataset.num_features, hidden_channels=16, num_classes=dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing function
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    
    return train_acc, val_acc, test_acc

# Training loop
start_time = time.time()
print('\nTraining...')
for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        train_acc, val_acc, test_acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

# Final evaluation
train_acc, val_acc, test_acc = test()
print(f'\nFinal Results:')
print(f'Train Accuracy: {train_acc:.4f}')
print(f'Validation Accuracy: {val_acc:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Total Training Time: {time.time() - start_time:.2f} seconds')

"""
Training...
Epoch: 020, Loss: 1.7242, Train: 0.9000, Val: 0.7180, Test: 0.7410
Epoch: 040, Loss: 1.3540, Train: 0.9500, Val: 0.7580, Test: 0.7880
Epoch: 060, Loss: 0.9784, Train: 0.9643, Val: 0.7680, Test: 0.8060
Epoch: 080, Loss: 0.7138, Train: 0.9786, Val: 0.7860, Test: 0.8080
Epoch: 100, Loss: 0.5760, Train: 0.9786, Val: 0.8000, Test: 0.8240
Epoch: 120, Loss: 0.5034, Train: 0.9929, Val: 0.7900, Test: 0.8140
Epoch: 140, Loss: 0.4248, Train: 1.0000, Val: 0.7900, Test: 0.8160
Epoch: 160, Loss: 0.3934, Train: 1.0000, Val: 0.7920, Test: 0.8170
Epoch: 180, Loss: 0.3183, Train: 1.0000, Val: 0.7920, Test: 0.8060
Epoch: 200, Loss: 0.3206, Train: 1.0000, Val: 0.7820, Test: 0.7960

Final Results:
Train Accuracy: 1.0000
Validation Accuracy: 0.7820
Test Accuracy: 0.7960
Total Training Time: 1.04 seconds
"""
