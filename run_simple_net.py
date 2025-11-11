import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from graph.graph import Graph
from net import SimpleNet
from graph.visualizer import visualize_graph


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

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}')

# Convert PyTorch Geometric data to our Graph format
def pyg_to_graph(data) -> Graph:
    """Convert PyTorch Geometric data to our Graph object."""
    config = {
        'num_nodes': data.num_nodes,
        'node_feature_dim': data.x.shape[1],
        'edge_feature_dim': 1,  # We'll create dummy edge features
        'node_features': data.x.cpu().numpy().tolist(),
        'edges': data.edge_index.t().cpu().numpy().tolist()
    }
    
    graph = Graph(config=config)
    # Initialize edge features as ones (dummy features)
    graph.edge_features = torch.ones(graph.num_edges, 1)
    return graph

# Create our graph
graph = pyg_to_graph(data)

# Visualize graph (optional)
# visualize_graph(graph)

graph = graph.to(device)

# Store masks and labels separately
train_mask = data.train_mask.to(device)
val_mask = data.val_mask.to(device)
test_mask = data.test_mask.to(device)
labels = data.y.to(device)

print(f'\nConverted to our Graph format')
print(f'Node features shape: {graph.node_features.shape}')
print(f'Edge features shape: {graph.edge_features.shape}')

# Initialize model
model = SimpleNet(node_feature_dim=dataset.num_features, edge_feature_dim=1).to(device)

# Add a classification layer (SimpleNet outputs 4-dim node features)
classifier = torch.nn.Linear(4, dataset.num_classes).to(device)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(classifier.parameters()), 
    lr=0.01, 
    weight_decay=5e-4
)

# Training function
def train():
    model.train()
    classifier.train()
    optimizer.zero_grad()
    
    # Need to recreate graph each time since forward pass modifies it
    graph_copy = pyg_to_graph(data)
    graph_copy = graph_copy.to(device)
    
    # Forward pass through GNN
    graph_copy = model(graph_copy)
    
    # Classification
    out = classifier(graph_copy.node_features)
    
    # Compute loss only on training nodes
    loss = F.cross_entropy(out[train_mask], labels[train_mask])
    
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing function
def test():
    model.eval()
    classifier.eval()
    
    with torch.no_grad():
        # Create fresh graph
        graph_copy = pyg_to_graph(data)
        graph_copy = graph_copy.to(device)
        
        # Forward pass
        graph_copy = model(graph_copy)
        out = classifier(graph_copy.node_features)
        pred = out.argmax(dim=1)
        
        # Calculate accuracies
        train_correct = (pred[train_mask] == labels[train_mask]).sum().item()
        train_acc = train_correct / train_mask.sum().item()
        
        val_correct = (pred[val_mask] == labels[val_mask]).sum().item()
        val_acc = val_correct / val_mask.sum().item()
        
        test_correct = (pred[test_mask] == labels[test_mask]).sum().item()
        test_acc = test_correct / test_mask.sum().item()
    
    return train_acc, val_acc, test_acc

# Training loop
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
"""
Training...
Epoch: 020, Loss: 1.9533, Train: 0.1429, Val: 0.1220, Test: 0.1300
Epoch: 040, Loss: 1.9462, Train: 0.1429, Val: 0.1620, Test: 0.1490
Epoch: 060, Loss: 1.6377, Train: 0.2857, Val: 0.2320, Test: 0.2460
Epoch: 080, Loss: 1.1582, Train: 0.4071, Val: 0.2960, Test: 0.3430
Epoch: 100, Loss: 0.9727, Train: 0.5000, Val: 0.2100, Test: 0.2540
Epoch: 120, Loss: 0.8630, Train: 0.5643, Val: 0.2540, Test: 0.3280
Epoch: 140, Loss: 0.8478, Train: 0.6071, Val: 0.3080, Test: 0.3510
Epoch: 160, Loss: 0.7519, Train: 0.5786, Val: 0.3020, Test: 0.3470
Epoch: 180, Loss: 0.7019, Train: 0.7000, Val: 0.2960, Test: 0.3120
Epoch: 200, Loss: 0.6043, Train: 0.7143, Val: 0.2760, Test: 0.2870

Final Results:
Train Accuracy: 0.7143
Validation Accuracy: 0.2760
Test Accuracy: 0.2870
"""