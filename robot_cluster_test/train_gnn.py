import os
import copy
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.layers import NodeToNodeLayer
from graph.graph import Graph
from .cluster_visualizer import FlockingVisualizer

DATA_FILE = "test_data_3-5r_100s_no_target.pkl"  # test_data_3-5r_50s test_data_3-5r_100s_no_target
CKPT_LOAD_NAME = None  # "flocking_gnn_checkpoint_no_targets.pth"
CKPT_SAVE_NAME = "flocking_gnn_checkpoint_no_targets.pth"
TRAIN_EPOCHS = 200  # Set to -1 to not train
TEST_N = 10

script_dir = os.path.dirname(os.path.abspath(__file__))

assert CKPT_SAVE_NAME.endswith('.pth'), "Checkpoint filename must end with .pth"
assert TRAIN_EPOCHS > 0 or CKPT_LOAD_NAME is not None, "If not training, a checkpoint must be provided for testing."

# Load data
if not os.path.exists(os.path.join(script_dir, "data", DATA_FILE)):
    raise FileNotFoundError(f"Data file not found: {DATA_FILE}.")

with open(os.path.join(script_dir, "data", DATA_FILE), "rb") as f:
    data = pickle.load(f)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}')

# Initialize model
class FlockingGnn(nn.Module):
    def __init__(self, with_target: bool = False):
        super(FlockingGnn, self).__init__()

        """
        NodeToNodeLayer(in_dim, out_dim) =>
            node_embeddings -> aggregation -> torch.Linear(in_dim, hidden_dim) -> intermidiate_embedding
            intermidiate_embedding -> torch.Linear(hidden_dim, out_dim) -> final_node_embedding
        """
        self.with_target = with_target
        self.layer_1 = NodeToNodeLayer(2, 8, hidden_dim=4)
        self.layer_2 = NodeToNodeLayer(8, 2, hidden_dim=4)

    def forward(self, graph: Graph) -> Graph:
        graph = self.layer_1(graph)
        graph = self.layer_2(graph)
        return graph

model = FlockingGnn().to(device)

optimizer = torch.optim.Adam(
    list(model.parameters()), 
    lr=0.01, 
    weight_decay=5e-4
)

def save_checkpoint(epoch, model, optimizer, loss, filename=CKPT_SAVE_NAME):
    checkpoint = {
        'epoch': epoch,
        'with_target': model.with_target,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, os.path.join(script_dir, "ckpts", filename))
    print(f'Checkpoint saved at epoch {epoch} to {filename}')

if CKPT_LOAD_NAME is not None:
    ckpt_path = os.path.join(script_dir, "ckpts", CKPT_LOAD_NAME)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}.")
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.with_target = checkpoint['with_target']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f'Loaded checkpoint from {CKPT_LOAD_NAME} at epoch {start_epoch}')
else:
    start_epoch = 0

def train_step(graph_in: Graph, graph_out: Graph) -> float:
    optimizer.zero_grad()

    graph_in.to(device)
    graph_out.to(device)
    
    # Forward pass through GNN
    graph_pred: Graph = model(graph_in)
    
    # Compute loss (MSE between predicted controls and true controls)
    loss = F.mse_loss(graph_pred.node_features, graph_out.node_features)
    
    loss.backward()
    optimizer.step()
    return loss.item()

def train(epochs=50):
    if epochs <= 0:
        return
    if start_epoch >= epochs:
        print("Model already trained for the specified number of epochs.")
        return

    model.train()

    for epoch in range(start_epoch, epochs):
        train_loss = 0.0
        for i in range(data["samples_num"]):
            graph_in = data["graphs_in"][i]
            graph_out = data["graphs_out"][i]
            loss = train_step(copy.copy(graph_in), graph_out)
            train_loss += loss
        train_loss /= data["samples_num"]

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.5f}')
        
        if (epoch + 1) % 25 == 0:
            save_checkpoint(epoch + 1, model, optimizer, train_loss)

def test(n=1):
    if n <= 0:
        return

    model.eval()
    
    with torch.no_grad():
        samples_data = []
        for _ in range(n):
            # generate random data for testing
            robot_states = np.random.rand(3, 2)  # x, y
            target_pos = np.random.rand(2)  # x, y

            in_graph = Graph({
                "num_nodes": 3,
                "node_feature_dim": 2,
                "edge_feature_dim": 0,
                "node_features": robot_states.tolist(),
                "edges": [[i, j] for i in range(3) for j in range(3) if i != j]
            })
            in_graph_copy = copy.copy(in_graph)
            in_graph = in_graph.to(device)
            
            # Forward pass
            graph_pred: Graph = model(in_graph).to("cpu")

            robot_states = in_graph_copy.node_features.numpy()
            controls = graph_pred.node_features.cpu().numpy()
            samples_data.append((robot_states, target_pos, controls))

        FlockingVisualizer(samples_data)

train(epochs=TRAIN_EPOCHS)
test(n=TEST_N)
