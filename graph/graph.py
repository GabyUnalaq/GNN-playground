import torch
import json
from pathlib import Path
from typing import Optional, Dict, Any


FEATURE_DATA_TYPE = torch.float32
EDGE_INDEX_TYPE = torch.long

__all__ = ['Graph', 'FeatureSizeError']


class FeatureSizeError(Exception):
    """Custom exception for feature size mismatches in graph representation."""
    pass


class Graph:
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        """
        Initialize a Graph from a config dict or config file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to configuration file (JSON or YAML)
        """
        if config_path:
            config = self._load_config(config_path)
        elif config is None:
            raise ValueError("Either config or config_path must be provided")
        
        # Store configuration
        self.config = config
        
        # Graph structure
        self.num_nodes = config['num_nodes']
        self.edge_index, self.num_edges = None, None
        self.node_feature_dim = config['node_feature_dim']
        self.edge_feature_dim = config['edge_feature_dim']
        self.graph_feature_dim = config.get('graph_feature_dim', None)

        # Graph features
        self.node_features = None
        self.edge_features = None
        self.graph_features = None

        self._init_features()
    
    def __repr__(self):
        return (f"Graph(num_nodes={self.num_nodes}, num_edges={self.num_edges}, "
                f"node_dim={self.node_feature_dim}, edge_dim={self.edge_feature_dim})")

    def _init_features(self):
        # Initialize node features
        if 'node_features' in self.config and self.config['node_features'] is not None:
            self.node_features = torch.tensor(self.config['node_features'], dtype=FEATURE_DATA_TYPE)
            if self.node_features.shape != (self.num_nodes, self.node_feature_dim):
                raise FeatureSizeError(f"Node features must have shape ({self.num_nodes}, {self.node_feature_dim}), "
                                       f"but got {self.node_features.shape}")
        else:
            init_method = self.config.get('node_init', 'zeros')
            if init_method == 'zeros':
                self.node_features = torch.zeros(self.num_nodes, self.node_feature_dim)
            elif init_method == 'random':
                self.node_features = torch.randn(self.num_nodes, self.node_feature_dim)
            elif init_method == 'uniform':
                self.node_features = torch.rand(self.num_nodes, self.node_feature_dim)
        
        # Initialize edge index
        if 'edges' in self.config and self.config['edges'] is not None:
            # edges should be a list of [source, target] pairs
            edges = self.config['edges']
            if not all(len(edge) == 2 for edge in edges):
                raise ValueError("Each edge must be a [source, target] pair")
            self.edge_index = torch.tensor(edges, dtype=EDGE_INDEX_TYPE).t()  # Transpose to [2, num_edges]
            self.num_edges = self.edge_index.shape[1]
        else:
            self.edge_index = torch.empty((2, 0), dtype=EDGE_INDEX_TYPE)
            self.num_edges = 0
        
        # Initialize edge features
        if 'edge_features' in self.config and self.config['edge_features'] is not None:
            self.edge_features = torch.tensor(self.config['edge_features'], dtype=FEATURE_DATA_TYPE)
            if self.edge_features.shape != (self.num_edges, self.edge_feature_dim):
                raise FeatureSizeError(f"Edge features must have shape ({self.num_edges}, {self.edge_feature_dim}), "
                                       f"but got {self.edge_features.shape}")
        else:
            if self.num_edges > 0:
                init_method = self.config.get('edge_init', 'zeros')
                if init_method == 'zeros':
                    self.edge_features = torch.zeros(self.num_edges, self.edge_feature_dim)
                elif init_method == 'random':
                    self.edge_features = torch.randn(self.num_edges, self.edge_feature_dim)
                elif init_method == 'uniform':
                    self.edge_features = torch.rand(self.num_edges, self.edge_feature_dim)
            else:
                self.edge_features = torch.empty((0, self.edge_feature_dim), dtype=FEATURE_DATA_TYPE)
        
        # Initialize graph-level features
        if 'graph_features' in self.config and self.config['graph_features'] is not None:
            self.graph_features = torch.tensor(self.config['graph_features'], dtype=FEATURE_DATA_TYPE)
            if self.graph_features.shape != (1, self.graph_feature_dim):
                raise FeatureSizeError(f"Graph features must have shape (1, {self.graph_feature_dim}), "
                                       f"but got {self.graph_features.shape}")
        else:
            if self.graph_feature_dim is not None:
                init_method = self.config.get('graph_init', 'zeros')
                if init_method == 'zeros':
                    self.graph_features = torch.zeros(1, self.graph_feature_dim)
                elif init_method == 'random':
                    self.graph_features = torch.randn(1, self.graph_feature_dim)
                elif init_method == 'uniform':
                    self.graph_features = torch.rand(1, self.graph_feature_dim)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a JSON file."""
        path = Path(config_path)
        if path.suffix not in ['.json']:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            return json.load(f)
    
    def add_edge(self, source: int, target: int, features: Optional[torch.Tensor] = None):
        """Add a single edge to the graph."""
        if not (0 <= source < self.num_nodes):
            raise IndexError(f"Source node index {source} out of bounds")
        if not (0 <= target < self.num_nodes):
            raise IndexError(f"Target node index {target} out of bounds")
        if features is not None and features.shape != (self.edge_feature_dim,):
            raise FeatureSizeError(f"Edge features must have shape ({self.edge_feature_dim},), "
                                   f"but got {features.shape}")

        new_edge = torch.tensor([[source], [target]], dtype=torch.long)
        self.edge_index = torch.cat([self.edge_index, new_edge], dim=1)
        
        if features is None:
            features = torch.zeros(1, self.edge_feature_dim)
        else:
            features = features.reshape(1, -1)
        
        self.edge_features = torch.cat([self.edge_features, features], dim=0)
        self.num_edges += 1
    
    def remove_edge(self, edge_idx: int):
        """Remove an edge by its index."""
        if edge_idx < 0 or edge_idx >= self.num_edges:
            raise IndexError(f"Edge index {edge_idx} out of range")
        
        mask = torch.ones(self.num_edges, dtype=torch.bool)
        mask[edge_idx] = False
        
        self.edge_index = self.edge_index[:, mask]
        self.edge_features = self.edge_features[mask]
        self.num_edges -= 1
    
    def update_node_feature(self, node_idx: int, features: torch.Tensor):
        """Update features for a specific node."""
        self.node_features[node_idx] = features
    
    def update_edge_feature(self, edge_idx: int, features: torch.Tensor):
        """Update feature of a specific edge."""
        self.edge_features[edge_idx] = features

    def update_node_features(self, features: torch.Tensor):
        """Update features for all nodes."""
        if features.shape[0] != self.num_nodes:
            raise FeatureSizeError(f"Node features must have {self.num_nodes} rows, "
                       f"but got {features.shape[0]}")
        self.node_features = features
        self.node_feature_dim = features.shape[1]

    def update_edge_features(self, features: torch.Tensor):
        """Update features for all edges."""
        if features.shape[0] != self.num_edges:
            raise FeatureSizeError(f"Edge features must have {self.num_edges} rows, "
                       f"but got {features.shape[0]}")
        self.edge_features = features
        self.edge_feature_dim = features.shape[1]

    def update_graph_features(self, features: torch.Tensor):
        """Update graph-level features."""
        if features.shape[0] != 1:
            raise FeatureSizeError(f"Graph features must have shape (1, feature_dim), "
                       f"but got {features.shape}")
        self.graph_features = features
        self.graph_feature_dim = features.shape[1]

    def update_edges(self, edge_index: torch.Tensor, edge_features: Optional[torch.Tensor] = None):
        """Update the entire edge index and edge features."""
        if edge_index.shape[0] != 2:
            raise ValueError("Edge index must have shape [2, num_edges]")
        
        self.edge_index = edge_index
        self.num_edges = edge_index.shape[1]
        
        if edge_features is not None:
            if edge_features.shape[0] != self.num_edges:
                raise FeatureSizeError(f"Edge features must have {self.num_edges} rows, "
                                       f"but got {edge_features.shape[0]}")
            self.edge_features = edge_features
            self.edge_feature_dim = edge_features.shape[1]
        else:
            self.edge_features = torch.zeros(self.num_edges, self.edge_feature_dim)
    
    def get_neighbors(self, node_idx: int, direction: str = 'out') -> torch.Tensor:
        """
        Get neighbors of a node.
        
        Args:
            node_idx: Index of the node
            direction: 'out' for outgoing edges, 'in' for incoming edges, 'both' for all
        
        Returns:
            Tensor of neighbor node indices
        """
        if direction == 'out':
            mask = self.edge_index[0] == node_idx
            return self.edge_index[1, mask]
        elif direction == 'in':
            mask = self.edge_index[1] == node_idx
            return self.edge_index[0, mask]
        elif direction == 'both':
            out_neighbors = self.get_neighbors(node_idx, 'out')
            in_neighbors = self.get_neighbors(node_idx, 'in')
            return torch.unique(torch.cat([out_neighbors, in_neighbors]))
        else:
            raise ValueError(f"Invalid direction: {direction}")
    
    def to(self, device: torch.device):
        """Move graph to device (CPU/GPU)."""
        self.node_features = self.node_features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_features = self.edge_features.to(device)
        if self.graph_features is not None:
            self.graph_features = self.graph_features.to(device)
        return self
    
    def save_config(self, path: str):
        """Save current graph configuration to a file."""
        config = {
            'num_nodes': self.num_nodes,
            'node_feature_dim': self.node_feature_dim,
            'edge_feature_dim': self.edge_feature_dim,
            'node_features': self.node_features.tolist(),
            'edges': self.edge_index.t().tolist(),
            'edge_features': self.edge_features.tolist(),
        }
        
        if self.graph_features is not None:
            config['graph_features'] = self.graph_features.tolist()
        
        path_obj = Path(path)
        if path_obj.suffix not in ['.json']:
            raise ValueError(f"Unsupported config file format: {path_obj.suffix}")

        with open(path_obj, 'w') as f:
            json.dump(config, f, indent=2)
    
    def summary(self):
        """Print detailed information about the graph."""
        print("=" * 60)
        print("GRAPH SUMMARY")
        print("=" * 60)
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Number of edges: {self.num_edges}")
        print(f"Node feature dimension: {self.node_feature_dim}")
        print(f"Edge feature dimension: {self.edge_feature_dim}")
        print(f"\nNode features shape: {self.node_features.shape}")
        print(f"Edge index shape: {self.edge_index.shape}")
        print(f"Edge features shape: {self.edge_features.shape}")
        
        if self.num_edges > 0:
            print(f"\nAverage node degree: {self.num_edges / self.num_nodes:.2f}")
        
        if self.graph_features is not None:
            print(f"Graph features shape: {self.graph_features.shape}")
        
        print("=" * 60)


if __name__ == "__main__":
    config = {
        'num_nodes': 5,
        'node_feature_dim': 3,
        'edge_feature_dim': 2,
        'graph_feature_dim': 4,
        'node_init': 'random',
        'edge_init': 'uniform',
        'edges': [
            [0, 1], [1, 0],
            [1, 2], [2, 1],
            [2, 3], [3, 2],
            [3, 4], [4, 3],
            [4, 0], [0, 4]
        ]
    }
    
    graph = Graph(config=config)
    graph.summary()
    
    # Add and remove edges
    graph.add_edge(0, 2, torch.tensor([0.5, 0.8]))
    print("\nAfter adding an edge (0 -> 2):")
    print(f"  After adding edge: {graph}")
    print(f"  New edge index: {graph.edge_index}")
    
    # Get neighbors
    out_neighbors = graph.get_neighbors(0, 'out')
    in_neighbors = graph.get_neighbors(0, 'in')
    print(f"\nNeighbors of node 0:")
    print(f"  - Outgoing neighbors: {out_neighbors.tolist()}")
    print(f"  - Incoming neighbors: {in_neighbors.tolist()}")
