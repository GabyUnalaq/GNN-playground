import torch
import torch.nn as nn
import torch.nn.functional as F
from graph.graph import Graph


__all__ = [
    'NodeToEdgeLayer',
    'EdgeToNodeLayer',
    'NodeToNodeLayer',
    'NodeToGraphLayer',
    'GraphToNodeLayer',
    'EdgeRemovalLayer',
    'EdgeAdditionLayer'
]


class NodeToEdgeLayer(nn.Module):
    """
    Graph Neural Network layer that aggregates node features to edge features.
    
    For each edge, this layer combines:
    - Source node features
    - Target node features  
    - Existing edge features (optional)
    
    The combined features are then transformed through a neural network.
    """
    
    def __init__(self, 
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 output_dim: int,
                 aggregation: str = 'concat',
                 hidden_dim: int = None,
                 activation: nn = nn.ReLU,
                 use_edge_features: bool = True,
                 dropout: float = 0.0):
        """
        Initialize NodeToEdgeLayer.
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            output_dim: Dimension of output edge features
            aggregation: How to combine source and target nodes ('concat', 'add', 'mul', 'max', 'mean')
            hidden_dim: Hidden layer dimension (if None, uses output_dim)
            activation: Activation function (nn Module of the chosen activation)
            use_edge_features: Whether to incorporate existing edge features
            dropout: Dropout probability
        """
        super(NodeToEdgeLayer, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.output_dim = output_dim
        self.aggregation = aggregation
        self.use_edge_features = use_edge_features
        self.dropout = dropout
        
        # Calculate input dimension based on aggregation method
        if aggregation == 'concat':
            input_dim = 2 * node_feature_dim
        elif aggregation in ['add', 'mul', 'max', 'mean']:
            input_dim = node_feature_dim
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        # Add edge feature dimension if using existing edge features
        if use_edge_features:
            input_dim += edge_feature_dim
        
        # Build neural network
        if hidden_dim is None:
            hidden_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Activation function
        self.activation = activation()
        
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, graph: Graph) -> Graph:
        """
        Forward pass.
        
        Args:
            graph: Graph object with:
                - node_features: [num_nodes, node_feature_dim]
                - edge_index: [2, num_edges] 
                - edge_features: [num_edges, edge_feature_dim] (optional)
        
        Returns:
            Updated graph with new edge_features: [num_edges, output_dim]
        """
        node_features = graph.node_features
        edge_index = graph.edge_index
        edge_features = graph.edge_features if self.use_edge_features else None
        
        # Get source and target node features for each edge
        source_nodes = edge_index[0]  # [num_edges]
        target_nodes = edge_index[1]  # [num_edges]
        
        source_features = node_features[source_nodes]  # [num_edges, node_feature_dim]
        target_features = node_features[target_nodes]  # [num_edges, node_feature_dim]
        
        # Aggregate source and target features
        if self.aggregation == 'concat':
            aggregated = torch.cat([source_features, target_features], dim=1)
        elif self.aggregation == 'add':
            aggregated = source_features + target_features
        elif self.aggregation == 'mul':
            aggregated = source_features * target_features
        elif self.aggregation == 'max':
            aggregated = torch.max(source_features, target_features)
        elif self.aggregation == 'mean':
            aggregated = (source_features + target_features) / 2.0
        
        # Incorporate existing edge features if requested
        if self.use_edge_features and edge_features is not None:
            aggregated = torch.cat([aggregated, edge_features], dim=1)
        
        # Pass through neural network
        x = self.fc1(aggregated)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        # Update graph's edge features
        graph.update_edge_features(x)
        
        return graph
    
    def __repr__(self):
        return (f"NodeToEdgeLayer(node_dim={self.node_feature_dim}, "
                f"edge_dim={self.edge_feature_dim}, output_dim={self.output_dim}, "
                f"aggregation={self.aggregation})")


class EdgeToNodeLayer(nn.Module):
    """
    Aggregates edge features to node features.
    
    For each node, this layer aggregates features from:
    - Incoming edges
    - Outgoing edges (optional)
    - Existing node features (optional)
    """
    
    def __init__(self,
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 output_dim: int,
                 aggregation: str = 'mean',
                 edge_direction: str = 'in',
                 hidden_dim: int = None,
                 activation: nn = nn.ReLU,
                 use_node_features: bool = True,
                 dropout: float = 0.0):
        """
        Initialize EdgeToNodeLayer.
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            output_dim: Dimension of output node features
            aggregation: How to aggregate edges ('mean', 'sum', 'max', 'min')
            edge_direction: Which edges to consider ('in', 'out', 'both')
            hidden_dim: Hidden layer dimension
            activation: Activation function (nn Module of the chosen activation)
            use_node_features: Whether to incorporate existing node features
            dropout: Dropout probability
        """
        super(EdgeToNodeLayer, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.output_dim = output_dim
        self.aggregation = aggregation
        self.edge_direction = edge_direction
        self.use_node_features = use_node_features
        self.dropout = dropout
        
        # Calculate input dimension
        input_dim = edge_feature_dim
        if use_node_features:
            input_dim += node_feature_dim
        
        if hidden_dim is None:
            hidden_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Activation
        self.activation = activation()
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, graph: Graph) -> Graph:
        """
        Forward pass.
        
        Args:
            graph: Graph object
        
        Returns:
            Updated graph with new node_features
        """
        node_features = graph.node_features
        edge_index = graph.edge_index
        edge_features = graph.edge_features
        
        num_nodes = node_features.shape[0]
        device = node_features.device
        
        # Aggregate edge features to nodes
        aggregated_edge_features = torch.zeros(num_nodes, self.edge_feature_dim, device=device)
        edge_counts = torch.zeros(num_nodes, device=device)
        
        for i in range(edge_index.shape[1]):
            src, tgt = edge_index[0, i], edge_index[1, i]
            
            if self.edge_direction == 'in':
                node_idx = tgt
            elif self.edge_direction == 'out':
                node_idx = src
            elif self.edge_direction == 'both':
                # Add to both source and target
                if self.aggregation == 'sum':
                    aggregated_edge_features[src] += edge_features[i]
                    aggregated_edge_features[tgt] += edge_features[i]
                elif self.aggregation == 'mean':
                    aggregated_edge_features[src] += edge_features[i]
                    aggregated_edge_features[tgt] += edge_features[i]
                    edge_counts[src] += 1
                    edge_counts[tgt] += 1
                elif self.aggregation == 'max':
                    aggregated_edge_features[src] = torch.max(aggregated_edge_features[src], edge_features[i])
                    aggregated_edge_features[tgt] = torch.max(aggregated_edge_features[tgt], edge_features[i])
                elif self.aggregation == 'min':
                    aggregated_edge_features[src] = torch.min(aggregated_edge_features[src], edge_features[i])
                    aggregated_edge_features[tgt] = torch.min(aggregated_edge_features[tgt], edge_features[i])
                continue
            else:
                raise ValueError(f"Unknown edge_direction: {self.edge_direction}")
            
            if self.aggregation == 'sum':
                aggregated_edge_features[node_idx] += edge_features[i]
            elif self.aggregation == 'mean':
                aggregated_edge_features[node_idx] += edge_features[i]
                edge_counts[node_idx] += 1
            elif self.aggregation == 'max':
                aggregated_edge_features[node_idx] = torch.max(aggregated_edge_features[node_idx], edge_features[i])
            elif self.aggregation == 'min':
                aggregated_edge_features[node_idx] = torch.min(aggregated_edge_features[node_idx], edge_features[i])
        
        # Average for mean aggregation
        if self.aggregation == 'mean':
            edge_counts = edge_counts.clamp(min=1)
            aggregated_edge_features = aggregated_edge_features / edge_counts.unsqueeze(1)
        
        # Combine with existing node features
        if self.use_node_features:
            combined = torch.cat([aggregated_edge_features, node_features], dim=1)
        else:
            combined = aggregated_edge_features
        
        # Pass through network
        x = self.fc1(combined)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        graph.update_node_features(x)

        return graph
    
    def __repr__(self):
        return (f"EdgeToNodeLayer(node_dim={self.node_feature_dim}, "
                f"edge_dim={self.edge_feature_dim}, output_dim={self.output_dim}, "
                f"aggregation={self.aggregation}, direction={self.edge_direction})")


class NodeToNodeLayer(nn.Module):
    """
    Standard message passing layer between nodes (neighbor aggregation).
    
    Each node aggregates features from its neighboring nodes.
    """
    
    def __init__(self,
                 node_feature_dim: int,
                 output_dim: int,
                 aggregation: str = 'mean',
                 neighbor_direction: str = 'in',
                 hidden_dim: int = None,
                 activation: nn = nn.ReLU,
                 self_loop: bool = True,
                 dropout: float = 0.0):
        """
        Initialize NodeToNodeLayer.
        
        Args:
            node_feature_dim: Dimension of node features
            output_dim: Dimension of output node features
            aggregation: How to aggregate neighbors ('mean', 'sum', 'max', 'min')
            neighbor_direction: Which neighbors to consider ('in', 'out', 'both')
            hidden_dim: Hidden layer dimension
            activation: Activation function (nn Module of the chosen activation)
            self_loop: Whether to include node's own features
            dropout: Dropout probability
        """
        super(NodeToNodeLayer, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.output_dim = output_dim
        self.aggregation = aggregation
        self.neighbor_direction = neighbor_direction
        self.self_loop = self_loop
        self.dropout = dropout
        
        # Input dimension
        input_dim = node_feature_dim
        if self_loop:
            input_dim += node_feature_dim
        
        if hidden_dim is None:
            hidden_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.activation = activation()
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, graph: Graph) -> Graph:
        """
        Forward pass.
        
        Args:
            graph: Graph object
        
        Returns:
            Updated graph with new node_features
        """
        node_features = graph.node_features
        edge_index = graph.edge_index
        
        num_nodes = node_features.shape[0]
        device = node_features.device
        
        # Aggregate neighbor features
        aggregated = torch.zeros(num_nodes, self.node_feature_dim, device=device)
        neighbor_counts = torch.zeros(num_nodes, device=device)
        
        for i in range(edge_index.shape[1]):
            src, tgt = edge_index[0, i], edge_index[1, i]
            
            if self.neighbor_direction == 'in':
                # Target receives from source
                receiver, sender = tgt, src
            elif self.neighbor_direction == 'out':
                # Source sends to target
                receiver, sender = src, tgt
            elif self.neighbor_direction == 'both':
                # Bidirectional
                if self.aggregation == 'sum':
                    aggregated[src] += node_features[tgt]
                    aggregated[tgt] += node_features[src]
                elif self.aggregation == 'mean':
                    aggregated[src] += node_features[tgt]
                    aggregated[tgt] += node_features[src]
                    neighbor_counts[src] += 1
                    neighbor_counts[tgt] += 1
                elif self.aggregation == 'max':
                    aggregated[src] = torch.max(aggregated[src], node_features[tgt])
                    aggregated[tgt] = torch.max(aggregated[tgt], node_features[src])
                elif self.aggregation == 'min':
                    aggregated[src] = torch.min(aggregated[src], node_features[tgt])
                    aggregated[tgt] = torch.min(aggregated[tgt], node_features[src])
                continue
            else:
                raise ValueError(f"Unknown neighbor_direction: {self.neighbor_direction}")
            
            if self.aggregation == 'sum':
                aggregated[receiver] += node_features[sender]
            elif self.aggregation == 'mean':
                aggregated[receiver] += node_features[sender]
                neighbor_counts[receiver] += 1
            elif self.aggregation == 'max':
                aggregated[receiver] = torch.max(aggregated[receiver], node_features[sender])
            elif self.aggregation == 'min':
                aggregated[receiver] = torch.min(aggregated[receiver], node_features[sender])
        
        # Average for mean aggregation
        if self.aggregation == 'mean':
            neighbor_counts = neighbor_counts.clamp(min=1)
            aggregated = aggregated / neighbor_counts.unsqueeze(1)
        
        # Include self features
        if self.self_loop:
            combined = torch.cat([aggregated, node_features], dim=1)
        else:
            combined = aggregated
        
        # Pass through network
        x = self.fc1(combined)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        graph.update_node_features(x)

        return graph
    
    def __repr__(self):
        return (f"NodeToNodeLayer(node_dim={self.node_feature_dim}, "
                f"output_dim={self.output_dim}, aggregation={self.aggregation}, "
                f"direction={self.neighbor_direction})")


class NodeToGraphLayer(nn.Module):
    """
    Aggregates all node features to a single graph-level embedding.
    """
    
    def __init__(self,
                 node_feature_dim: int,
                 output_dim: int,
                 aggregation: str = 'mean',
                 hidden_dim: int = None,
                 activation: nn = nn.ReLU,
                 dropout: float = 0.0):
        """
        Initialize NodeToGraphLayer.
        
        Args:
            node_feature_dim: Dimension of node features
            output_dim: Dimension of graph-level embedding
            aggregation: How to aggregate nodes ('mean', 'sum', 'max', 'min', 'attention')
            hidden_dim: Hidden layer dimension
            activation: Activation function (nn Module of the chosen activation)
            dropout: Dropout probability
        """
        super(NodeToGraphLayer, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.output_dim = output_dim
        self.aggregation = aggregation
        self.dropout = dropout
        
        if hidden_dim is None:
            hidden_dim = output_dim
        
        self.fc1 = nn.Linear(node_feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Attention mechanism for attention aggregation
        if aggregation == 'attention':
            self.attention_fc = nn.Linear(node_feature_dim, 1)

        self.activation = activation()
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, graph: Graph) -> Graph:
        """
        Forward pass.
        
        Args:
            graph: Graph object
        
        Returns:
            Updated graph with graph_features
        """
        node_features = graph.node_features
        
        # Aggregate nodes to graph level
        if self.aggregation == 'mean':
            graph_embedding = node_features.mean(dim=0, keepdim=True)
        elif self.aggregation == 'sum':
            graph_embedding = node_features.sum(dim=0, keepdim=True)
        elif self.aggregation == 'max':
            graph_embedding = node_features.max(dim=0, keepdim=True)[0]
        elif self.aggregation == 'min':
            graph_embedding = node_features.min(dim=0, keepdim=True)[0]
        elif self.aggregation == 'attention':
            # Compute attention weights
            attention_scores = self.attention_fc(node_features)  # [num_nodes, 1]
            attention_weights = F.softmax(attention_scores, dim=0)  # [num_nodes, 1]
            graph_embedding = (node_features * attention_weights).sum(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Pass through network
        x = self.fc1(graph_embedding)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        graph.update_graph_features(x)

        return graph
    
    def __repr__(self):
        return (f"NodeToGraphLayer(node_dim={self.node_feature_dim}, "
                f"output_dim={self.output_dim}, aggregation={self.aggregation})")


class GraphToNodeLayer(nn.Module):
    """
    Broadcasts graph-level features to all nodes.
    Combines graph features with existing node features.
    """
    
    def __init__(self,
                 node_feature_dim: int,
                 graph_feature_dim: int,
                 output_dim: int,
                 combination: str = 'concat',
                 hidden_dim: int = None,
                 activation: nn = nn.ReLU,
                 dropout: float = 0.0):
        """
        Initialize GraphToNodeLayer.
        
        Args:
            node_feature_dim: Dimension of node features
            graph_feature_dim: Dimension of graph features
            output_dim: Dimension of output node features
            combination: How to combine graph and node features ('concat', 'add', 'mul')
            hidden_dim: Hidden layer dimension
            activation: Activation function (nn Module of the chosen activation)
            dropout: Dropout probability
        """
        super(GraphToNodeLayer, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.graph_feature_dim = graph_feature_dim
        self.output_dim = output_dim
        self.combination = combination
        self.dropout = dropout
        
        # Calculate input dimension
        if combination == 'concat':
            input_dim = node_feature_dim + graph_feature_dim
        elif combination in ['add', 'mul']:
            if node_feature_dim != graph_feature_dim:
                raise ValueError(f"For {combination} combination, node and graph feature dims must match")
            input_dim = node_feature_dim
        else:
            raise ValueError(f"Unknown combination: {combination}")
        
        if hidden_dim is None:
            hidden_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.activation = activation()
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, graph: Graph) -> Graph:
        """
        Forward pass.
        
        Args:
            graph: Graph object (must have graph_features set)
        
        Returns:
            Updated graph with new node_features
        """
        node_features = graph.node_features
        graph_features = graph.graph_features
        
        if graph_features is None:
            raise ValueError("graph_features must be set before using GraphToNodeLayer")
        
        num_nodes = node_features.shape[0]
        
        # Broadcast graph features to all nodes
        broadcasted_graph = graph_features.expand(num_nodes, -1)
        
        # Combine with node features
        if self.combination == 'concat':
            combined = torch.cat([node_features, broadcasted_graph], dim=1)
        elif self.combination == 'add':
            combined = node_features + broadcasted_graph
        elif self.combination == 'mul':
            combined = node_features * broadcasted_graph
        
        # Pass through network
        x = self.fc1(combined)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        graph.update_node_features(x)

        return graph
    
    def __repr__(self):
        return (f"GraphToNodeLayer(node_dim={self.node_feature_dim}, "
                f"graph_dim={self.graph_feature_dim}, output_dim={self.output_dim}, "
                f"combination={self.combination})")


class EdgeRemovalLayer(nn.Module):
    """
    Removes edges from the graph based on learned importance scores.
    
    Scores each existing edge using:
    - Source node features
    - Target node features
    - Edge features
    
    Edges with scores below threshold are removed.
    """
    
    def __init__(self,
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 hidden_dim: int = 64,
                 threshold: float = 0.5,
                 use_edge_features: bool = True,
                 activation: nn = nn.ReLU,
                 dropout: float = 0.0):
        """
        Initialize EdgeRemovalLayer.
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Hidden layer dimension for scoring network
            threshold: Edges with score < threshold are removed
            use_edge_features: Whether to use edge features in scoring
            activation: Activation function (nn Module of the chosen activation)
            dropout: Dropout probability
        """
        super(EdgeRemovalLayer, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.use_edge_features = use_edge_features
        self.dropout = dropout
        
        # Calculate input dimension for scoring network
        # Concatenate: source_node + target_node + edge_features
        input_dim = 2 * node_feature_dim
        if use_edge_features:
            input_dim += edge_feature_dim
        
        # Scoring network: outputs probability of keeping the edge
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.activation = activation()
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, graph: Graph, return_scores=False) -> Graph:
        """
        Forward pass - removes edges based on learned scores.
        
        Args:
            graph: Graph object
            return_scores: If True, returns (graph, edge_scores, removed_count)
        
        Returns:
            Updated graph with edges removed
        """
        node_features = graph.node_features
        edge_index = graph.edge_index
        edge_features = graph.edge_features
        
        if edge_index.shape[1] == 0:
            # No edges to remove
            if return_scores:
                return graph, torch.tensor([]), 0
            return graph
        
        # Get source and target node features for each edge
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        source_features = node_features[source_nodes]
        target_features = node_features[target_nodes]
        
        # Combine features
        if self.use_edge_features:
            combined = torch.cat([source_features, target_features, edge_features], dim=1)
        else:
            combined = torch.cat([source_features, target_features], dim=1)
        
        # Compute edge scores (probability of keeping)
        x = self.fc1(combined)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.fc3(x)
        edge_scores = torch.sigmoid(x).squeeze()  # [num_edges]
        
        # Create mask for edges to keep
        keep_mask = edge_scores >= self.threshold
        
        # Filter edges
        new_edge_index = edge_index[:, keep_mask]
        new_edge_features = edge_features[keep_mask]
        
        removed_count = (~keep_mask).sum().item()
        
        graph.update_edges(new_edge_index, new_edge_features)
        
        if return_scores:
            return graph, edge_scores, removed_count
        return graph
    
    def __repr__(self):
        return (f"EdgeRemovalLayer(node_dim={self.node_feature_dim}, "
                f"edge_dim={self.edge_feature_dim}, threshold={self.threshold})")


class EdgeAdditionLayer(nn.Module):
    """
    Adds edges to the graph based on learned node similarity.
    
    Uses k-NN approach: for each node, finds k most similar nodes
    and adds directed edges to them.
    
    Similarity is computed directionally: similarity(A→B) may differ from similarity(B→A)
    """
    
    def __init__(self,
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 k: int = 3,
                 hidden_dim: int = 64,
                 threshold: float = 0.5,
                 activation: nn = nn.ReLU,
                 dropout: float = 0.0,
                 avoid_duplicates: bool = True):
        """
        Initialize EdgeAdditionLayer.
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features (for new edges)
            k: Number of nearest neighbors to consider per node
            hidden_dim: Hidden layer dimension for similarity network
            threshold: Only add edges with similarity > threshold
            activation: Activation function (nn Module of the chosen activation)
            dropout: Dropout probability
            avoid_duplicates: If True, don't add edges that already exist
        """
        super(EdgeAdditionLayer, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.k = k
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.dropout = dropout
        self.avoid_duplicates = avoid_duplicates
        
        # Directional similarity network
        # Input: concatenation of source and target node features
        # Output: similarity score (0 to 1) and edge features
        input_dim = 2 * node_feature_dim
        
        # Similarity scoring network
        self.similarity_fc1 = nn.Linear(input_dim, hidden_dim)
        self.similarity_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.similarity_fc3 = nn.Linear(hidden_dim // 2, 1)
        
        # Edge feature generation network
        self.edge_feat_fc1 = nn.Linear(input_dim, hidden_dim)
        self.edge_feat_fc2 = nn.Linear(hidden_dim, edge_feature_dim)
        
        self.activation = activation()
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def compute_similarity(self, source_features, target_features):
        """
        Compute directional similarity score between source and target nodes.
        
        Args:
            source_features: [batch_size, node_feature_dim]
            target_features: [batch_size, node_feature_dim]
        
        Returns:
            similarity_scores: [batch_size]
        """
        combined = torch.cat([source_features, target_features], dim=1)
        
        x = self.similarity_fc1(combined)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.similarity_fc2(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.similarity_fc3(x)
        
        return torch.sigmoid(x).squeeze(-1)
    
    def generate_edge_features(self, source_features, target_features):
        """
        Generate edge features for new edges.
        
        Args:
            source_features: [batch_size, node_feature_dim]
            target_features: [batch_size, node_feature_dim]
        
        Returns:
            edge_features: [batch_size, edge_feature_dim]
        """
        combined = torch.cat([source_features, target_features], dim=1)
        
        x = self.edge_feat_fc1(combined)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.edge_feat_fc2(x)
        
        return x
    
    def forward(self, graph: Graph, return_stats=False) -> Graph:
        """
        Forward pass - adds edges based on k-NN similarity.
        
        Args:
            graph: Graph object
            return_stats: If True, returns (graph, added_count, avg_similarity)
        
        Returns:
            Updated graph with new edges added
        """
        node_features = graph.node_features
        edge_index = graph.edge_index
        edge_features = graph.edge_features
        
        num_nodes = node_features.shape[0]
        device = node_features.device
        
        # Build set of existing edges for duplicate checking
        if self.avoid_duplicates:
            existing_edges = set()
            for i in range(edge_index.shape[1]):
                src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
                existing_edges.add((src, tgt))
        
        new_edges = []
        new_edge_features = []
        similarity_scores = []
        
        # For each node, find k most similar nodes
        for src_idx in range(num_nodes):
            src_features = node_features[src_idx].unsqueeze(0)  # [1, node_feature_dim]
            
            # Compute similarity to all other nodes
            src_features_repeated = src_features.repeat(num_nodes, 1)  # [num_nodes, node_feature_dim]
            
            similarities = self.compute_similarity(src_features_repeated, node_features)  # [num_nodes]
            
            # Don't connect to self
            similarities[src_idx] = -1
            
            # Get top-k most similar nodes
            topk_similarities, topk_indices = torch.topk(similarities, min(self.k, num_nodes - 1))
            
            # Add edges above threshold
            for i in range(len(topk_indices)):
                tgt_idx = topk_indices[i].item()
                similarity = topk_similarities[i].item()
                
                if similarity >= self.threshold:
                    # Check if edge already exists
                    if self.avoid_duplicates and (src_idx, tgt_idx) in existing_edges:
                        continue
                    
                    # Add edge
                    new_edges.append([src_idx, tgt_idx])
                    
                    # Generate edge features
                    edge_feat = self.generate_edge_features(
                        node_features[src_idx].unsqueeze(0),
                        node_features[tgt_idx].unsqueeze(0)
                    )
                    new_edge_features.append(edge_feat)
                    similarity_scores.append(similarity)
        
        # Add new edges to graph
        added_count = len(new_edges)
        
        if added_count > 0:
            new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=device).t()
            new_edge_feat_tensor = torch.cat(new_edge_features, dim=0)
            
            # Concatenate with existing edges
            graph.edge_index = torch.cat([edge_index, new_edge_index], dim=1)
            graph.edge_features = torch.cat([edge_features, new_edge_feat_tensor], dim=0)
            graph.num_edges = graph.edge_index.shape[1]
        
        if return_stats:
            avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
            return graph, added_count, avg_similarity
        
        return graph
    
    def __repr__(self):
        return (f"EdgeAdditionLayer(node_dim={self.node_feature_dim}, "
                f"edge_dim={self.edge_feature_dim}, k={self.k}, threshold={self.threshold})")


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a sample graph
    config = {
        'num_nodes': 5,
        'node_feature_dim': 8,
        'edge_feature_dim': 4,
        'graph_feature_dim': 16,
        'node_init': 'random',
        'edge_init': 'random',
        'edges': [
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
            [0, 2], [1, 3], [2, 4]
        ]
    }
    graph = Graph(config=config)
    graph = graph.to(device)
    print(f"Original Graph: \n{graph}")

    # Test simple GNN with multiple layers
    test_net = nn.Sequential(
        NodeToEdgeLayer(
            node_feature_dim=graph.node_features.shape[1],
            edge_feature_dim=graph.edge_features.shape[1],
            output_dim=10,
            aggregation='concat',
            hidden_dim=16,
            activation=nn.ReLU,
            use_edge_features=True,
            dropout=0.1
        ),
        EdgeToNodeLayer(
            node_feature_dim=graph.node_features.shape[1],
            edge_feature_dim=10,
            output_dim=12,
            aggregation='mean',
            edge_direction='in',
            hidden_dim=16,
            activation=nn.ReLU,
            use_node_features=True,
            dropout=0.1
        ),
        NodeToGraphLayer(
            node_feature_dim=12,
            output_dim=16,
            aggregation='attention',
            hidden_dim=32,
            activation=nn.ReLU,
            dropout=0.1
        ),
        # GraphToNodeLayer(
        #     node_feature_dim=12,
        #     graph_feature_dim=16,
        #     output_dim=14,
        #     combination='concat',
        #     hidden_dim=32,
        #     activation=nn.ReLU,
        #     dropout=0.1
        # )
    ).to(device)

    with torch.no_grad():
        updated_graph: Graph = test_net(graph)

    updated_graph = updated_graph.to('cpu')
    print(f"Output edge features shape: {updated_graph.edge_features.shape}")
    print(f"Sample output (first edge): {updated_graph.edge_features[0, :5].numpy()}")

    # Test edge manipulation - Addaptive GNN layers
    updated_graph = updated_graph.to(device)

    edge_removal_layer = EdgeRemovalLayer(
        node_feature_dim=updated_graph.node_features.shape[1],
        edge_feature_dim=updated_graph.edge_features.shape[1],
        hidden_dim=32,
        threshold=0.5,
        use_edge_features=True,
        activation=nn.ReLU,
        dropout=0.1
    ).to(device)

    edge_addition_layer = EdgeAdditionLayer(
        node_feature_dim=updated_graph.node_features.shape[1],
        edge_feature_dim=updated_graph.edge_features.shape[1],
        k=2,
        hidden_dim=32,
        threshold=0.6,
        activation=nn.ReLU,
        dropout=0.1,
        avoid_duplicates=True
    ).to(device)

    with torch.no_grad():
        pruned_graph: Graph
        pruned_graph, edge_scores, removed_count = edge_removal_layer(updated_graph, return_scores=True)
    
    pruned_graph = pruned_graph.to('cpu')
    print(f"Number of edges after pruning: {pruned_graph.num_edges}")
    print(f"Number of edges removed: {removed_count}")
    print(f"Sample edge scores: {edge_scores[:5].cpu().numpy()}")

    pruned_graph = pruned_graph.to(device)
    with torch.no_grad():
        expanded_graph: Graph
        expanded_graph, added_count, avg_similarity = edge_addition_layer(pruned_graph, return_stats=True)
    
    expanded_graph = expanded_graph.to('cpu')
    print(f"Number of edges after addition: {expanded_graph.num_edges}")
    print(f"Number of edges added: {added_count}")
    print(f"Average similarity of added edges: {avg_similarity}")
