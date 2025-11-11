import torch.nn as nn
from graph.graph import Graph
from net.layers import NodeToEdgeLayer, EdgeToNodeLayer, NodeToNodeLayer


class ExampleNet(nn.Module):
    def __init__(self, node_feature_dim: int, edge_feature_dim: int):
        super(ExampleNet, self).__init__()
        new_edge_dim = 16
        new_node_dim = 8
        final_node_dim = 4

        self.node_to_edge = NodeToEdgeLayer(node_feature_dim, edge_feature_dim, new_edge_dim)
        self.edge_to_node = EdgeToNodeLayer(node_feature_dim, new_edge_dim, new_node_dim)
        self.node_to_node = NodeToNodeLayer(new_node_dim, final_node_dim)

    def forward(self, graph: Graph) -> Graph:
        graph = self.node_to_edge(graph)
        graph = self.edge_to_node(graph)
        graph = self.node_to_node(graph)
        return graph
