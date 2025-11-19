import torch.nn as nn
from graph.graph import Graph
from net.layers import NodeToNodeLayer


class SimpleNet(nn.Module):
    def __init__(self, node_feature_dim: int, edge_feature_dim: int = 0):
        super(SimpleNet, self).__init__()
        final_node_dim = 4

        self.layer_1 = NodeToNodeLayer(node_feature_dim, node_feature_dim // 2)
        self.layer_2 = NodeToNodeLayer(node_feature_dim // 2, final_node_dim)

    def forward(self, graph: Graph) -> Graph:
        graph = self.layer_1(graph)
        graph = self.layer_2(graph)
        return graph
