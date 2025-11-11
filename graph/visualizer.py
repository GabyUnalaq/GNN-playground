import plotly.graph_objects as go
import numpy as np
import networkx as nx
from .graph import Graph


__all__ = ['GraphVisualizer', 'visualize_graph']


def visualize_graph(graph: Graph, save: bool = False):
    print("Creating visualization...")
    visualizer = GraphVisualizer(graph)
    
    visualizer.show(
        layout='spring',
        node_size=25,
        edge_width=2.5,
        title="Interactive Graph - Hover over nodes and edges!"
    )
    
    # Save to HTML file
    if save:
        visualizer.save('graph_visualization.html', layout='spring')


class GraphVisualizer:
    """
    Interactive graph visualization with hover info for node and edge embeddings.
    Uses Plotly for interactive visualization.
    """
    
    def __init__(self, graph: Graph):
        """
        Initialize visualizer with a Graph object.
        
        Args:
            graph: Graph object to visualize
        """
        self.graph = graph
        self.pos = None  # Node positions
    
    def compute_layout(self, layout: str = 'spring', seed: int = 42):
        """
        Compute node positions using various layout algorithms.
        
        Args:
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'random')
            seed: Random seed for reproducibility
        """
        # Create NetworkX graph for layout computation
        G = nx.DiGraph()
        G.add_nodes_from(range(self.graph.num_nodes))
        
        edges = self.graph.edge_index.t().cpu().numpy()
        G.add_edges_from(edges)
        
        # Compute layout
        if layout == 'spring':
            self.pos = nx.spring_layout(G, seed=seed)
        elif layout == 'circular':
            self.pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            if len(G.nodes()) > 1:
                self.pos = nx.kamada_kawai_layout(G)
            else:
                self.pos = {0: (0, 0)}
        elif layout == 'random':
            self.pos = nx.random_layout(G, seed=seed)
        else:
            raise ValueError(f"Unknown layout: {layout}")
    
    def visualize(self, 
                  layout: str = 'spring',
                  node_size: int = 20,
                  edge_width: float = 2.0,
                  show_edge_labels: bool = False,
                  title: str = "Interactive Graph Visualization",
                  width: int = 1000,
                  height: int = 800):
        """
        Create interactive visualization of the graph.
        
        Args:
            layout: Layout algorithm for node positioning
            node_size: Size of nodes
            edge_width: Width of edges
            show_edge_labels: Whether to show edge feature summaries as labels
            title: Plot title
            width: Plot width in pixels
            height: Plot height in pixels
        
        Returns:
            Plotly figure object
        """
        # Compute layout if not already done
        if self.pos is None:
            self.compute_layout(layout)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges (returns list of traces now)
        edge_traces = self._create_edge_trace(edge_width)
        for trace in edge_traces:
            fig.add_trace(trace)
        
        # Add edge hover points (invisible points at edge midpoints for hover)
        if self.graph.num_edges > 0:
            edge_hover_trace = self._create_edge_hover_trace()
            fig.add_trace(edge_hover_trace)
        
        # Add nodes
        node_trace = self._create_node_trace(node_size)
        fig.add_trace(node_trace)
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            width=width,
            height=height,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            plot_bgcolor='white'
        )
        
        return fig
    
    def _create_edge_trace(self, edge_width: float):
        """Create edge trace for visualization with curved bidirectional edges."""
        edge_index = self.graph.edge_index.cpu().numpy()
        
        # Find bidirectional edges
        bidirectional = set()
        for i in range(self.graph.num_edges):
            src, tgt = edge_index[0, i], edge_index[1, i]
            # Check if reverse edge exists
            for j in range(self.graph.num_edges):
                if edge_index[0, j] == tgt and edge_index[1, j] == src:
                    if (min(src, tgt), max(src, tgt)) not in bidirectional:
                        bidirectional.add((min(src, tgt), max(src, tgt)))
        
        edge_traces = []
        
        for i in range(self.graph.num_edges):
            src, tgt = edge_index[0, i], edge_index[1, i]
            x0, y0 = self.pos[src]
            x1, y1 = self.pos[tgt]
            
            # Check if this edge is part of a bidirectional pair
            is_bidirectional = (min(src, tgt), max(src, tgt)) in bidirectional
            
            if is_bidirectional and src > tgt:
                # For one direction of bidirectional edge, create curved path
                edge_x, edge_y = self._create_curved_edge(x0, y0, x1, y1, curve_offset=0.15)
            else:
                # Straight edge
                edge_x = [x0, x1, None]
                edge_y = [y0, y1, None]
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=edge_width, color='#888'),
                hoverinfo='none',
                mode='lines',
                showlegend=False,
                name='edges'
            )
            edge_traces.append(edge_trace)
        
        return edge_traces
    
    def _create_curved_edge(self, x0, y0, x1, y1, curve_offset=0.15):
        """Create a curved edge path using quadratic bezier curve."""
        # Calculate midpoint
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        
        # Calculate perpendicular offset
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        
        if length > 0:
            # Perpendicular direction
            perp_x = -dy / length
            perp_y = dx / length
            
            # Control point for bezier curve
            ctrl_x = mid_x + perp_x * curve_offset
            ctrl_y = mid_y + perp_y * curve_offset
            
            # Generate curve points
            t = np.linspace(0, 1, 20)
            curve_x = (1-t)**2 * x0 + 2*(1-t)*t * ctrl_x + t**2 * x1
            curve_y = (1-t)**2 * y0 + 2*(1-t)*t * ctrl_y + t**2 * y1
            
            return list(curve_x) + [None], list(curve_y) + [None]
        else:
            return [x0, x1, None], [y0, y1, None]
    
    def _create_edge_hover_trace(self):
        """Create invisible points at edge midpoints for hover information."""
        edge_index = self.graph.edge_index.cpu().numpy()
        edge_features = self.graph.edge_features.cpu().numpy()
        
        # Find bidirectional edges
        bidirectional = set()
        for i in range(self.graph.num_edges):
            src, tgt = edge_index[0, i], edge_index[1, i]
            for j in range(self.graph.num_edges):
                if edge_index[0, j] == tgt and edge_index[1, j] == src:
                    if (min(src, tgt), max(src, tgt)) not in bidirectional:
                        bidirectional.add((min(src, tgt), max(src, tgt)))
        
        mid_x = []
        mid_y = []
        hover_text = []
        
        for i in range(self.graph.num_edges):
            src, tgt = edge_index[0, i], edge_index[1, i]
            x0, y0 = self.pos[src]
            x1, y1 = self.pos[tgt]
            
            # Check if bidirectional
            is_bidirectional = (min(src, tgt), max(src, tgt)) in bidirectional
            
            if is_bidirectional and src > tgt:
                # For curved edge, place hover point on the curve
                curve_x, curve_y = self._create_curved_edge(x0, y0, x1, y1, curve_offset=0.15)
                mid_idx = len(curve_x) // 2
                mid_x.append(curve_x[mid_idx])
                mid_y.append(curve_y[mid_idx])
            else:
                # Regular midpoint
                mid_x.append((x0 + x1) / 2)
                mid_y.append((y0 + y1) / 2)
            
            # Hover text
            edge_feat_str = np.array2string(edge_features[i], precision=3, separator=', ')
            text = f"<b>Edge {i}</b><br>"
            text += f"From Node {src} → To Node {tgt}<br>"
            text += f"<b>Edge Features:</b><br>{edge_feat_str}"
            hover_text.append(text)
        
        edge_hover_trace = go.Scatter(
            x=mid_x, y=mid_y,
            mode='markers',
            hoverinfo='text',
            text=hover_text,
            marker=dict(
                size=8,
                color='rgba(200, 200, 200, 0.3)',
                line=dict(width=0)
            ),
            showlegend=False,
            name='edge_info'
        )
        
        return edge_hover_trace
    
    def _create_node_trace(self, node_size: int):
        """Create node trace for visualization."""
        node_x = []
        node_y = []
        hover_text = []
        node_colors = []
        
        node_features = self.graph.node_features.cpu().numpy()
        
        for node_idx in range(self.graph.num_nodes):
            x, y = self.pos[node_idx]
            node_x.append(x)
            node_y.append(y)
            
            # Create hover text with node features
            node_feat_str = np.array2string(node_features[node_idx], precision=3, separator=', ')
            text = f"<b>Node {node_idx}</b><br>"
            text += f"<b>Node Features:</b><br>{node_feat_str}<br>"
            
            # Add degree information
            in_degree = (self.graph.edge_index[1] == node_idx).sum().item()
            out_degree = (self.graph.edge_index[0] == node_idx).sum().item()
            text += f"<b>Degree:</b> in={in_degree}, out={out_degree}"
            hover_text.append(text)
            
            # Color by degree
            total_degree = in_degree + out_degree
            node_colors.append(total_degree)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[str(i) for i in range(self.graph.num_nodes)],
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            hovertext=hover_text,
            marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=node_size,
            color=node_colors,
            colorbar=dict(
                thickness=15,
                title=dict(text="Node Degree", side='right'),
                xanchor='left'
            ),
            line=dict(width=2, color='white')
            ),
            name='nodes'
        )
        
        return node_trace
    
    def show(self, **kwargs):
        """Create and display the visualization."""
        fig = self.visualize(**kwargs)
        fig.show()
        return fig
    
    def save(self, filename: str, **kwargs):
        """Save visualization to HTML file."""
        fig = self.visualize(**kwargs)
        fig.write_html(filename)
        print(f"Visualization saved to {filename}")
        return fig


# Example usage
if __name__ == "__main__":
    # Create a sample graph
    print("Creating sample graph...\n")
    config = {
        'num_nodes': 8,
        'node_feature_dim': 4,
        'edge_feature_dim': 2,
        'node_init': 'random',
        'edge_init': 'uniform',
        'edges': [
            [0, 1], [1, 0], [1, 2], [2, 3], [3, 0],  # Square with bidirectional edge
            [4, 5], [5, 6], [6, 7], [7, 4],  # Another square
            [0, 4], [2, 6]  # Connect the two squares
        ],
    }
    
    graph = Graph(config=config)
    graph.summary()
    
    # Create visualizer
    print("\n\nCreating visualization...")
    visualizer = GraphVisualizer(graph)
    
    # Show interactive plot
    print("\nGenerating interactive plot...")
    print("Hover over nodes to see their features and connections!")
    print("Hover over the faint points on edges to see edge features!")
    
    # Try different layouts
    fig = visualizer.show(
        layout='spring',
        node_size=25,
        edge_width=2.5,
        title="Interactive Graph - Hover over nodes and edges!"
    )
    
    # Save to HTML file
    visualizer.save('graph_visualization.html', layout='spring')