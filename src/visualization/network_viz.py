"""
Network visualization utilities.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


def create_graph_from_connectivity(
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    n_nodes: int
) -> nx.Graph:
    """
    Create NetworkX graph from edge index and weights.
    
    Args:
        edge_index: Edge connectivity [2, num_edges]
        edge_weight: Edge weights [num_edges]
        n_nodes: Number of nodes
        
    Returns:
        NetworkX Graph object
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    
    for i in range(edge_index.shape[1]):
        source = int(edge_index[0, i])
        target = int(edge_index[1, i])
        weight = float(edge_weight[i])
        G.add_edge(source, target, weight=weight)
    
    return G


def plot_network_graph(
    G: nx.Graph,
    positions: Optional[dict] = None,
    figsize: Tuple[int, int] = (12, 12),
    node_size: int = 50,
    node_color: str = 'royalblue',
    edge_alpha: float = 0.5,
    title: str = "Network Graph",
    save_path: Optional[str] = None
):
    """
    Plot network graph with customizable styling.
    
    Args:
        G: NetworkX graph
        positions: Node positions dictionary {node_id: (x, y)}
        figsize: Figure size
        node_size: Size of nodes
        node_color: Color of nodes
        edge_alpha: Transparency of edges
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use spring layout if positions not provided
    if positions is None:
        positions = nx.spring_layout(G, seed=42)
    
    # Extract edge weights
    weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
    
    # Draw network
    nx.draw_networkx_nodes(
        G, positions,
        node_color=node_color,
        node_size=node_size,
        ax=ax,
        alpha=0.8
    )
    
    nx.draw_networkx_edges(
        G, positions,
        edge_color=weights,
        width=weights * 2,
        edge_cmap=plt.cm.viridis,
        alpha=edge_alpha,
        ax=ax
    )
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=plt.Normalize(vmin=weights.min(), vmax=weights.max())
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label('Edge Weight', rotation=270, labelpad=20)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compute_network_statistics(G: nx.Graph) -> dict:
    """
    Compute various network statistics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of network statistics
    """
    degree_sequence = [d for n, d in G.degree()]
    clustering_coeffs = list(nx.clustering(G).values())
    
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'avg_degree': np.mean(degree_sequence),
        'max_degree': max(degree_sequence),
        'min_degree': min(degree_sequence),
        'avg_clustering': np.mean(clustering_coeffs),
        'density': nx.density(G)
    }
    
    return stats


def print_network_statistics(stats: dict):
    """
    Print network statistics in a formatted way.
    
    Args:
        stats: Dictionary of network statistics
    """
    print("\n" + "="*60)
    print("NETWORK STATISTICS")
    print("="*60)
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print(f"Max degree: {stats['max_degree']}")
    print(f"Min degree: {stats['min_degree']}")
    print(f"Average clustering coefficient: {stats['avg_clustering']:.4f}")
    print(f"Network density: {stats['density']:.4f}")
    print("="*60 + "\n")


def plot_degree_distribution(
    G: nx.Graph,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
):
    """
    Plot degree distribution of the network.
    
    Args:
        G: NetworkX graph
        figsize: Figure size
        save_path: Optional path to save figure
    """
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(degree_sequence, bins=20, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Degree')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Degree Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Rank plot
    axes[1].plot(degree_sequence, marker='o', markersize=3)
    axes[1].set_xlabel('Node Rank')
    axes[1].set_ylabel('Degree')
    axes[1].set_title('Degree Rank Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()