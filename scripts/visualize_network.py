"""
Network visualization script for traffic sensor network.

Usage:
    python scripts/visualize_network.py --config config/config.yaml
"""

import argparse
import yaml
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_loader import DatasetLoader
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize traffic sensor network')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='network_visualization.png',
                        help='Output file path')
    parser.add_argument('--locations', type=str, default='data/locations.csv',
                        help='Path to locations CSV')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def plot_geographic_network(locations_df, edge_index, edge_weight, output_path, config):
    """
    Plot the geographic sensor network.
    
    Args:
        locations_df: DataFrame with sensor coordinates
        edge_index: Graph connectivity
        edge_weight: Edge weights
        output_path: Path to save the plot
        config: Configuration dictionary
    """
    # Create position dictionary for NetworkX
    pos = {i: (lon, lat) for i, lon, lat in zip(
        locations_df.index, 
        locations_df['longitude'], 
        locations_df['latitude']
    )}
    
    num_nodes = len(locations_df)
    
    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    for i in range(edge_index.shape[1]):
        source = edge_index[0, i].item()
        target = edge_index[1, i].item()
        weight = edge_weight[i].item()
        G.add_edge(source, target, weight=weight)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Extract weights for visualization
    weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_color='royalblue',
        node_size=config['visualization']['node_size'],
        ax=ax,
        alpha=0.8
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edge_color=weights,
        width=weights * config['visualization']['edge_width_multiplier'],
        edge_cmap=plt.cm.viridis,
        alpha=config['visualization']['edge_alpha'],
        ax=ax
    )
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=plt.Normalize(vmin=weights.min(), vmax=weights.max())
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label('Connection Strength (Edge Weight)', rotation=270, labelpad=20)
    
    # Set labels and title
    ax.set_title('Geographic Traffic Sensor Network', fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=config['visualization']['dpi'], bbox_inches='tight')
    plt.close()
    
    return G


def plot_network_statistics(G, output_dir):
    """
    Plot network statistics.
    
    Args:
        G: NetworkX graph
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Compute statistics
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    clustering_coeffs = list(nx.clustering(G).values())
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Degree distribution
    axes[0, 0].hist(degree_sequence, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Node Degree')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Degree Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Degree rank
    axes[0, 1].plot(degree_sequence, marker='o', markersize=3)
    axes[0, 1].set_xlabel('Node Rank')
    axes[0, 1].set_ylabel('Degree')
    axes[0, 1].set_title('Degree Rank Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Clustering coefficient distribution
    axes[1, 0].hist(clustering_coeffs, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Clustering Coefficient')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Clustering Coefficient Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Network statistics summary
    stats_text = f"""
    Network Statistics
    ─────────────────
    Nodes: {G.number_of_nodes()}
    Edges: {G.number_of_edges()}
    
    Avg Degree: {np.mean(degree_sequence):.2f}
    Max Degree: {max(degree_sequence)}
    Min Degree: {min(degree_sequence)}
    
    Avg Clustering: {np.mean(clustering_coeffs):.4f}
    Density: {nx.density(G):.4f}
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    stats_path = output_path / 'network_statistics.png'
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return stats_path


def main():
    """Main visualization function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logger(config['logging']['verbosity'])
    logger.info("="*60)
    logger.info("Network Visualization")
    logger.info("="*60)
    
    # Load dataset
    logger.info("\nLoading dataset...")
    data_loader = DatasetLoader(config['data'])
    dataset = data_loader.load_dataset()
    connectivity = data_loader.compute_connectivity()
    
    # Load locations
    logger.info(f"Loading sensor locations from {args.locations}...")
    locations_df = pd.read_csv(args.locations)
    
    edge_index, edge_weight = connectivity
    
    # Plot geographic network
    logger.info("Creating geographic network visualization...")
    G = plot_geographic_network(
        locations_df, 
        edge_index, 
        edge_weight, 
        args.output,
        config
    )
    logger.info(f"✓ Saved geographic network to: {args.output}")
    
    # Plot network statistics
    logger.info("Generating network statistics...")
    output_dir = Path(args.output).parent
    stats_path = plot_network_statistics(G, output_dir)
    logger.info(f"✓ Saved network statistics to: {stats_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("NETWORK SUMMARY")
    logger.info("="*60)
    logger.info(f"Number of nodes: {G.number_of_nodes()}")
    logger.info(f"Number of edges: {G.number_of_edges()}")
    logger.info(f"Average degree: {np.mean([d for n, d in G.degree()]):.2f}")
    logger.info(f"Network density: {nx.density(G):.4f}")
    logger.info(f"Average clustering coefficient: {nx.average_clustering(G):.4f}")
    logger.info("="*60 + "\n")
    
    logger.info("✓ Visualization completed successfully!")


if __name__ == '__main__':
    main()