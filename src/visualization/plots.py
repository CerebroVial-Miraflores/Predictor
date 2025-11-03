"""
Plotting utilities for spatiotemporal data visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, List


def plot_predictions_vs_ground_truth(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    node_idx: int = 0,
    sample_idx: int = 0,
    figsize: tuple = (14, 7),
    save_path: Optional[str] = None
):
    """
    Plot predictions against ground truth for a specific node and sample.
    
    Args:
        predictions: Predicted values [samples, horizon, nodes, features]
        ground_truth: True values [samples, horizon, nodes, features]
        node_idx: Index of node to plot
        sample_idx: Index of sample to plot
        figsize: Figure size
        save_path: Optional path to save figure
    """
    pred = predictions[sample_idx, :, node_idx, 0].cpu().numpy()
    gt = ground_truth[sample_idx, :, node_idx, 0].cpu().numpy()
    
    plt.figure(figsize=figsize)
    plt.plot(gt, 'o-', label='Ground Truth', color='royalblue', alpha=0.7)
    plt.plot(pred, 's--', label='Prediction', color='orangered', alpha=0.7)
    
    plt.title(f'Prediction vs Ground Truth (Sample {sample_idx}, Node {node_idx})')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_error_distribution(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None
):
    """
    Plot error distribution histogram and error over time.
    
    Args:
        predictions: Predicted values
        ground_truth: True values
        figsize: Figure size
        save_path: Optional path to save figure
    """
    errors = (predictions - ground_truth).cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Error histogram
    axes[0].hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Prediction Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Error over time
    error_over_time = np.abs(errors).mean(axis=(0, 2, 3))
    axes[1].plot(error_over_time, marker='o', color='darkorange')
    axes[1].set_xlabel('Prediction Horizon (steps)')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].set_title('Error by Prediction Horizon')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_multiple_sensors(
    data: torch.Tensor,
    sensor_indices: List[int],
    labels: Optional[List[str]] = None,
    figsize: tuple = (15, 5),
    title: str = "Sensor Comparison",
    save_path: Optional[str] = None
):
    """
    Plot time series for multiple sensors.
    
    Args:
        data: Time series data [time, nodes, features]
        sensor_indices: List of sensor indices to plot
        labels: Optional labels for each sensor
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=figsize)
    
    for i, idx in enumerate(sensor_indices):
        sensor_data = data[:, idx, 0].cpu().numpy()
        label = labels[i] if labels else f'Sensor #{idx}'
        plt.plot(sensor_data, label=label, alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()