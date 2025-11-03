"""
Unit tests for data loading functionality.
"""

import pytest
import tempfile
from pathlib import Path

from src.data.dataset_loader import DatasetLoader


def test_dataset_loader_initialization():
    """Test DatasetLoader initialization."""
    config = {
        'dataset_name': 'MetrLA',
        'root_path': './data',
        'window': 12,
        'horizon': 12,
        'stride': 1,
        'threshold': 0.1,
        'include_self': False,
        'normalize_axis': 1
    }
    
    loader = DatasetLoader(config)
    assert loader.config == config
    assert loader.dataset is None
    assert loader.torch_dataset is None


def test_connectivity_computation():
    """Test connectivity matrix computation."""
    config = {
        'dataset_name': 'MetrLA',
        'root_path': './data',
        'window': 12,
        'horizon': 12,
        'stride': 1,
        'threshold': 0.1,
        'include_self': False,
        'normalize_axis': 1
    }
    
    loader = DatasetLoader(config)
    dataset = loader.load_dataset()
    connectivity = loader.compute_connectivity()
    
    assert connectivity is not None
    assert len(connectivity) == 2  # edge_index and edge_weight
    
    edge_index, edge_weight = connectivity
    assert edge_index.shape[0] == 2  # Source and target nodes
    assert edge_index.shape[1] == edge_weight.shape[0]  # Same number of edges


def test_torch_dataset_creation():
    """Test SpatioTemporalDataset creation."""
    config = {
        'dataset_name': 'MetrLA',
        'root_path': './data',
        'window': 12,
        'horizon': 12,
        'stride': 1,
        'threshold': 0.1,
        'include_self': False,
        'normalize_axis': 1
    }
    
    loader = DatasetLoader(config)
    loader.load_dataset()
    loader.compute_connectivity()
    torch_dataset = loader.create_torch_dataset()
    
    assert torch_dataset is not None
    assert len(torch_dataset) > 0
    assert torch_dataset.n_nodes == 207  # MetrLA has 207 nodes


if __name__ == '__main__':
    pytest.main([__file__])