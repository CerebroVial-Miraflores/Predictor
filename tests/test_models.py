"""
Unit tests for model architectures.
"""

import pytest
import torch

from src.models.time_then_space import TimeThenSpaceModel, create_model


def test_model_initialization():
    """Test TimeThenSpaceModel initialization."""
    model = TimeThenSpaceModel(
        input_size=1,
        n_nodes=207,
        horizon=12,
        emb_size=16,
        hidden_size=32,
        rnn_layers=1,
        gnn_kernel=2
    )
    
    assert model.input_size == 1
    assert model.n_nodes == 207
    assert model.horizon == 12
    assert model.emb_size == 16
    assert model.hidden_size == 32


def test_model_forward_pass():
    """Test model forward pass."""
    batch_size = 4
    time_steps = 12
    n_nodes = 207
    n_features = 1
    
    model = TimeThenSpaceModel(
        input_size=n_features,
        n_nodes=n_nodes,
        horizon=12,
        emb_size=16,
        hidden_size=32
    )
    
    # Create dummy input
    x = torch.randn(batch_size, time_steps, n_nodes, n_features)
    edge_index = torch.randint(0, n_nodes, (2, 100))
    edge_weight = torch.rand(100)
    
    # Forward pass
    output = model(x, edge_index, edge_weight)
    
    # Check output shape
    assert output.shape == (batch_size, 12, n_nodes, n_features)


def test_model_parameters():
    """Test that model has trainable parameters."""
    model = TimeThenSpaceModel(
        input_size=1,
        n_nodes=207,
        horizon=12
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert total_params > 0
    assert trainable_params > 0
    assert total_params == trainable_params


def test_create_model_function():
    """Test model factory function."""
    config = {
        'horizon': 12,
        'emb_size': 16,
        'hidden_size': 32,
        'rnn_layers': 1,
        'gnn_kernel': 2
    }
    
    model = create_model(config, n_nodes=207, input_size=1)
    
    assert isinstance(model, TimeThenSpaceModel)
    assert model.n_nodes == 207


if __name__ == '__main__':
    pytest.main([__file__])