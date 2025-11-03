"""
Predictor module for training and inference.
"""

import torch
from typing import Dict, Any
from tsl.engines import Predictor
from tsl.metrics.torch import MaskedMAE


def create_predictor(model, config: Dict[str, Any], metrics: Dict) -> Predictor:
    """
    Create a configured Predictor module.
    
    Args:
        model: Neural network model
        config: Configuration dictionary
        metrics: Dictionary of evaluation metrics
        
    Returns:
        Configured Predictor
    """
    # Loss function
    loss_fn = MaskedMAE()
    
    # Optimizer configuration
    optim_class = getattr(torch.optim, config['training']['optimizer'].capitalize())
    optim_kwargs = {
        'lr': config['training']['learning_rate'],
        'weight_decay': config['training'].get('weight_decay', 0.0)
    }
    
    # Create predictor
    predictor = Predictor(
        model=model,
        optim_class=optim_class,
        optim_kwargs=optim_kwargs,
        loss_fn=loss_fn,
        metrics=metrics
    )
    
    return predictor