"""
Helper utilities for the project.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def print_model_size(model: torch.nn.Module):
    """
    Print model size information.
    
    Args:
        model: PyTorch model
    """
    params = count_parameters(model)
    print("\n" + "="*60)
    print("MODEL SIZE")
    print("="*60)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    print("="*60 + "\n")


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import pytorch_lightning as pl
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    pl.seed_everything(seed, workers=True)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_yaml_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_device(accelerator: str = 'auto') -> str:
    """
    Get the appropriate device for training.
    
    Args:
        accelerator: Accelerator type ('auto', 'cpu', 'gpu', 'mps')
        
    Returns:
        Device string
    """
    if accelerator == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return accelerator


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def create_checkpoint_path(base_dir: str, experiment_name: str) -> Path:
    """
    Create checkpoint directory path.
    
    Args:
        base_dir: Base directory for checkpoints
        experiment_name: Name of the experiment
        
    Returns:
        Path object for checkpoint directory
    """
    checkpoint_dir = Path(base_dir) / 'checkpoints' / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir