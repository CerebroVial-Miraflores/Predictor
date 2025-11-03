"""
Data preprocessing utilities.
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def fill_missing_values(
    data: np.ndarray,
    method: str = 'linear',
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Fill missing values in time series data.
    
    Args:
        data: Input data array [time, nodes, features]
        method: Interpolation method ('linear', 'forward', 'backward', 'mean')
        mask: Binary mask indicating valid values
        
    Returns:
        Data with filled missing values
    """
    if mask is None:
        return data
    
    filled_data = data.copy()
    
    for node_idx in range(data.shape[1]):
        for feat_idx in range(data.shape[2]):
            series = data[:, node_idx, feat_idx]
            node_mask = mask[:, node_idx, feat_idx]
            
            if method == 'linear':
                # Linear interpolation
                valid_indices = np.where(node_mask)[0]
                if len(valid_indices) > 1:
                    filled_data[:, node_idx, feat_idx] = np.interp(
                        np.arange(len(series)),
                        valid_indices,
                        series[valid_indices]
                    )
            elif method == 'forward':
                # Forward fill
                filled_data[:, node_idx, feat_idx] = pd.Series(series).fillna(method='ffill').values
            elif method == 'backward':
                # Backward fill
                filled_data[:, node_idx, feat_idx] = pd.Series(series).fillna(method='bfill').values
            elif method == 'mean':
                # Mean imputation
                mean_val = series[node_mask].mean()
                filled_data[~node_mask, node_idx, feat_idx] = mean_val
    
    return filled_data


def normalize_data(
    data: np.ndarray,
    method: str = 'standard',
    axis: Optional[Tuple[int, ...]] = (0, 1)
) -> Tuple[np.ndarray, dict]:
    """
    Normalize data using specified method.
    
    Args:
        data: Input data array
        method: Normalization method ('standard', 'minmax')
        axis: Axes over which to compute statistics
        
    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    if method == 'standard':
        mean = data.mean(axis=axis, keepdims=True)
        std = data.std(axis=axis, keepdims=True)
        normalized = (data - mean) / (std + 1e-8)
        params = {'mean': mean, 'std': std, 'method': 'standard'}
    elif method == 'minmax':
        min_val = data.min(axis=axis, keepdims=True)
        max_val = data.max(axis=axis, keepdims=True)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def denormalize_data(
    data: np.ndarray,
    params: dict
) -> np.ndarray:
    """
    Denormalize data using stored parameters.
    
    Args:
        data: Normalized data
        params: Normalization parameters from normalize_data
        
    Returns:
        Denormalized data
    """
    if params['method'] == 'standard':
        return data * params['std'] + params['mean']
    elif params['method'] == 'minmax':
        return data * (params['max'] - params['min']) + params['min']
    else:
        raise ValueError(f"Unknown normalization method: {params['method']}")


def create_temporal_features(
    timestamps: pd.DatetimeIndex
) -> np.ndarray:
    """
    Create temporal features from timestamps.
    
    Args:
        timestamps: DatetimeIndex with timestamps
        
    Returns:
        Array of temporal features [time, features]
    """
    features = []
    
    # Hour of day (cyclical)
    hour = timestamps.hour
    features.append(np.sin(2 * np.pi * hour / 24))
    features.append(np.cos(2 * np.pi * hour / 24))
    
    # Day of