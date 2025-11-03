"""
Custom metrics for spatiotemporal forecasting evaluation.
"""

import torch
from tsl.metrics.torch import MaskedMetric
from torchmetrics.functional.regression.r2 import _r2_score_update, _r2_score_compute


class MaskedR2(MaskedMetric):
    """
    R² Score metric with support for masked values.
    
    Computes the coefficient of determination while ignoring masked entries.
    """
    
    is_differentiable: bool = False
    higher_is_better: bool = True
    
    def __init__(self, **kwargs):
        """Initialize the Masked R² metric."""
        super(MaskedR2, self).__init__(metric_fn=lambda y_hat, y: None, **kwargs)
        
        # Register the four states that R2Score needs
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("residual", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> None:
        """
        Update metric state with new predictions and targets.
        
        Args:
            y_hat: Predicted values
            y: True values
            mask: Binary mask indicating valid entries
        """
        # Apply mask first
        if mask is not None:
            mask = mask.bool()
            y_hat = y_hat[mask]
            y = y[mask]
        
        # Unpack the four values returned by the update function
        sum_squared_error, sum_error, residual, total = _r2_score_update(y_hat, y)
        
        # Accumulate each state
        self.sum_squared_error += sum_squared_error
        self.sum_error += sum_error
        self.residual += residual
        self.total += total
    
    def compute(self) -> torch.Tensor:
        """
        Compute the final R² score.
        
        Returns:
            R² score value
        """
        return _r2_score_compute(
            self.sum_squared_error,
            self.sum_error,
            self.residual,
            self.total
        )


class MaskedRMSE(MaskedMetric):
    """
    Root Mean Squared Error metric with support for masked values.
    """
    
    is_differentiable: bool = True
    higher_is_better: bool = False
    
    def __init__(self, **kwargs):
        """Initialize the Masked RMSE metric."""
        def rmse_fn(y_hat, y):
            return torch.sqrt(torch.mean((y_hat - y) ** 2))
        
        super(MaskedRMSE, self).__init__(metric_fn=rmse_fn, **kwargs)


def create_metrics_dict(config: dict) -> dict:
    """
    Create a dictionary of metrics based on configuration.
    
    Args:
        config: Configuration dictionary with metric specifications
        
    Returns:
        Dictionary mapping metric names to metric objects
    """
    from tsl.metrics.torch import MaskedMAE, MaskedMAPE
    
    metrics = {}
    
    # Base metrics
    if 'mae' in config.get('metrics', []):
        metrics['mae'] = MaskedMAE()
    
    if 'mape' in config.get('metrics', []):
        metrics['mape'] = MaskedMAPE()
    
    if 'rmse' in config.get('metrics', []):
        metrics['rmse'] = MaskedRMSE()
    
    if 'r2' in config.get('metrics', []):
        metrics['r2'] = MaskedR2()
    
    # Temporal horizon metrics
    eval_horizons = config.get('eval_horizons', [])
    for horizon_minutes in eval_horizons:
        # Convert minutes to timestep index (5 min per step)
        step_idx = (horizon_minutes // 5) - 1
        metrics[f'mae_at_{horizon_minutes}min'] = MaskedMAE(at=step_idx)
    
    return metrics


def print_metrics_summary(metrics_dict: dict):
    """
    Print a summary of configured metrics.
    
    Args:
        metrics_dict: Dictionary of metric objects
    """
    print("\n" + "="*60)
    print("CONFIGURED METRICS")
    print("="*60)
    
    base_metrics = [k for k in metrics_dict.keys() if '_at_' not in k]
    temporal_metrics = [k for k in metrics_dict.keys() if '_at_' in k]
    
    if base_metrics:
        print("Base metrics:")
        for metric_name in base_metrics:
            print(f"  - {metric_name}")
    
    if temporal_metrics:
        print("\nTemporal evaluation metrics:")
        for metric_name in temporal_metrics:
            print(f"  - {metric_name}")
    
    print("="*60 + "\n")