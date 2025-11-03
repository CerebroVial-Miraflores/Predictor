"""
PyTorch Lightning DataModule for spatiotemporal datasets.
"""

from typing import Dict, Any
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
from tsl.data.preprocessing import StandardScaler


def create_data_module(torch_dataset, config: Dict[str, Any]) -> SpatioTemporalDataModule:
    """
    Create a configured SpatioTemporalDataModule.
    
    Args:
        torch_dataset: SpatioTemporalDataset object
        config: Configuration dictionary
        
    Returns:
        Configured data module
    """
    # Setup scalers
    scalers = {
        'target': StandardScaler(axis=(0, 1))  # Normalize over time and nodes
    }
    
    # Setup splitter
    splitter = TemporalSplitter(
        val_len=config['data']['val_split'],
        test_len=config['data']['test_split']
    )
    
    # Create data module
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=config['training']['batch_size'],
        workers=config['training']['num_workers']
    )
    
    return dm