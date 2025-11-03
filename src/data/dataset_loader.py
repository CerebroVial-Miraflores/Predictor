"""
Dataset loading and initialization module.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

from tsl.datasets import MetrLA
from tsl.data import SpatioTemporalDataset
import pandas as pd

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Handles loading and initial processing of traffic datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset = None
        self.torch_dataset = None
        self.connectivity = None
        
    def load_dataset(self) -> MetrLA:
        """Load the raw dataset."""
        logger.info(f"Loading {self.config['dataset_name']} dataset...")
        
        root_path = self.config['root_path']
        Path(root_path).mkdir(parents=True, exist_ok=True)
        
        if self.config['dataset_name'] == 'MetrLA':
            self.dataset = MetrLA(root=root_path)
        else:
            raise ValueError(f"Dataset {self.config['dataset_name']} not supported")
        
        self._log_dataset_info()
        return self.dataset
    
    def _log_dataset_info(self):
        """Log dataset information."""
        logger.info(f"Dataset loaded successfully:")
        logger.info(f"  - Shape: {self.dataset.dataframe().shape}")
        logger.info(f"  - Nodes: {self.dataset.n_nodes}")
        logger.info(f"  - Missing values: {(1 - self.dataset.mask.mean()) * 100:.2f}%")
    
    def compute_connectivity(self) -> tuple:
        """Compute connectivity matrix from similarity."""
        logger.info("Computing connectivity matrix...")
        
        self.connectivity = self.dataset.get_connectivity(
            threshold=self.config['threshold'],
            include_self=self.config['include_self'],
            normalize_axis=self.config['normalize_axis'],
            layout="edge_index"
        )
        
        edge_index, edge_weight = self.connectivity
        logger.info(f"  - Number of edges: {edge_index.shape[1]}")
        
        return self.connectivity
    
    def create_torch_dataset(self) -> SpatioTemporalDataset:
        """Create PyTorch-compatible spatiotemporal dataset."""
        logger.info("Creating SpatioTemporalDataset...")
        
        if self.connectivity is None:
            self.compute_connectivity()
        
        self.torch_dataset = SpatioTemporalDataset(
            target=self.dataset.dataframe(),
            connectivity=self.connectivity,
            mask=self.dataset.mask,
            horizon=self.config['horizon'],
            window=self.config['window'],
            stride=self.config['stride']
        )
        
        logger.info(f"  - Total samples: {len(self.torch_dataset)}")
        return self.torch_dataset