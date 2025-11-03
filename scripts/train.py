"""
Training script for Spatiotemporal Graph Neural Network.
"""

import argparse
import yaml
import logging
from pathlib import Path
import sys

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_loader import DatasetLoader
from src.data.data_module import create_data_module
from src.models.time_then_space import create_model
from src.training.predictor import create_predictor
from src.evaluation.metrics import create_metrics_dict
from src.utils.logger import setup_logger


def main():
    """Main training function."""
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logger(config['logging']['verbosity'])
    logger.info("Starting training...")
    
    # Load data
    data_loader = DatasetLoader(config['data'])
    dataset = data_loader.load_dataset()
    connectivity = data_loader.compute_connectivity()
    torch_dataset = data_loader.create_torch_dataset()
    
    # Create data module
    dm = create_data_module(torch_dataset, config)
    dm.setup()
    
    # Create model
    model = create_model(
        config['model'],
        n_nodes=torch_dataset.n_nodes,
        input_size=torch_dataset.n_channels
    )
    
    # Setup metrics
    metrics = create_metrics_dict(config['evaluation'])
    
    # Create predictor
    predictor = create_predictor(model, config, metrics)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='logs/checkpoints',
        monitor='val_mae',
        mode='min',
        save_top_k=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_mae',
        patience=10,
        mode='min'
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['training']['accelerator'],
        callbacks=[checkpoint_callback, early_stop],
        logger=TensorBoardLogger('logs', name='traffic_stgnn')
    )
    
    # Train
    trainer.fit(predictor, datamodule=dm)
    
    # Test
    trainer.test(predictor, datamodule=dm)


if __name__ == '__main__':
    main()