"""
Prediction script for making forecasts with trained STGNN model.

Usage:
    python scripts/predict.py --checkpoint logs/checkpoints/best-model.ckpt --output predictions.csv
"""

import argparse
import yaml
from pathlib import Path
import sys

import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_loader import DatasetLoader
from src.data.data_module import create_data_module
from src.models.time_then_space import create_model
from src.training.predictor import create_predictor
from src.evaluation.metrics import create_metrics_dict
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions with trained STGNN')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output file for predictions')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Number of samples to predict (default: all test set)')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def make_predictions(predictor, dm, trainer, logger, n_samples=None):
    """
    Generate predictions using the trained model.
    
    Args:
        predictor: Trained predictor module
        dm: Data module
        trainer: PyTorch Lightning trainer
        logger: Logger instance
        n_samples: Number of samples to predict (None for all)
        
    Returns:
        Tuple of (predictions, ground_truth, timestamps)
    """
    logger.info("Generating predictions...")
    
    test_dataloader = dm.test_dataloader()
    predictions_list = trainer.predict(model=predictor, dataloaders=test_dataloader)
    
    # Concatenate batches
    predictions = torch.cat([batch['y_hat'] for batch in predictions_list])
    
    # Limit to n_samples if specified
    if n_samples is not None:
        predictions = predictions[:n_samples]
    
    # Get ground truth
    ground_truth = torch.stack([sample.target['y'] for sample in dm.testset])
    if n_samples is not None:
        ground_truth = ground_truth[:n_samples]
    
    logger.info(f"Generated {len(predictions)} predictions")
    
    return predictions, ground_truth


def save_predictions(predictions, ground_truth, output_path, logger):
    """
    Save predictions to CSV file.
    
    Args:
        predictions: Model predictions
        ground_truth: True values
        output_path: Path to save CSV
        logger: Logger instance
    """
    logger.info(f"Saving predictions to {output_path}...")
    
    # Convert to numpy
    pred_np = predictions.cpu().numpy()
    gt_np = ground_truth.cpu().numpy()
    
    # Flatten arrays for CSV
    n_samples, horizon, n_nodes, n_features = pred_np.shape
    
    # Create DataFrame
    records = []
    for sample_idx in range(n_samples):
        for time_idx in range(horizon):
            for node_idx in range(n_nodes):
                records.append({
                    'sample_id': sample_idx,
                    'time_step': time_idx,
                    'node_id': node_idx,
                    'prediction': pred_np[sample_idx, time_idx, node_idx, 0],
                    'ground_truth': gt_np[sample_idx, time_idx, node_idx, 0],
                    'error': pred_np[sample_idx, time_idx, node_idx, 0] - gt_np[sample_idx, time_idx, node_idx, 0]
                })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved {len(df)} predictions to {output_path}")


def main():
    """Main prediction function."""
    args = parse_args()
    config = load_config(args.config)
    
    logger = setup_logger(config['logging']['verbosity'])
    logger.info("="*60)
    logger.info("Starting Prediction")
    logger.info("="*60)
    
    # Load dataset
    logger.info("\n[1/3] Loading dataset...")
    data_loader = DatasetLoader(config['data'])
    dataset = data_loader.load_dataset()
    connectivity = data_loader.compute_connectivity()
    torch_dataset = data_loader.create_torch_dataset()
    
    dm = create_data_module(torch_dataset, config)
    dm.setup()
    
    # Create model
    logger.info("\n[2/3] Loading model...")
    model = create_model(
        config['model'],
        n_nodes=torch_dataset.n_nodes,
        input_size=torch_dataset.n_channels
    )
    
    metrics = create_metrics_dict(config['evaluation'])
    predictor = create_predictor(model, config, metrics)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    predictor.load_model(args.checkpoint)
    predictor.freeze()
    
    # Setup trainer
    trainer = pl.Trainer(
        accelerator=config['training']['accelerator'],
        devices=1,
        logger=False
    )
    
    # Make predictions
    logger.info("\n[3/3] Making predictions...")
    predictions, ground_truth = make_predictions(
        predictor, dm, trainer, logger, args.n_samples
    )
    
    # Save predictions
    save_predictions(predictions, ground_truth, args.output, logger)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("PREDICTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Predictions saved to: {args.output}")
    logger.info(f"Number of predictions: {len(predictions)}")
    logger.info("="*60 + "\n")
    
    logger.info("✓ Prediction completed successfully!")


if __name__ == '__main__':
    main()