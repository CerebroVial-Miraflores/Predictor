"""
Evaluation script for trained STGNN model.

Usage:
    python scripts/evaluate.py --checkpoint logs/checkpoints/best-model.ckpt
"""

import argparse
import yaml
import logging
from pathlib import Path
import sys

import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_loader import DatasetLoader
from src.data.data_module import create_data_module
from src.models.time_then_space import create_model
from src.training.predictor import create_predictor
from src.evaluation.metrics import create_metrics_dict
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained STGNN model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(predictor, dm, trainer, logger):
    """Evaluate model on test set."""
    logger.info("Running evaluation on test set...")
    
    predictor.freeze()
    test_results = trainer.test(predictor, datamodule=dm)
    
    return test_results[0]


def generate_predictions(predictor, dm, trainer, logger):
    """Generate predictions for test set."""
    logger.info("Generating predictions...")
    
    test_dataloader = dm.test_dataloader()
    predictions_list = trainer.predict(model=predictor, dataloaders=test_dataloader)
    
    predictions = torch.cat([batch['y_hat'] for batch in predictions_list])
    ground_truth = torch.stack([sample.target['y'] for sample in dm.testset])
    
    logger.info(f"Generated {len(predictions)} predictions")
    
    return predictions, ground_truth


def save_results(test_metrics, predictions, ground_truth, output_dir, logger):
    """Save evaluation results to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([test_metrics])
    metrics_path = output_path / 'test_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save predictions summary
    pred_np = predictions.cpu().numpy()
    gt_np = ground_truth.cpu().numpy()
    
    summary = {
        'mean_prediction': pred_np.mean(),
        'std_prediction': pred_np.std(),
        'mean_ground_truth': gt_np.mean(),
        'std_ground_truth': gt_np.std(),
        'mean_error': (pred_np - gt_np).mean(),
        'std_error': (pred_np - gt_np).std()
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = output_path / 'prediction_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved prediction summary to {summary_path}")


def visualize_results(predictions, ground_truth, output_dir, logger):
    """Generate visualization plots."""
    logger.info("Generating visualizations...")
    
    output_path = Path(output_dir) / 'plots'
    output_path.mkdir(parents=True, exist_ok=True)
    
    pred_np = predictions.cpu().numpy()
    gt_np = ground_truth.cpu().numpy()
    
    # Sample predictions plot
    num_samples = min(5, len(predictions))
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 4*num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        sample_pred = pred_np[i, :, 0, 0]
        sample_gt = gt_np[i, :, 0, 0]
        
        ax.plot(sample_gt, 'o-', label='Ground Truth', alpha=0.7)
        ax.plot(sample_pred, 's--', label='Prediction', alpha=0.7)
        ax.set_title(f'Sample {i} - Node 0')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    sample_path = output_path / 'sample_predictions.png'
    plt.savefig(sample_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved sample predictions to {sample_path}")


def main():
    """Main evaluation function."""
    args = parse_args()
    config = load_config(args.config)
    
    logger = setup_logger(config['logging']['verbosity'])
    logger.info("="*60)
    logger.info("Starting Model Evaluation")
    logger.info("="*60)
    
    # Load dataset
    logger.info("\n[1/4] Loading dataset...")
    data_loader = DatasetLoader(config['data'])
    dataset = data_loader.load_dataset()
    connectivity = data_loader.compute_connectivity()
    torch_dataset = data_loader.create_torch_dataset()
    
    dm = create_data_module(torch_dataset, config)
    dm.setup()
    
    # Create model
    logger.info("\n[2/4] Loading model...")
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
    
    # Setup trainer
    trainer = pl.Trainer(
        accelerator=config['training']['accelerator'],
        devices=1,
        logger=False
    )
    
    # Evaluate
    logger.info("\n[3/4] Evaluating model...")
    test_metrics = evaluate_model(predictor, dm, trainer, logger)
    
    # Generate predictions
    predictions, ground_truth = generate_predictions(predictor, dm, trainer, logger)
    
    # Save results
    logger.info("\n[4/4] Saving results...")
    save_results(test_metrics, predictions, ground_truth, args.output_dir, logger)
    
    if args.visualize:
        visualize_results(predictions, ground_truth, args.output_dir, logger)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("\nTest Metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info("="*60 + "\n")
    
    logger.info("✓ Evaluation completed successfully!")


if __name__ == '__main__':
    main()