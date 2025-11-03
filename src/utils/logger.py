"""
Logging utilities for the project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(verbosity: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Setup project logger with console and file handlers.
    
    Args:
        verbosity: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
    # Get logger
    logger = logging.getLogger('traffic_prediction')
    logger.setLevel(getattr(logging, verbosity.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, verbosity.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_experiment_log_file(log_dir: str = "./logs") -> str:
    """
    Create a timestamped log file for an experiment.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Path to created log file
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    return str(log_file)