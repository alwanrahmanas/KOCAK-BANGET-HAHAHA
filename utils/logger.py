"""
Logging configuration for BPJS Fraud Detection
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = 'bpjs_fraud_detection',
    log_file: str = None,
    level: int = logging.INFO,
    format_string: str = None
):
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        format_string: Log format string
    
    Returns:
        logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # ========== CONSOLE HANDLER WITH UTF-8 ENCODING ==========
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Force UTF-8 encoding for console (Windows fix)
    if sys.platform == 'win32':
        # Reconfigure stdout to use UTF-8
        sys.stdout.reconfigure(encoding='utf-8')
    
    logger.addHandler(console_handler)
    # =========================================================
    
    # File handler
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Use UTF-8 encoding for file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'bpjs_fraud_detection'):
    """
    Get existing logger or create new one
    
    Args:
        name: Logger name
    
    Returns:
        logger: Logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Setup default logger if not exists
        logger = setup_logger(name)
    
    return logger
