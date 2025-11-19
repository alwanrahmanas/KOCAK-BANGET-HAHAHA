"""
Configuration module for BPJS Fraud Detection
"""

from pathlib import Path
import yaml
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file. If None, uses default settings.yaml
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default to settings.yaml in this directory
        config_path = Path(__file__).parent / 'settings.yaml'
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_model_config() -> Dict[str, Any]:
    """Get model configuration"""
    config = load_config()
    return config.get('model', {})


def get_paths_config() -> Dict[str, Any]:
    """Get paths configuration"""
    config = load_config()
    return config.get('paths', {})


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration"""
    config = load_config()
    return config.get('logging', {})


def get_shap_config() -> Dict[str, Any]:
    """Get SHAP configuration"""
    config = load_config()
    return config.get('shap', {})


def get_threshold_config() -> Dict[str, Any]:
    """Get threshold optimization configuration"""
    config = load_config()
    return config.get('threshold', {})


__all__ = [
    'load_config',
    'get_model_config',
    'get_paths_config',
    'get_logging_config',
    'get_shap_config',
    'get_threshold_config'
]
