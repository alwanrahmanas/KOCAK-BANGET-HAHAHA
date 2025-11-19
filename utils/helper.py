"""
Helper functions for BPJS Fraud Detection
"""

import json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime


def safe_json_serialize(obj: Any) -> Any:
    """
    Safely serialize objects to JSON-compatible format
    
    Args:
        obj: Object to serialize
    
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    return obj


def format_currency(value: float, currency: str = 'Rp') -> str:
    """
    Format numeric value as currency
    
    Args:
        value: Numeric value
        currency: Currency symbol
    
    Returns:
        Formatted currency string
    """
    if value is None or pd.isna(value):
        return "N/A"
    
    return f"{currency} {value:,.0f}"


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    raise_error: bool = True
) -> Dict[str, Any]:
    """
    Validate DataFrame has required columns
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        raise_error: Whether to raise error on validation failure
    
    Returns:
        Dictionary with validation results
    """
    missing_cols = set(required_columns) - set(df.columns)
    
    result = {
        'is_valid': len(missing_cols) == 0,
        'missing_columns': list(missing_cols),
        'total_rows': len(df),
        'total_columns': len(df.columns)
    }
    
    if not result['is_valid'] and raise_error:
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )
    
    return result


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default config path
        current_dir = Path(__file__).parent.parent
        config_path = current_dir / 'config' / 'settings.yaml'
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_output_filename(prefix: str, extension: str = 'csv') -> str:
    """
    Create timestamped output filename
    
    Args:
        prefix: Filename prefix
        extension: File extension
    
    Returns:
        Timestamped filename
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{timestamp}.{extension}"


def merge_prediction_with_input(
    predictions: pd.DataFrame,
    original_data: pd.DataFrame,
    primary_key: str = 'claim_id',
    how: str = 'left'
) -> pd.DataFrame:
    """
    Merge prediction results with original input data
    
    Args:
        predictions: DataFrame with prediction results
        original_data: Original input DataFrame
        primary_key: Column name to use as join key
        how: Type of merge (left, right, inner, outer)
    
    Returns:
        Merged DataFrame with predictions + original data
    """
    if primary_key not in predictions.columns:
        raise ValueError(f"Primary key '{primary_key}' not found in predictions")
    
    if primary_key not in original_data.columns:
        raise ValueError(f"Primary key '{primary_key}' not found in original_data")
    
    # Get columns to avoid duplicates (except primary key)
    pred_cols = [col for col in predictions.columns if col not in original_data.columns or col == primary_key]
    
    # Merge
    merged = original_data.merge(
        predictions[pred_cols],
        on=primary_key,
        how=how
    )
    
    return merged
