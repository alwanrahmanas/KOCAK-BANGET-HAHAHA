"""
Helper Utilities
================
"""

import pandas as pd
from typing import List, Dict, Any


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate DataFrame has required columns"""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def format_bytes(size: int) -> str:
    """Format bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (ZeroDivisionError, TypeError):
        return default
