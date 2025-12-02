"""
RAG Utilities
=============
Helper functions untuk RAG integration.
"""

import pandas as pd
from typing import Dict, List


def validate_rag_columns(df: pd.DataFrame) -> bool:
    """Validate DataFrame has required columns untuk RAG"""
    required = [
        'claim_id',
        'predicted_fraud_type',
        'fraud_probability',
        'explanation_summary'
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required RAG columns: {missing}")
    return True


def format_for_vector_db(row: pd.Series) -> Dict:
    """Format single row untuk vector DB indexing"""
    return {
        'id': row['claim_id'],
        'fraud_type': row['predicted_fraud_type'],
        'diagnosis': row.get('diagnosis_primary_desc', ''),
        'procedure': row.get('procedure_primary_desc', ''),
        'text': f"{row.get('diagnosis_primary_desc', '')} {row.get('procedure_primary_desc', '')}",
        'metadata': {
            'probability': float(row['fraud_probability']),
            'faskes_level': row.get('faskes_level', ''),
            'claim_ratio': float(row.get('claim_ratio', 0))
        }
    }
