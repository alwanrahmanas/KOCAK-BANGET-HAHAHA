"""
RAG Workflow
============
Prepare predictions untuk RAG explainer integration.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict
import json

from utils.logger import get_logger
from utils.rag_utils import prepare_rag_context, format_for_vector_db

logger = get_logger(__name__)


def prepare_fraud_cases_for_rag(
    predictions_path: str,
    original_data_path: str,
    output_path: str = 'outputs/fraud_cases_for_rag.csv',
    fraud_only: bool = True,
    top_n: Optional[int] = None,
    min_probability: float = 0.7
) -> pd.DataFrame:
    """
    Prepare comprehensive fraud cases untuk RAG explainer.
    
    Output columns untuk RAG integration:
    - Model predictions (fraud_type, probability, confidence)
    - SHAP explanations (top_features, explanation_json, z_scores)
    - Clinical context (diagnosis, procedure, INA-CBG)
    - Financial context (costs, ratios, deviations)
    - RAG query string (siap untuk vector DB)
    - Context summary (siap untuk LLM prompt)
    """
    
    logger.info("="*80)
    logger.info("PREPARING COMPREHENSIVE RAG DATA")
    logger.info("="*80)
    
    # Load predictions
    logger.info(f"📂 Loading predictions: {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    
    # Load original data
    logger.info(f"📂 Loading original data: {original_data_path}")
    original_df = pd.read_csv(original_data_path)
    
    # Merge
    logger.info("🔗 Merging predictions with original data...")
    merged = predictions.merge(
        original_df,
        on='claim_id',
        how='left',
        suffixes=('_pred', '_orig')
    )
    
    # Filter fraud cases
    if fraud_only:
        merged = merged[merged['predicted_fraud'] == 1]
        logger.info(f"   ✅ Fraud cases: {len(merged):,}")
    
    # Filter by probability
    merged = merged[merged['fraud_probability'] >= min_probability]
    logger.info(f"   ✅ High confidence (>={min_probability}): {len(merged):,}")
    
    # Sort and limit
    merged = merged.sort_values('fraud_probability', ascending=False)
    if top_n:
        merged = merged.head(top_n)
        logger.info(f"   ✅ Top {top_n} cases selected")
    
    # === PREPARE RAG-SPECIFIC COLUMNS ===
    
    # 1. RAG Query String (untuk query vector DB)
    logger.info("🔧 Building RAG query strings...")
    merged['rag_query_string'] = merged.apply(
        lambda row: prepare_rag_query(row), axis=1
    )
    
    # 2. Context Summary (untuk LLM prompt)
    logger.info("🔧 Building context summaries...")
    merged['context_summary'] = merged.apply(
        lambda row: prepare_context_summary(row), axis=1
    )
    
    # 3. Feature Importance (structured JSON)
    if 'explanation_json' in merged.columns:
        logger.info("🔧 Parsing SHAP explanations...")
        merged['feature_importance'] = merged['explanation_json'].apply(
            lambda x: parse_feature_importance(x) if pd.notna(x) else None
        )
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    
    logger.info(f"\n📊 RAG Data Summary:")
    logger.info(f"   Total cases: {len(merged):,}")
    logger.info(f"   Avg probability: {merged['fraud_probability'].mean():.3f}")
    logger.info(f"   Fraud types: {merged['predicted_fraud_type'].nunique()}")
    
    logger.info(f"\n💾 Saved: {output_path}")
    logger.info("="*80)
    logger.info("✅ RAG PREPARATION COMPLETED")
    logger.info("="*80)
    
    return merged


def prepare_rag_query(row) -> str:
    """Build query string untuk vector DB retrieval"""
    return f"""Fraud Type: {row.get('predicted_fraud_type', 'N/A')}
Diagnosis: {row.get('diagnosis_primary_desc', 'N/A')}
Procedure: {row.get('procedure_primary_desc', 'N/A')}
Service: {row.get('jenis_pelayanan', 'N/A')}
Faskes: {row.get('faskes_level', 'N/A')}
INA-CBG: {row.get('inacbg_code', 'N/A')}
Key Features: {row.get('top_features', 'N/A')}
Cost Ratio: {row.get('claim_ratio', 0):.2f}
"""


def prepare_context_summary(row) -> str:
    """Build context summary untuk LLM prompt"""
    return f"""Claim ID: {row.get('claim_id')}
Fraud Type: {row.get('predicted_fraud_type')} (prob: {row.get('fraud_probability', 0):.2%})
Patient: {row.get('age', 'N/A')}yo {row.get('sex', 'N/A')}
Faskes: {row.get('faskes_name', 'N/A')} ({row.get('faskes_level', 'N/A')})
Diagnosis: {row.get('diagnosis_primary_desc', 'N/A')}
Procedure: {row.get('procedure_primary_desc', 'N/A')}
Costs: Billed={row.get('billed_amount', 0):,.0f}, INA-CBG={row.get('tarif_inacbg', 0):,.0f}
Explanation: {row.get('explanation_summary', 'N/A')}
"""


def parse_feature_importance(explanation_json_str) -> Dict:
    """Parse SHAP explanation JSON untuk structured feature importance"""
    try:
        explanation = json.loads(explanation_json_str)
        return {
            'top_features': explanation.get('top_features', []),
            'summary': explanation.get('summary', ''),
            'explanation_type': explanation.get('explanation_type', 'unknown')
        }
    except:
        return None


# === INTEGRATION DENGAN RAG EXPLAINER ===

def integrate_with_rag_explainer(
    rag_data_path: str,
    rag_explainer_module,  # Import modul RAG explainer Anda
    regulations_db_path: str,
    output_path: str = 'outputs/fraud_explanations.csv'
) -> pd.DataFrame:
    """
    Integration dengan RAG explainer module yang sudah ada.
    
    Workflow:
    1. Load RAG-prepared data
    2. Query regulations dari vector DB
    3. Generate LLM explanations
    4. Save final explanations
    """
    
    logger.info("="*80)
    logger.info("RAG EXPLAINER INTEGRATION")
    logger.info("="*80)
    
    # Load RAG data
    logger.info(f"📂 Loading RAG data: {rag_data_path}")
    rag_df = pd.read_csv(rag_data_path)
    
    # Initialize RAG explainer (sesuaikan dengan interface modul Anda)
    logger.info("🔧 Initializing RAG explainer...")
    # explainer = rag_explainer_module.RAGExplainer(regulations_db_path)
    
    explanations = []
    for idx, row in rag_df.iterrows():
        logger.info(f"Processing {idx+1}/{len(rag_df)}: {row['claim_id']}")
        
        # Query regulations
        # regulations = explainer.retrieve_regulations(row['rag_query_string'])
        
        # Generate explanation
        # explanation = explainer.generate_explanation(
        #     context=row['context_summary'],
        #     regulations=regulations
        # )
        
        # explanations.append({
        #     'claim_id': row['claim_id'],
        #     'llm_explanation': explanation
        # })
    
    # Merge explanations
    # explanations_df = pd.DataFrame(explanations)
    # final_df = rag_df.merge(explanations_df, on='claim_id')
    # final_df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Saved: {output_path}")
    logger.info("="*80)
    
    # return final_df
