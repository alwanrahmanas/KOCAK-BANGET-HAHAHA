"""
Inference Workflow
==================
Prediction workflow dengan SHAP explanations untuk RAG integration.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import json

from models.inference_engine import BPJSFraudInferenceEngine
from utils.logger import get_logger

logger = get_logger(__name__)


def predict_fraud_batch(
    data_path: str,
    binary_artifacts_dir: str,
    multiclass_artifacts_dir: Optional[str] = None,
    output_path: str = 'outputs/predictions.csv',
    evaluate: bool = False,
    generate_explanations: bool = True,
    explanation_top_k: int = 5,
    use_optimal_threshold: bool = True,
    return_probabilities: bool = True
) -> Dict:
    """
    Batch inference dengan SHAP explanations (RAG-ready output).
    
    Returns:
        Dict dengan:
        - predictions: DataFrame dengan kolom RAG-ready
        - evaluation: Metrics (jika evaluate=True)
        - output_path: Path ke file hasil
    """
    
    logger.info("="*80)
    logger.info("BATCH FRAUD PREDICTION (RAG-READY)")
    logger.info("="*80)
    
    # Load data
    logger.info(f"📂 Loading data: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"   ✅ Rows: {len(df):,}")
    
    has_labels = 'fraud_flag' in df.columns
    logger.info(f"   ℹ️  Has labels: {has_labels}")
    
    if evaluate and not has_labels:
        logger.warning("   ⚠️  No labels found, setting evaluate=False")
        evaluate = False
    
    # Initialize engine
    logger.info("🔧 Loading inference engine...")
    engine = BPJSFraudInferenceEngine()
    engine.load_models(
        binary_artifacts_dir=binary_artifacts_dir,
        multiclass_artifacts_dir=multiclass_artifacts_dir
    )
    
    # Predict
    logger.info(f"🔮 Running predictions...")
    logger.info(f"   - Generate explanations: {generate_explanations}")
    logger.info(f"   - Evaluate: {evaluate}")
    
    results = engine.predict(
        df=df,
        return_probabilities=return_probabilities,
        use_optimal_threshold=use_optimal_threshold,
        evaluate=evaluate,
        generate_explanations=generate_explanations,
        explanation_top_k=explanation_top_k
    )
    
    predictions = results['predictions']
    evaluation = results.get('evaluation')
    
    # Summary
    logger.info(f"\n📊 Prediction Summary:")
    logger.info(f"   Total: {len(predictions):,}")
    logger.info(f"   Fraud: {(predictions['predicted_fraud']==1).sum():,} ({predictions['predicted_fraud'].mean():.2%})")
    
    # Evaluation metrics
    if evaluation:
        logger.info(f"\n📊 Evaluation Metrics:")
        stage1 = evaluation['stage1_binary']
        logger.info(f"   Accuracy:  {stage1['accuracy']:.4f}")
        logger.info(f"   Precision: {stage1['precision']:.4f}")
        logger.info(f"   Recall:    {stage1['recall']:.4f}")
        logger.info(f"   F1-Score:  {stage1['f1_score']:.4f}")
        logger.info(f"   ROC-AUC:   {stage1['roc_auc']:.4f}")
    
    # Save predictions
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    logger.info(f"\n💾 Saved: {output_path}")
    
    # Save evaluation
    if evaluation:
        eval_path = output_path.replace('.csv', '_evaluation.json')
        with open(eval_path, 'w') as f:
            json.dump(evaluation, f, indent=2, default=str)
        logger.info(f"💾 Saved: {eval_path}")
    
    logger.info("="*80)
    logger.info("✅ INFERENCE COMPLETED")
    logger.info("="*80)
    
    return {
        'predictions': predictions,
        'evaluation': evaluation,
        'output_path': output_path
    }
