# workflows/training_workflow.py

import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from models.detection_pipeline import BPJSFraudDetectionPipeline
from models.model_artifact_manager import ModelArtifactManager
from models.threshold_optimizer import ThresholdOptimizer
from utils.logger import get_logger

logger = get_logger(__name__)


def train_fraud_detection_models(
    train_data_path: str,
    artifacts_dir: str = './model_artifacts',
    test_size: float = 0.2,
    random_state: int = 42,
    enable_shap: bool = True,
    tune_threshold: bool = True,
    threshold_metric: str = 'f1',
    save_threshold_plot: bool = True
) -> Dict:
    """
    Training workflow sederhana: train Stage 1 & Stage 2 + save artifacts.
    """

    logger.info("=" * 80)
    logger.info("TRAINING BPJS FRAUD DETECTION MODELS")
    logger.info("=" * 80)

    # Load training data
    logger.info(f"📂 Loading training data from: {train_data_path}")
    df = pd.read_csv(train_data_path)

    # Validasi kolom target
    if 'fraud_flag' not in df.columns:
        raise ValueError("Training requires 'fraud_flag' column!")
    if 'fraud_type' not in df.columns:
        raise ValueError("Training requires 'fraud_type' column for Stage 2!")

    logger.info(f"   ✅ Rows: {len(df):,}")
    logger.info(f"   ✅ Fraud rate: {df['fraud_flag'].mean():.2%}")

    # Init pipeline
    pipeline = BPJSFraudDetectionPipeline(
        random_state=random_state,
        enable_shap=enable_shap
    )

    # Stage 1
    logger.info("🎯 Training Stage 1 (binary)...")
    results_stage1 = pipeline.train_stage1_binary(
        df=df,
        test_size=test_size,
        tune_threshold=tune_threshold,
        threshold_metric=threshold_metric,
        create_explainer=enable_shap
    )

    # Optional: plot threshold
    threshold_plot_path = None
    if save_threshold_plot and 'Ensemble' in results_stage1.get('threshold_analysis', {}):
        threshold_data = results_stage1['threshold_analysis']['Ensemble']
        threshold_plot_path = Path(artifacts_dir) / 'threshold_analysis.png'
        threshold_plot_path.parent.mkdir(parents=True, exist_ok=True)
        ThresholdOptimizer.plot_threshold_analysis(
            threshold_data['metrics_df'],
            threshold_data['optimal_threshold'],
            save_path=str(threshold_plot_path)
        )

    # Stage 2
    logger.info("🎯 Training Stage 2 (multiclass)...")
    results_stage2 = pipeline.train_stage2_multiclass(
        df=df,
        test_size=test_size,
        create_explainer=False
    )

    # Save models
    logger.info(f"💾 Saving models to: {artifacts_dir}")
    manager = ModelArtifactManager(artifacts_dir=artifacts_dir)

    binary_dir = manager.save_pipeline(
        pipeline=pipeline,
        stage='binary',
        model_name='ensemble',
        metadata={
            'dataset_size': len(df),
            'fraud_ratio': df['fraud_flag'].mean(),
            'accuracy': results_stage1['metrics']['Ensemble']['accuracy'],
            'f1_score': results_stage1['metrics']['Ensemble']['f1_score'],
            'roc_auc': results_stage1['metrics']['Ensemble']['roc_auc'],
        },
        shap_explainer=results_stage1.get('shap_explainer')
    )

    multiclass_dir = None
    if results_stage2:
        multiclass_dir = manager.save_pipeline(
            pipeline=pipeline,
            stage='multiclass',
            model_name='ensemble',
            metadata={
                'accuracy': results_stage2['metrics']['Ensemble']['accuracy'],
                'f1_score': results_stage2['metrics']['Ensemble']['f1_score'],
            }
        )

    logger.info("✅ TRAINING COMPLETED")

    return {
        'pipeline': pipeline,
        'results_stage1': results_stage1,
        'results_stage2': results_stage2,
        'binary_artifacts_dir': binary_dir,
        'multiclass_artifacts_dir': multiclass_dir,
        'threshold_plot_path': str(threshold_plot_path) if threshold_plot_path else None,
    }
