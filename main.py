"""
Main entry point for BPJS Fraud Detection System
"""

import sys
from pathlib import Path
from rag_integration import RAGIntegration

# Add parent directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import pandas as pd
import argparse
from datetime import datetime

# Change to absolute imports from current package
from models.detection_pipeline import BPJSFraudDetectionPipeline
from models.inference_engine import BPJSFraudInferenceEngine
from models.model_artifact_manager import ModelArtifactManager
from utils.logger import setup_logger
from utils.helper import load_config, create_output_filename
import os
from typing import Dict
from dotenv import load_dotenv

load_dotenv(override=True)

# Setup logger
logger = setup_logger('bpjs_fraud_detection', log_file='./logs/app.log')
if sys.platform == 'win32':
    # Force UTF-8 for Windows console
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def verify_model_artifacts(model_path: str) -> Dict[str, bool]:
    """
    Verify that all required artifacts were saved correctly
    
    Args:
        model_path: Path to model artifacts directory
    
    Returns:
        Dictionary with verification results
    """
    from pathlib import Path
    import pickle
    from sklearn.preprocessing import LabelEncoder  # Required for unpickling
    
    results = {
        'shap_explainer': False,
        'fraud_type_encoder': False,
        'label_encoders': False,
        'feature_config': False
    }
    
    model_dir = Path(model_path)
    
    # Check SHAP explainer
    shap_path = model_dir / 'shap_explainer.pkl'
    if shap_path.exists():
        results['shap_explainer'] = True
    
    # Check label encoders
    encoders_path = model_dir / 'label_encoders.pkl'
    if encoders_path.exists():
        results['label_encoders'] = True
        
        try:
            with open(encoders_path, 'rb') as f:
                encoders = pickle.load(f)
            
            if 'fraud_type_encoder' in encoders:
                results['fraud_type_encoder'] = True
                results['fraud_type_classes'] = list(encoders['fraud_type_encoder'].classes_)
        except Exception as e:
            logger.warning(f"Could not verify encoders: {e}")
    
    # Check feature config
    config_path = model_dir / 'features_config.json'
    if config_path.exists():
        results['feature_config'] = True
    
    return results

def train_models(config_path: str = None):
    """Train fraud detection models"""
    logger.info("\n" + "="*80)
    logger.info("TRAINING MODE")
    logger.info("="*80)
    
    # ========== 1. LOAD CONFIGURATION ==========
    config = load_config(config_path)
    
    # ========== 2. LOAD TRAINING DATA ==========
    train_data_path = input("Enter path to training data CSV: ")
    df_train = pd.read_csv(train_data_path)
    
    logger.info(f"\nTraining data loaded: {len(df_train)} samples")
    
    # ========== 3. INITIALIZE PIPELINE ==========
    pipeline = BPJSFraudDetectionPipeline(
        random_state=config['model']['random_state'],
        enable_shap=config['model']['enable_shap'],
        config=config
    )
    
    # ========== 4. TRAIN STAGE 1 (BINARY) ==========
    logger.info("\n" + "="*80)
    logger.info("STARTING STAGE 1: BINARY CLASSIFICATION")
    logger.info("="*80)
    
    train_config = config['model']['training']
    
    results_stage1 = pipeline.train_stage1_binary(
        df_train,
        test_size=train_config['test_size'],
        tune_threshold=train_config['tune_threshold'],
        threshold_metric=train_config['threshold_metric'],
        create_explainer=True
    )
    
    # ========== 5. TRAIN STAGE 2 (MULTICLASS) ==========
    logger.info("\n" + "="*80)
    logger.info("STARTING STAGE 2: MULTICLASS CLASSIFICATION")
    logger.info("="*80)
    
    results_stage2 = None
    
    if 'fraud_flag' in df_train.columns:
        fraud_cases = df_train[df_train['fraud_flag'] == 1]
        
        if len(fraud_cases) > 0 and 'fraud_type' in df_train.columns:
            logger.info(f"Found {len(fraud_cases)} fraud cases for Stage 2 training")
            
            X_stage2, y_stage2 = pipeline._prepare_features_multiclass(df_train)
            
            results_stage2 = pipeline.train_stage2_multiclass(
                X_stage2,
                y_stage2,
                test_size=train_config['test_size'],
                random_state=config['model']['random_state'],
                use_stratify=True,
            )
        else:
            logger.warning("Insufficient fraud cases or missing 'fraud_type' column")
            logger.warning("Stage 2 training skipped")
    else:
        logger.warning("'fraud_flag' column not found - Stage 2 skipped")
    
    # ========== 6. SAVE MODELS ==========
    logger.info("\n" + "="*80)
    logger.info("SAVING MODEL ARTIFACTS")
    logger.info("="*80)
    
    manager = ModelArtifactManager(artifacts_dir=config['paths']['artifacts_dir'])
    
    # ========== SAVE STAGE 1 (BINARY) ==========
    binary_metadata = {
        'accuracy': results_stage1['metrics']['Ensemble']['accuracy'],
        'f1_score': results_stage1['metrics']['Ensemble']['f1_score'],
        'roc_auc': results_stage1['metrics']['Ensemble']['roc_auc'],
        'optimal_threshold': results_stage1.get('optimal_threshold', 0.5),
        'threshold_metric': train_config['threshold_metric']
    }
    
    binary_path = manager.save_pipeline(
        pipeline,
        stage='binary',
        model_name='ensemble',
        shap_explainer=results_stage1.get('shap_explainer'),
        metadata=binary_metadata
    )
    
    logger.info(f"✓ Binary model saved: {binary_path}")
    
    # ========== SIMPLE FILE EXISTENCE CHECK (NO PICKLE LOADING) ==========
    from pathlib import Path
    
    # Check SHAP explainer
    shap_path = Path(binary_path) / 'shap_explainer.pkl'
    if shap_path.exists():
        logger.info(f"  ✓ SHAP explainer file exists")
    else:
        logger.warning(f"  ⚠️  SHAP explainer file not found")
    
    # Check label encoders
    encoders_path = Path(binary_path) / 'label_encoders.pkl'
    if encoders_path.exists():
        logger.info(f"  ✓ Label encoders file exists")
    else:
        logger.warning(f"  ⚠️  Label encoders file not found")
    
    # Check feature config
    features_path = Path(binary_path) / 'features_config.json'
    if features_path.exists():
        logger.info(f"  ✓ Feature config file exists")
    else:
        logger.warning(f"  ⚠️  Feature config file not found")
    # =====================================================================
    
    # ========== SAVE STAGE 2 (MULTICLASS) ==========
    multiclass_path = None
    
    if results_stage2 is not None:
        if 'metrics' in results_stage2 and 'Ensemble' in results_stage2['metrics']:
            multiclass_metadata = {
                'accuracy': results_stage2['metrics']['Ensemble']['accuracy'],
                'f1_score': results_stage2['metrics']['Ensemble']['f1_score'],
                'f1_macro': results_stage2['metrics']['Ensemble'].get('f1_macro', 0.0)
            }
            
            multiclass_path = manager.save_pipeline(
                pipeline,
                stage='multiclass',
                model_name='ensemble',
                metadata=multiclass_metadata
            )
            
            logger.info(f"✓ Multiclass model saved: {multiclass_path}")
        else:
            logger.warning("⚠️  Stage 2 results incomplete - metrics not available")
    else:
        logger.warning("⚠️  Stage 2 training was skipped")
    
    # ========== 7. TRAINING SUMMARY ==========
    logger.info("\n" + "="*80)
    logger.info("✅ TRAINING COMPLETED!")
    logger.info("="*80)
    
    logger.info(f"\n📊 Training Summary:")
    logger.info(f"  Total samples: {len(df_train):,}")
    
    if 'fraud_flag' in df_train.columns:
        fraud_count = df_train['fraud_flag'].sum()
        logger.info(f"  Fraud cases: {fraud_count} ({fraud_count/len(df_train)*100:.2f}%)")
    
    logger.info(f"\n📁 Model Artifacts:")
    logger.info(f"  Binary (Stage 1):  {binary_path}")
    if multiclass_path:
        logger.info(f"  Multiclass (Stage 2): {multiclass_path}")
    else:
        logger.info(f"  Multiclass (Stage 2): Not saved (Stage 2 skipped)")
    
    logger.info(f"\n📈 Binary Model Performance:")
    logger.info(f"  Accuracy:  {binary_metadata['accuracy']:.4f}")
    logger.info(f"  F1-Score:  {binary_metadata['f1_score']:.4f}")
    logger.info(f"  ROC-AUC:   {binary_metadata['roc_auc']:.4f}")
    logger.info(f"  Threshold: {binary_metadata['optimal_threshold']:.4f}")
    
    if results_stage2 and 'metrics' in results_stage2 and 'Ensemble' in results_stage2['metrics']:
        logger.info(f"\n📈 Multiclass Model Performance:")
        logger.info(f"  Accuracy:  {results_stage2['metrics']['Ensemble']['accuracy']:.4f}")
        logger.info(f"  F1-Score:  {results_stage2['metrics']['Ensemble']['f1_score']:.4f}")
        if 'f1_macro' in results_stage2['metrics']['Ensemble']:
            logger.info(f"  F1-Macro:  {results_stage2['metrics']['Ensemble']['f1_macro']:.4f}")
    
    logger.info(f"\n🎯 Next Steps:")
    logger.info(f"  1. Verify artifacts:")
    logger.info(f"     python verify_encoder.py")
    logger.info(f"  2. Run inference:")
    logger.info(f"     python main.py inference")
    logger.info(f"  3. Check fraud_type_encoder:")
    logger.info(f"     The encoder will be validated during inference")
    
    logger.info("\n" + "="*80)



def run_inference(config_path: str = None):
    """Run inference on new data with optional RAG explanations"""
    logger.info("\n" + "="*80)
    logger.info("INFERENCE MODE")
    logger.info("="*80)
    
    # Load configuration
    config = load_config(config_path)
    
    # ⭐ SAFE CONFIG ACCESS WITH DEFAULTS
    inference_config = config.get('model', {}).get('inference', {
        'primary_key': 'claim_id',
        'use_optimal_threshold': True,
        'return_probabilities': True,
        'generate_explanations': True,
        'explanation_top_k': 5
    })
    
    # Get model paths
    binary_model_path = input("Enter path to binary model directory: ")
    multiclass_model_path = input("Enter path to multiclass model directory (press Enter to skip): ")
    
    if not multiclass_model_path.strip():
        multiclass_model_path = None
    
    # Load models
    engine = BPJSFraudInferenceEngine()
    engine.load_models(
        binary_artifacts_dir=binary_model_path,
        multiclass_artifacts_dir=multiclass_model_path
    )
    
    # Load inference data
    inference_data_path = input("Enter path to inference data CSV: ")
    df_inference = pd.read_csv(inference_data_path)
    
    logger.info(f"\nInference data loaded: {len(df_inference)} samples")
    
    # Run ML inference
    result = engine.predict(
        df_inference,
        primary_key=inference_config.get('primary_key', 'claim_id'),
        use_optimal_threshold=inference_config.get('use_optimal_threshold', True),
        return_probabilities=inference_config.get('return_probabilities', True),
        generate_explanations=inference_config.get('generate_explanations', True),
        explanation_top_k=inference_config.get('explanation_top_k', 5),
        merge_with_input=True
    )
    
    # Extract predictions
    predictions_df = result['predictions']
    
    # Save ML predictions
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = create_output_filename('fraud_detection_results', 'csv')
    output_path = output_dir / output_filename
    
    predictions_df.to_csv(output_path, index=False)
    
    logger.info("\n" + "="*80)
    logger.info("✅ ML INFERENCE COMPLETED!")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {output_path}")
    logger.info(f"Total samples: {len(predictions_df)}")
    logger.info(f"Fraud detected: {(predictions_df['predicted_fraud'] == 1).sum()}")
    logger.info(f"Columns in output: {len(predictions_df.columns)}")
    
    # ========== RAG INTEGRATION ==========
    rag_integration = RAGIntegration(
        rag_system_path=config['paths'].get('rag_system_path')
    )
    
    if rag_integration.is_available():
        logger.info("\n" + "="*80)
        logger.info("RAG SYSTEM DETECTED")
        logger.info("="*80)
        logger.info(f"Path: {rag_integration.rag_system_path}")
        
        # enable_rag = input("\n🤖 Generate RAG explanations? (y/n): ").strip().lower()
        
        # Initialize RAG
        rag_integration.initialize_rag_system(rebuild_vectors=False)
        
        # Generate explanations
        final_df = rag_integration.generate_explanations(
            predictions_df=predictions_df,
            original_claims_df=df_inference,
            fraud_only=True,
            show_progress=True
        )
        
        # Save with RAG explanations
        rag_output_filename = create_output_filename('fraud_detection_with_rag', 'csv')
        rag_output_path = output_dir / rag_output_filename
        
        final_df.to_csv(rag_output_path, index=False)
        
        logger.info("\n" + "="*80)
        logger.info("✅ RAG INTEGRATION COMPLETED!")
        logger.info("="*80)
        logger.info(f"\nFull output saved to: {rag_output_path}")
        logger.info(f"Total columns: {len(final_df.columns)}")
        
        # Show sample with RAG explanation
        fraud_with_rag = final_df[
            (final_df['predicted_fraud'] == 1) & 
            (final_df['explanation_text'].notna())
        ]
        
        if len(fraud_with_rag) > 0:
            logger.info(f"\n📊 Sample fraud case with RAG explanation:")
            sample = fraud_with_rag.iloc[0]
            logger.info(f"\n  Claim ID: {sample['claim_id']}")
            logger.info(f"  Fraud Type: {sample['predicted_fraud_type']}")
            logger.info(f"  Confidence: {sample['fraud_probability']:.2%}")
            logger.info(f"\n  RAG Explanation (first 500 chars):")
            logger.info(f"  {sample['explanation_text'][:500]}...")
        
            
        
    else:
        logger.info("\n⚠️  RAG system not available")
        logger.info("   To enable RAG:")
        logger.info("   1. Install dependencies: pip install langchain langchain-openai supabase")
        logger.info("   2. Set RAG_SYSTEM_PATH in .env or config/settings.yaml")
    # =====================================
    
    # Sample fraud cases summary
    fraud_cases = predictions_df[predictions_df['predicted_fraud'] == 1]
    if len(fraud_with_rag) > 0:
        sample = fraud_with_rag.iloc[0]
        logger.info(f"\n  Claim ID: {sample['claim_id']}")
        logger.info(f"  Fraud Type: {sample['predicted_fraud_type']}")
        logger.info(f"  Confidence: {sample['fraud_probability']:.2%}")
        logger.info(f"\n  RAG Explanation (first 500 chars):")
        logger.info(f"  {sample['explanation_text'][:500]}...")
    else:
        logger.info("No fraud cases with explanation available to show")



def check_rag_system_availability() -> bool:
    """Check if RAG system is available"""
    try:
        # Try to import RAG system
        import sys
        
        # Check if RAG system path is configured
        rag_path = os.getenv('RAG_SYSTEM_PATH')
        if not rag_path:
            # Try default path
            rag_path = os.path.abspath('../RAG-System')
        
        if os.path.exists(rag_path):
            sys.path.insert(0, rag_path)
            
            # Try importing RAG modules
            from bpjs_fraud_rag_system import BPJSFraudRAGSystem
            from ml_pipeline_rag_bridge import MLPipelineRAGBridge
            
            return True
        else:
            return False
            
    except ImportError:
        return False


def run_rag_explanations(
    predictions_df: pd.DataFrame,
    original_claims_df: pd.DataFrame,
    output_dir: Path,
    config: dict
) -> str:
    """
    Generate RAG explanations for fraud cases
    
    Args:
        predictions_df: ML predictions with fraud detection results
        original_claims_df: Original claim data
        output_dir: Output directory for results
        config: Configuration dict
    
    Returns:
        Path to output file with RAG explanations
    """
    # Import RAG system
    from bpjs_fraud_rag_system import BPJSFraudRAGSystem
    from ml_pipeline_rag_bridge import MLPipelineRAGBridge
    
    logger.info("\n" + "="*80)
    logger.info("GENERATING RAG EXPLANATIONS")
    logger.info("="*80)
    
    # Initialize RAG system
    logger.info("\n[1/4] Initializing RAG system...")
    rag_system = BPJSFraudRAGSystem()
    
    # Setup RAG (load vector store)
    logger.info("[2/4] Loading vector store...")
    rag_system.setup(rebuild_vectors=False)
    
    # Prepare fraud cases
    logger.info("[3/4] Preparing fraud cases...")
    fraud_cases = MLPipelineRAGBridge.prepare_fraud_cases_for_rag(
        predictions_df=predictions_df,
        original_claims_df=original_claims_df,
        fraud_only=True
    )
    
    logger.info(f"   Found {len(fraud_cases)} fraud cases to explain")
    
    if len(fraud_cases) == 0:
        logger.info("   No fraud cases to explain, skipping RAG...")
        return None
    
    # Generate explanations
    logger.info("[4/4] Generating explanations with RAG...")
    logger.info(f"   Processing {len(fraud_cases)} cases...")
    logger.info(f"   This may take a few minutes...")
    
    rag_explanations_df = rag_system.explain_fraud_cases_batch(fraud_cases)
    
    logger.info(f"   ✓ Generated {len(rag_explanations_df)} explanations")
    
    # Merge back with predictions
    logger.info("\n[Merge] Combining ML predictions with RAG explanations...")
    final_df = MLPipelineRAGBridge.merge_rag_explanations_back(
        predictions_df=predictions_df,
        rag_explanations_df=rag_explanations_df
    )
    
    # Save final output
    rag_output_filename = create_output_filename('fraud_detection_with_rag', 'csv')
    rag_output_path = output_dir / rag_output_filename
    
    final_df.to_csv(rag_output_path, index=False)
    
    logger.info(f"\n✓ RAG integration complete!")
    logger.info(f"  Total columns: {len(final_df.columns)}")
    logger.info(f"  Fraud cases with RAG explanations: {len(rag_explanations_df)}")
    
    return str(rag_output_path)




def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='BPJS Fraud Detection System')
    parser.add_argument(
        'mode',
        choices=['train', 'inference'],
        help='Operation mode: train or inference'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_models(config_path=args.config)
    elif args.mode == 'inference':
        run_inference(config_path=args.config)


if __name__ == '__main__':
    main()
