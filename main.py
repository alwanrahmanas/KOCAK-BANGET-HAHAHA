"""
Main CLI Entry Point
====================
Usage:
    python main.py train      # Training mode
    python main.py inference  # Inference / prediction mode
    python main.py rag        # (optional) RAG prep mode
"""

import argparse
import yaml
from pathlib import Path

from workflows.training_workflow import train_fraud_detection_models
from workflows.inference_workflow import predict_fraud_batch
# from workflows.rag_workflow import prepare_fraud_cases_for_rag
from utils.logger import setup_logging, get_logger
import logging
# from logging import logger

from datetime import datetime

# def make_timestamped_path(prefix="predictions", ext="csv", dir="./outputs"):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     return f"{dir}/{prefix}_{timestamp}.{ext}"

def get_timestamped_filename(prefix="predictions", ext="csv", dir="outputs"):
    """Generate timestamped filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(dir).mkdir(parents=True, exist_ok=True)
    return f"{dir}/{prefix}_{timestamp}.{ext}"

# # Saat infer atau rag
# output_path = args.output or make_timestamped_path("predictions")

# # Untuk rag data
# rag_output_path = args.output or make_timestamped_path("rag_data")

# Version with logger and debugger integration (breakpoints), without changing logic.

import os
import pandas as pd
import json
import logging

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def cmd_explain(args):
    """Generate explanation for fraud cases from CSV (FIXED VERSION)"""

    logger.info("Starting cmd_explain execution")

    print("\n" + "="*80)
    print("COMMAND: GENERATE EXPLANATIONS")
    print("="*80)

    # Check input file
    if not args.input:
        logger.error("--input file missing")
        print("✗ Error: --input file is required")
        return

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        print(f"✗ Error: Input file not found: {args.input}")
        return

    # Load predictions
    logger.debug(f"Loading predictions CSV: {args.input}")
    print(f"\n[1/4] Loading predictions from: {args.input}")
    predictions_df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(predictions_df)} predictions from CSV")

    fraud_df = predictions_df[predictions_df['predicted_fraud'] == 1].copy()
    logger.info(f"Detected {len(fraud_df)} fraud cases")

    print(f"   Loaded {len(predictions_df)} predictions")
    print(f"   Fraud cases detected: {len(fraud_df)}")

    if len(fraud_df) == 0:
        logger.warning("No fraud cases found; stopping execution")
        print("   No fraud cases to explain. Exiting.")
        return

    # Optional RAG data merge
    if hasattr(args, 'rag_data') and args.rag_data:
        logger.debug(f"rag_data argument detected: {args.rag_data}")

        if os.path.exists(args.rag_data):
            print(f"\n[2/4] Loading comprehensive RAG data from: {args.rag_data}")
            rag_df = pd.read_csv(args.rag_data)
            logger.info(f"Loaded {len(rag_df)} rows of RAG enrichment data")

            fraud_rag_df = fraud_df.merge(rag_df, on='claim_id', suffixes=('_pred', '_rag'))
            logger.info(f"Merged dataset resulting rows: {len(fraud_rag_df)}")
            print(f"   Merged fraud and RAG data rows: {len(fraud_rag_df)}")
        else:
            logger.error(f"RAG file not found: {args.rag_data}")
            print(f"✗ Error: RAG data file not found: {args.rag_data}")
            return
    else:
        logger.debug("No rag_data argument provided; using fraud_df only")
        fraud_rag_df = fraud_df

    # Initialize RAG system
    logger.debug("Initializing RAG system")
    print("\n[3/4] Loading RAG system...")
    rag_system = BPJSFraudRAGSystem()
    rag_system.setup(rebuild_vectors=False)

    # Prepare fraud cases
    logger.debug("Preparing fraud cases for RAG explanation")
    print("\n[4/4] Preparing fraud cases for explanation...")
    fraud_cases = []

    # FIXED: Column mapping - map CSV columns to expected names
    colmap = {
        # Core identifiers
        "claim_id": "claim_id",
        
        # Fraud prediction data
        "predicted_fraud": "predicted_fraud",
        "fraud_probability": "fraud_probability",
        "predicted_fraud_type": "predicted_fraud_type",
        
        # SHAP explanation data
        "shap_explanation_summary": "shap_explanation_summary",
        "shap_top_features": "shap_top_features",
        
        # Clinical data
        "kode_icd10": "kode_icd10",
        "diagnosis_name": "clinical_pathway_name",  # Map to expected name
        "kode_prosedur": "kode_prosedur",
        "procedure_name": "procedure_name",
        
        # Financial data
        "inacbg_code": "inacbg_code",  # Add if exists in CSV
        "tarif_inacbg": "tarif_inacbg",
        "billed_amount": "billed_amount",
        "paid_amount": "paid_amount",
        "selisih_klaim": "selisih_klaim",
        "claim_ratio": "claim_ratio",
        
        # Ratio indicators
        "drug_ratio": "drug_ratio",
        "procedure_ratio": "procedure_ratio",
        
        # Service data
        "jenis_pelayanan": "jenis_pelayanan",
        "room_class": "room_class",
        "lama_dirawat": "lama_dirawat",
        
        # Provider data
        "faskes_name": "faskes_name",
        "faskes_level": "faskes_level",
        "provider_monthly_claims": "provider_monthly_claims",
        
        # Patient history
        "visit_count_30d": "visit_count_30d",
        "clinical_pathway_deviation_score": "clinical_pathway_deviation_score",
    }

    for idx, row in fraud_rag_df.iterrows():
        logger.debug(f"Processing claim_id: {row.get('claim_id')}")

        # Parse explanation_json if string
        explanation_json = row.get('explanation_json', {})
        if isinstance(explanation_json, str):
            try:
                explanation_json = json.loads(explanation_json)
            except Exception as e:
                logger.warning(f"Failed to parse explanation_json for claim_id {row.get('claim_id')}: {e}")
                explanation_json = {}

        # Build fraud_case dictionary with FLAT structure
        fraud_case = {}
        
        # Map all columns according to colmap
        for csv_col, target_col in colmap.items():
            if csv_col in row.index:
                fraud_case[target_col] = row[csv_col]
            else:
                # Log missing columns (for debugging)
                logger.debug(f"Column '{csv_col}' not found in row for claim_id {row.get('claim_id')}")
                fraud_case[target_col] = None

        # Add explanation_json separately
        fraud_case['explanation_json'] = explanation_json

        # Add derived fields if not present
        if 'fraud_label' not in fraud_case:
            fraud_case['fraud_label'] = "FRAUD" if fraud_case.get('predicted_fraud') == 1 else "LEGITIMATE"

        fraud_cases.append(fraud_case)

    logger.info(f"Total fraud cases prepared: {len(fraud_cases)}")

    # Print diagnostic info about first case
    if fraud_cases:
        print("\n[Diagnostic] First fraud case structure:")
        first_case = fraud_cases[0]
        print(f"   Keys available: {list(first_case.keys())}")
        print(f"   Sample values:")
        for key in ['claim_id', 'kode_icd10', 'clinical_pathway_name', 'tarif_inacbg', 'billed_amount']:
            print(f"      {key}: {first_case.get(key, 'MISSING')}")

    # Call RAG batch explanation
    logger.debug("Calling RAG batch explanation engine")
    rag_explanations_df = rag_system.explain_fraud_cases_batch(fraud_cases)
    logger.info("RAG explanation batch completed")

    # Set output path
    if not args.output:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f'./output/rag_explanations_{timestamp}.csv'
        logger.debug(f"Output path auto-generated: {args.output}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Merge explanations back
    logger.debug("Merging RAG explanations back to full prediction dataframe")
    final_df = MLPipelineRAGBridge.merge_rag_explanations_back(predictions_df, rag_explanations_df)
    final_df.to_csv(args.output, index=False)

    logger.info(f"Saved explanation output to: {args.output}")
    logger.info(f"Total rows in final output: {len(final_df)}")

    print("\n" + "="*80)
    print("✅ EXPLANATIONS COMPLETE!")
    print("="*80)
    print(f"📄 Output saved to: {args.output}")
    print(f"   Total rows: {len(final_df)}")
    print(f"   Fraud cases explained: {len(rag_explanations_df)}")
    # Print available columns
    print("\n[Debug] CSV Columns:")
    print(fraud_rag_df.columns.tolist())

    # Check first fraud case
    if fraud_cases:
        import json
        print("\n[Debug] First Fraud Case Structure:")
        print(json.dumps(fraud_cases[0], indent=2, default=str))

def load_config(config_path: str = 'config/settings.yaml'):
    """Load configuration from YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BPJS Fraud Detection System CLI"
    )

    parser.add_argument(
        '--config',
        default='config/settings.yaml',
        help='Path to configuration YAML (default: config/settings.yaml)'
    )

    subparsers = parser.add_subparsers(
        dest='command',
        required=True,
        help='Sub-commands: train, inference, rag'
    )

    # ---------- train ----------
    train_parser = subparsers.add_parser(
        'train',
        help='Train fraud detection models'
    )
    train_parser.add_argument(
        '--train-data',
        required=True,
        help='Path to training CSV (must contain fraud_flag & fraud_type)'
    )
    train_parser.add_argument(
        '--artifacts-dir',
        default=None,
        help='Directory to save model artifacts (override config.paths.artifacts_dir)'
    )

    # ---------- inference ----------
    infer_parser = subparsers.add_parser(
        'inference',
        help='Run inference / prediction'
    )
    infer_parser.add_argument(
        '--data',
        required=True,
        help='Path to CSV for inference (with or without labels)'
    )
    infer_parser.add_argument(
        '--binary-model',
        required=True,
        help='Path to binary model artifacts directory'
    )
    infer_parser.add_argument(
        '--multiclass-model',
        default=None,
        help='Path to multiclass model artifacts directory (optional)'
    )
    infer_parser.add_argument(
        '--output',
        default='outputs/predictions.csv',
        help='Path to save predictions CSV (default: outputs/predictions.csv)'
    )
    infer_parser.add_argument(
        '--evaluate',
        action='store_true',
        help='If set, compute evaluation metrics (requires labels in data)'
    )
    infer_parser.add_argument(
        '--no-explanations',
        action='store_true',
        help='Disable SHAP explanations (override config.inference.generate_explanations)'
    )

    # ---------- rag (optional) ----------
    rag_parser = subparsers.add_parser(
        'rag',
        help='Prepare fraud cases for RAG'
    )
    rag_parser.add_argument(
        '--predictions',
        required=True,
        help='Path to predictions CSV (output from inference)'
    )
    rag_parser.add_argument(
        '--original-data',
        required=True,
        help='Path to original data CSV used for those predictions'
    )
    rag_parser.add_argument(
        '--output',
        default='outputs/fraud_cases_for_rag.csv',
        help='Path to save RAG-ready CSV'
    )

    return parser


def main():
    parser = argparse.ArgumentParser(description="BPJS Fraud Detection CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Train
    train_parser = subparsers.add_parser('train', help='Train fraud detection models')
    train_parser.add_argument('--train-data', required=True, help='Path ke training CSV')
    train_parser.add_argument('--artifacts-dir', default=None, help='Model artifacts dir')

    # Inference
    infer_parser = subparsers.add_parser('inference', help='Run 2-stage inference')
    infer_parser.add_argument('--data', required=True, help='Path ke inference CSV')
    infer_parser.add_argument('--binary-model', required=True, help='Path ke model binary')
    infer_parser.add_argument('--multiclass-model', required=True, help='Path ke model multiclass')
    infer_parser.add_argument('--output', default=None, help='Output CSV (default: timestamped filename)')
    infer_parser.add_argument('--no-explanations', action='store_true', help='Disable SHAP explanations')
    infer_parser.add_argument('--evaluate', action='store_true', help='Evaluate if ground truth exists')


    args = parser.parse_args()
    config = load_config(args.config if hasattr(args, "config") else 'config/settings.yaml')
    setup_logging(config['paths']['logs_dir'])
    logger = get_logger(__name__)    # ⚠️ Pastikan logger didefinisikan sebelum dipakai
    # ---------- TRAIN ----------
    if args.command == 'train':
        artifacts_dir = args.artifacts_dir or config['paths']['artifacts_dir']

        logger.info("Starting training workflow...")
        res = train_fraud_detection_models(
            train_data_path=args.train_data,
            artifacts_dir=artifacts_dir,
            test_size=config['model']['binary']['test_size'],
            random_state=config['model']['random_state'],
            enable_shap=config['model']['enable_shap'],
            tune_threshold=config['model']['binary']['tune_threshold'],
            threshold_metric=config['model']['binary']['threshold_metric'],
            save_threshold_plot=True
        )

        logger.info(f"Binary model saved to: {res['binary_artifacts_dir']}")
        if res['multiclass_artifacts_dir']:
            logger.info(f"Multiclass model saved to: {res['multiclass_artifacts_dir']}")

    # ---------- INFERENCE ----------
    # Di blok inference command:
    elif args.command == 'inference':
        logger.info("Starting inference workflow...")
        
        infer_cfg = config.get('inference', {})
        generate_explanations = infer_cfg.get('generate_explanations', True)
        if hasattr(args, "no_explanations") and args.no_explanations:
            generate_explanations = False
        
        evaluate = False
        if hasattr(args, "evaluate") and args.evaluate:
            evaluate = True
        
        # Generate timestamped output if not provided
        if not args.output:
            args.output = get_timestamped_filename("predictions")
        
        res = predict_fraud_batch(
            data_path=args.data,
            binary_artifacts_dir=args.binary_model,
            multiclass_artifacts_dir=args.multiclass_model,
            output_path=args.output,
            evaluate=evaluate,
            generate_explanations=generate_explanations,
            explanation_top_k=infer_cfg.get('explanation_top_k', 5),
            use_optimal_threshold=infer_cfg.get('use_optimal_threshold', True),
            return_probabilities=infer_cfg.get('return_probabilities', True)
        )
        
        logger.info(f"✅ Predictions saved to: {res['output_path']}")
        logger.info(f"   Total claims: {len(res['predictions'])}")
        logger.info(f"   Fraud cases: {(res['predictions']['predicted_fraud']==1).sum()}")
        logger.info("\n🔗 Next step - Generate RAG explanations:")
        logger.info(f"   python RAGSystem/main.py explain --input {res['output_path']}")


    # ---------- RAG ----------
    # Validate rag command args:
    elif args.command == 'rag':
        if not args.predictions:
            parser.error("--predictions required for rag command")
        if not args.original_data:
            parser.error("--original-data required for rag command")

        rag_cfg = config.get('rag', {})
        rag_df = prepare_fraud_cases_for_rag(
            predictions_path=args.predictions,
            original_data_path=args.original_data,
            output_path=args.output,
            fraud_only=True,
            top_n=rag_cfg.get('top_n_cases'),
            min_probability=rag_cfg.get('min_fraud_probability', 0.7)
        )

        logger.info(f"RAG data saved to: {args.output} (rows={len(rag_df)})")


if __name__ == "__main__":
    main()
