"""
BPJS Data Generator Utility
Generate synthetic datasets for training, testing, and inference
"""

import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger

logger = get_logger(__name__)


def _generate_base_claims(
    num_rows: int,
    generator: any,  # BPJSDataGeneratorV3 instance
    seed: int = 123123,
    fraud_ratio: float = 0.15,
    graph_fraud_ratio: float = 0.15,
    year: int = 2024,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Internal function to generate base claims with fraud patterns
    
    Args:
        num_rows: Number of claims to generate
        generator: Data generator instance (BPJSDataGeneratorV3)
        seed: Random seed
        fraud_ratio: Ratio of traditional fraud
        graph_fraud_ratio: Ratio of graph-based fraud
        year: Year for claims
        verbose: Print verbose output
    
    Returns:
        DataFrame with generated claims
    """
    if verbose:
        logger.info(f"Generating {num_rows:,} claims with fraud_ratio={fraud_ratio:.2%}")
    
    # 1. Provider & participant pool
    n_providers = max(50, int(num_rows * 0.03))
    n_participants = max(200, int(num_rows * 0.1))
    
    providers_df = generator.make_providers(n_providers=n_providers)
    participants_df = generator.make_participants(n_participants=n_participants)
    
    # 2. Assemble base claims
    claims_df = generator.assemble_enhanced_claims(
        n_rows=num_rows,
        providers_df=providers_df,
        participants_df=participants_df,
        year=year
    )
    
    # 3. Inject traditional fraud
    claims_df = generator.inject_fraud(
        claims_df,
        fraud_ratio=fraud_ratio
    )
    
    # 4. Inject GRAPH fraud patterns (if available)
    if hasattr(generator, 'graph_injector'):
        claims_df = generator.graph_injector.inject_all_graph_patterns(
            claims_df,
            graph_fraud_ratio=graph_fraud_ratio
        )
    else:
        if verbose:
            logger.warning("Graph injector not available, skipping graph fraud patterns")
    
    # 5. Ensure compatibility (no NaN in graph columns)
    for col in ["referral_to", "graph_pattern_id"]:
        if col not in claims_df.columns:
            claims_df[col] = ""
        claims_df[col] = claims_df[col].fillna("")
    
    # 6. Add ML features
    claims_df = generator.featurize(claims_df)
    
    # 7. Validation/Summary
    if verbose:
        _print_dataset_summary(claims_df, fraud_ratio)
    
    return claims_df


def make_train_dataset(
    num_rows: int,
    generator: any,
    csv_path: str = "data/bpjs_train.csv",
    seed: int = 123123,
    fraud_ratio: float = 0.08,
    graph_fraud_ratio: float = 0.02,
    year: int = 2024,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate TRAINING dataset
    
    Includes:
        - fraud_flag (binary label for Stage 1)
        - fraud_type (multiclass label for Stage 2)
    
    Args:
        num_rows: Number of claims to generate
        generator: BPJSDataGeneratorV3 instance
        csv_path: Path to save CSV
        seed: Random seed
        fraud_ratio: Traditional fraud ratio
        graph_fraud_ratio: Graph-based fraud ratio
        year: Year for claims
        verbose: Print verbose output
    
    Returns:
        DataFrame with training data
    """
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING TRAINING DATASET")
    logger.info(f"{'='*80}")
    
    claims_df = _generate_base_claims(
        num_rows=num_rows,
        generator=generator,
        seed=seed,
        fraud_ratio=fraud_ratio,
        graph_fraud_ratio=graph_fraud_ratio,
        year=year,
        verbose=verbose
    )
    
    # Validate required columns
    required_cols = ["fraud_flag", "fraud_type"]
    missing = [c for c in required_cols if c not in claims_df.columns]
    if missing:
        raise ValueError(f"Missing target columns in generated data: {missing}")
    
    # Save to CSV
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    claims_df.to_csv(output_path, index=False)
    
    logger.info(f"\n✅ TRAINING dataset saved:")
    logger.info(f"   Path: {output_path}")
    logger.info(f"   Rows: {len(claims_df):,}")
    logger.info(f"   Columns: {len(claims_df.columns)}")
    logger.info(f"   Fraud rate: {claims_df['fraud_flag'].mean():.2%}")
    
    # Show fraud type distribution
    if 'fraud_type' in claims_df.columns:
        fraud_types = claims_df[claims_df['fraud_flag'] == 1]['fraud_type'].value_counts()
        logger.info(f"\n   Fraud type distribution:")
        for fraud_type, count in fraud_types.items():
            logger.info(f"     - {fraud_type}: {count} ({count/len(claims_df)*100:.2f}%)")
    
    return claims_df


def make_test_dataset(
    num_rows: int,
    generator: any,
    csv_path: str = "data/bpjs_test.csv",
    seed: int = 456456,
    fraud_ratio: float = 0.08,
    graph_fraud_ratio: float = 0.02,
    year: int = 2024,
    drop_fraud_type: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate TEST dataset
    
    Includes:
        - fraud_flag (for Stage 1 evaluation)
        - NO fraud_type (simulate real-world scenario)
    
    Args:
        num_rows: Number of claims to generate
        generator: BPJSDataGeneratorV3 instance
        csv_path: Path to save CSV
        seed: Random seed
        fraud_ratio: Traditional fraud ratio
        graph_fraud_ratio: Graph-based fraud ratio
        year: Year for claims
        drop_fraud_type: Whether to drop fraud_type column
        verbose: Print verbose output
    
    Returns:
        DataFrame with test data
    """
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING TEST DATASET")
    logger.info(f"{'='*80}")
    
    claims_df = _generate_base_claims(
        num_rows=num_rows,
        generator=generator,
        seed=seed,
        fraud_ratio=fraud_ratio,
        graph_fraud_ratio=graph_fraud_ratio,
        year=year,
        verbose=verbose
    )
    
    # Validate fraud_flag exists
    if "fraud_flag" not in claims_df.columns:
        raise ValueError("fraud_flag not found in generated data")
    
    # Drop fraud_type if requested (simulate real-world)
    if drop_fraud_type and "fraud_type" in claims_df.columns:
        claims_df = claims_df.drop(columns=["fraud_type"])
        logger.info("   ⚠️  fraud_type column dropped (test mode)")
    
    # Save to CSV
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    claims_df.to_csv(output_path, index=False)
    
    logger.info(f"\n✅ TEST dataset saved:")
    logger.info(f"   Path: {output_path}")
    logger.info(f"   Rows: {len(claims_df):,}")
    logger.info(f"   Columns: {len(claims_df.columns)}")
    logger.info(f"   Has fraud_flag: {'fraud_flag' in claims_df.columns}")
    logger.info(f"   Has fraud_type: {'fraud_type' in claims_df.columns}")
    if 'fraud_flag' in claims_df.columns:
        logger.info(f"   Fraud rate: {claims_df['fraud_flag'].mean():.2%}")
    
    return claims_df


def make_inference_dataset(
    num_rows: int,
    generator: any,
    csv_path: str = "data/bpjs_inference.csv",
    seed: int = 789789,
    fraud_ratio: float = 0.20,  # Internal only, labels will be dropped
    graph_fraud_ratio: float = 0.02,
    year: int = 2024,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate INFERENCE dataset
    
    Includes:
        - NO fraud_flag
        - NO fraud_type
        - Simulates production data entering the model
    
    Note: Internal fraud patterns are still injected for realism,
          but all label columns are dropped.
    
    Args:
        num_rows: Number of claims to generate
        generator: BPJSDataGeneratorV3 instance
        csv_path: Path to save CSV
        seed: Random seed
        fraud_ratio: Internal fraud ratio (labels dropped)
        graph_fraud_ratio: Graph-based fraud ratio
        year: Year for claims
        verbose: Print verbose output
    
    Returns:
        DataFrame with inference data (no labels)
    """
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING INFERENCE DATASET")
    logger.info(f"{'='*80}")
    
    # Generate with internal fraud patterns
    claims_df = _generate_base_claims(
        num_rows=num_rows,
        generator=generator,
        seed=seed,
        fraud_ratio=fraud_ratio,  # Internal only
        graph_fraud_ratio=graph_fraud_ratio,
        year=year,
        verbose=verbose
    )
    
    # Drop ALL label columns
    drop_cols = []
    for col in ["fraud_flag", "fraud_type", "severity", "evidence_type"]:
        if col in claims_df.columns:
            drop_cols.append(col)
    
    if drop_cols:
        claims_df = claims_df.drop(columns=drop_cols)
        logger.info(f"   ⚠️  Dropped label columns: {drop_cols}")
    
    # Save to CSV
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    claims_df.to_csv(output_path, index=False)
    
    logger.info(f"\n✅ INFERENCE dataset saved:")
    logger.info(f"   Path: {output_path}")
    logger.info(f"   Rows: {len(claims_df):,}")
    logger.info(f"   Columns: {len(claims_df.columns)}")
    logger.info(f"   Has fraud_flag: {'fraud_flag' in claims_df.columns}")
    logger.info(f"   Has fraud_type: {'fraud_type' in claims_df.columns}")
    logger.info(f"\n   Ready for pure inference (no ground truth)")
    
    return claims_df


def _print_dataset_summary(df: pd.DataFrame, expected_fraud_ratio: float = None):
    """Print dataset summary statistics"""
    logger.info(f"\n{'='*60}")
    logger.info("DATASET SUMMARY")
    logger.info(f"{'='*60}")
    
    logger.info(f"Total rows: {len(df):,}")
    logger.info(f"Total columns: {len(df.columns)}")
    
    # Check for labels
    has_fraud_flag = 'fraud_flag' in df.columns
    has_fraud_type = 'fraud_type' in df.columns
    
    logger.info(f"\nLabel columns:")
    logger.info(f"  - fraud_flag: {has_fraud_flag}")
    logger.info(f"  - fraud_type: {has_fraud_type}")
    
    if has_fraud_flag:
        fraud_rate = df['fraud_flag'].mean()
        fraud_count = df['fraud_flag'].sum()
        logger.info(f"\nFraud statistics:")
        logger.info(f"  - Fraud cases: {fraud_count:,} ({fraud_rate:.2%})")
        logger.info(f"  - Benign cases: {(len(df) - fraud_count):,} ({(1-fraud_rate):.2%})")
        
        if expected_fraud_ratio:
            diff = abs(fraud_rate - expected_fraud_ratio)
            if diff > 0.01:
                logger.warning(f"  ⚠️  Fraud rate differs from expected: {expected_fraud_ratio:.2%}")
    
    if has_fraud_type:
        logger.info(f"\nFraud type distribution:")
        fraud_types = df[df['fraud_flag'] == 1]['fraud_type'].value_counts()
        for fraud_type, count in fraud_types.items():
            logger.info(f"  - {fraud_type}: {count} ({count/len(df)*100:.2f}%)")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    
    if len(cols_with_missing) > 0:
        logger.warning(f"\nColumns with missing values:")
        for col, count in cols_with_missing.items():
            logger.warning(f"  - {col}: {count} ({count/len(df)*100:.2f}%)")
    else:
        logger.info(f"\n✅ No missing values detected")
    
    logger.info(f"{'='*60}\n")


def generate_all_datasets(
    generator: any,
    train_rows: int = 100000,
    test_rows: int = 5000,
    inference_rows: int = 2500,
    output_dir: str = "data",
    fraud_ratio: float = 0.08,
    graph_fraud_ratio: float = 0.02,
    year: int = 2024,
    verbose: bool = True
) -> dict:
    """
    Generate all three datasets (train, test, inference) at once
    
    Args:
        generator: BPJSDataGeneratorV3 instance
        train_rows: Number of training samples
        test_rows: Number of test samples
        inference_rows: Number of inference samples
        output_dir: Output directory for CSV files
        fraud_ratio: Traditional fraud ratio
        graph_fraud_ratio: Graph-based fraud ratio
        year: Year for claims
        verbose: Print verbose output
    
    Returns:
        Dictionary with paths to generated files
    """
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING ALL DATASETS")
    logger.info(f"{'='*80}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate train dataset
    train_path = output_dir / "bpjs_train.csv"
    df_train = make_train_dataset(
        num_rows=train_rows,
        generator=generator,
        csv_path=str(train_path),
        seed=123123,
        fraud_ratio=fraud_ratio,
        graph_fraud_ratio=graph_fraud_ratio,
        year=year,
        verbose=verbose
    )
    
    # Generate test dataset
    test_path = output_dir / "bpjs_test.csv"
    df_test = make_test_dataset(
        num_rows=test_rows,
        generator=generator,
        csv_path=str(test_path),
        seed=456456,
        fraud_ratio=fraud_ratio,
        graph_fraud_ratio=graph_fraud_ratio,
        year=year,
        drop_fraud_type=True,
        verbose=verbose
    )
    
    # Generate inference dataset
    inference_path = output_dir / "bpjs_inference.csv"
    df_inference = make_inference_dataset(
        num_rows=inference_rows,
        generator=generator,
        csv_path=str(inference_path),
        seed=789789,
        fraud_ratio=0.20,  # Internal only
        graph_fraud_ratio=graph_fraud_ratio,
        year=year,
        verbose=verbose
    )
    
    logger.info(f"\n{'='*80}")
    logger.info("✅ ALL DATASETS GENERATED SUCCESSFULLY!")
    logger.info(f"{'='*80}")
    
    result = {
        'train': str(train_path),
        'test': str(test_path),
        'inference': str(inference_path),
        'train_df': df_train,
        'test_df': df_test,
        'inference_df': df_inference
    }
    
    logger.info(f"\nGenerated files:")
    logger.info(f"  - Training:   {result['train']} ({len(df_train):,} rows)")
    logger.info(f"  - Test:       {result['test']} ({len(df_test):,} rows)")
    logger.info(f"  - Inference:  {result['inference']} ({len(df_inference):,} rows)")
    
    return result


# =====================================================================
# Example main usage (for testing)
# =====================================================================

if __name__ == "__main__":
    """
    Example usage - requires BPJSDataGeneratorV3
    
    To use:
    1. Import your BPJSDataGeneratorV3 class
    2. Instantiate generator
    3. Call make_*_dataset functions
    """
    
    print("\n" + "="*80)
    print("BPJS DATA GENERATOR - EXAMPLE USAGE")
    print("="*80)
    print("\n⚠️  To run this example:")
    print("   1. Import your BPJSDataGeneratorV3 class")
    print("   2. Uncomment the code below")
    print("   3. Run: python -m dataset.data_generator")
    print("="*80 + "\n")
    
    # Uncomment to test:
    from dataset.datasets import BPJSDataGeneratorV3
    
    generator = BPJSDataGeneratorV3(seed=42)
    
    # Generate small test datasets
    train_df = make_train_dataset(
        num_rows=20000,
        generator=generator,
        csv_path="test_train.csv",
        fraud_ratio=0.15,
        verbose=True
    )
    
    test_df = make_test_dataset(
        num_rows=500,
        generator=generator,
        csv_path="test_test.csv",
        fraud_ratio=0.08,
        drop_fraud_type=True,
        verbose=True
    )
    
    inf_df = make_inference_dataset(
        num_rows=10,
        generator=generator,
        csv_path="test_inference.csv",
        verbose=True
    )
    
    print("\n✅ Test datasets generated!")
