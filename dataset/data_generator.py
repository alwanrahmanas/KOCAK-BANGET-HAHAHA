"""
dataset.data_generator
======================

Modular wrapper di atas BPJSDataGeneratorV32 untuk membuat:
- Train dataset  : dengan fraud_flag & fraud_type
- Test dataset   : dengan fraud_flag saja
- Inference data : tanpa label (untuk production inference)
"""

import pandas as pd
from typing import Optional

from dataset.bpjs_data_generator_v32 import BPJSDataGeneratorV32
# from dataset.bpjs_data_generator_v32 import validate, print_summary  # opsional


def _generate_base_claims(
    num_rows: int,
    seed: int = 123123,
    fraud_ratio: float = 0.08,
    phantom_ratio: float = 0.02,
    graph_fraud_ratio: float = 0.02,
    year: int = 2024,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Internal helper untuk membuat data dasar klaim BPJS dengan semua injeksi aktif.
    Dipakai oleh: make_train_dataset, make_test_dataset, make_inference_dataset.
    """

    generator = BPJSDataGeneratorV32(seed=seed)

    # 1. Provider & participant pool (kalau nanti mau diaktifkan)
    n_providers = max(20, int(num_rows * 0.005))
    n_participants = max(200, int(num_rows * 0.1))

    # providers_df = generator.make_providers(n_providers=n_providers)
    # participants_df = generator.make_participants(n_participants=n_participants)

    # 2. Generate complete dataset
    claims_df = generator.generate_complete_dataset(
        n_rows=num_rows,
        fraud_ratio=fraud_ratio,
        phantom_ratio=phantom_ratio,
        graph_fraud_ratio=graph_fraud_ratio,
        year=year
    )

    # 3. Ensure compatibility (kolom opsional)
    for col in ["referral_to", "graph_pattern_id"]:
        if col not in claims_df.columns:
            claims_df[col] = ""
        claims_df[col] = claims_df[col].fillna("")

    

    # 5. (opsional) Summary / validation
    # if verbose:
    #     try:
    #         validate(claims_df, fraud_ratio=fraud_ratio)
    #     except Exception:
    #         pass
    #     print_summary(claims_df)

    if claims_df is None or not isinstance(claims_df, pd.DataFrame):
        raise RuntimeError(
            "🛑 Generator failed to return valid DataFrame. "
            "Check BPJSDataGeneratorV32 logs for root cause."
        )

    return claims_df


def make_train_dataset(
    num_rows: int,
    csv_path: str = "bpjs_train.csv",
    seed: int = 123123,
    fraud_ratio: float = 0.08,
    phantom_ratio: float = 0.02,
    graph_fraud_ratio: float = 0.02,
    year: int = 2024,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate TRAIN DATASET
    ----------------------
    - Ada: fraud_flag (binary), fraud_type (multiclass)
    """

    claims_df = _generate_base_claims(
        num_rows=num_rows,
        seed=seed,
        fraud_ratio=0.3,
        phantom_ratio=0.075,
        graph_fraud_ratio=0.1,
        year=year,
        verbose=verbose
    )

    required_cols = ["fraud_flag", "fraud_type"]
    missing = [c for c in required_cols if c not in claims_df.columns]
    if missing:
        raise ValueError(f"Missing target columns in generated data: {missing}")

    claims_df.to_csv(csv_path, index=False)
    if verbose:
        print(f"[TRAIN] ✅ Saved: {csv_path} ({len(claims_df):,} rows)")
    return claims_df


def make_test_dataset(
    num_rows: int,
    csv_path: str = "bpjs_test.csv",
    seed: int = 456456,
    fraud_ratio: float = 0.08,
    phantom_ratio: float = 0.02,
    graph_fraud_ratio: float = 0.02,
    year: int = 2024,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate TEST DATASET
    ---------------------
    - Ada: fraud_flag
    - Tidak ada: fraud_type
    """

    claims_df = _generate_base_claims(
        num_rows=num_rows,
        seed=seed,
        fraud_ratio=fraud_ratio,
        phantom_ratio=phantom_ratio,
        graph_fraud_ratio=graph_fraud_ratio,
        year=year,
        verbose=verbose
    )

    if "fraud_flag" not in claims_df.columns:
        raise ValueError("fraud_flag not found in generated dataset")

    if "fraud_type" in claims_df.columns:
        claims_df = claims_df.drop(columns=["fraud_type"])

    claims_df.to_csv(csv_path, index=False)
    if verbose:
        print(f"[TEST] ✅ Saved: {csv_path} ({len(claims_df):,} rows)")
    return claims_df


def make_inference_dataset(
    num_rows: int,
    csv_path: str = "bpjs_inference.csv",
    seed: int = 789789,
    phantom_ratio: float = 0.2,
    graph_fraud_ratio: float = 0.1,
    year: int = 2024,
    verbose: bool = True,
    sample_n: Optional[int] = 100
) -> pd.DataFrame:
    """
    Generate INFERENCE DATASET
    --------------------------
    - Tidak ada: fraud_flag, fraud_type, severity, evidence_type
    - Tetap inject fraud pattern (fraud_ratio=0.20) agar distribusi realistis
    """

    claims_df = _generate_base_claims(
        num_rows=num_rows,
        seed=seed,
        fraud_ratio=0.5,  # tetap inject fraud pattern
        phantom_ratio=0.2,
        graph_fraud_ratio=0.1,
        year=year,
        verbose=verbose
    )

    drop_cols = ["fraud_flag", "fraud_type", "severity", "evidence_type"]
    claims_df = claims_df.drop(columns=[c for c in drop_cols if c in claims_df.columns])

    if sample_n is not None and len(claims_df) > sample_n:
        claims_df = claims_df.sample(sample_n, random_state=seed)

    claims_df.to_csv(csv_path, index=False)
    if verbose:
        print(f"[INFERENCE] ✅ Saved: {csv_path} ({len(claims_df):,} rows)")
    return claims_df


if __name__ == "__main__":
    # Contoh: small sanity test
    print("Generating small train / test / inference datasets...\n")

    train_df = make_train_dataset(
        num_rows=10000,
        csv_path="bpjs_train_small.csv",
    )

    test_df = make_test_dataset(
        num_rows=1000,
        csv_path="bpjs_test_small.csv",
    )

    inf_df = make_inference_dataset(
        num_rows=20,
        csv_path="bpjs_inference_small.csv",
        verbose=True
    )

    print("\nDone.")