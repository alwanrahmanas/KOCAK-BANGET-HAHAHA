"""
Utility functions to generate:
1) Training dataset  : ada fraud_flag dan fraud_type
2) Test dataset      : ada fraud_flag tapi TIDAK ada fraud_type
3) Inference dataset : TIDAK ada fraud_flag dan TIDAK ada fraud_type
"""

import pandas as pd
from models.dataset.datasets import BPJSDataGeneratorV3
from utils.data_validation import validate, print_summary
# Asumsi: BPJSDataGeneratorV3, validate, print_summary
# sudah didefinisikan/import di atas.


def _generate_base_claims(
    num_rows: int,
    seed: int = 123123,
    fraud_ratio: float = 0.08,
    graph_fraud_ratio: float = 0.02,
    year: int = 2024,
    verbose: bool = True
) -> pd.DataFrame:

    generator = BPJSDataGeneratorV3(seed=seed)

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

    # ⭐ 4. Inject GRAPH fraud patterns
    claims_df = generator.graph_injector.inject_all_graph_patterns(
        claims_df,
        graph_fraud_ratio=graph_fraud_ratio
    )

    # ⭐ 5. Ensure compatibility (tidak boleh ada NaN)
    for col in ["referral_to", "graph_pattern_id"]:
        if col not in claims_df.columns:
            claims_df[col] = ""
        claims_df[col] = claims_df[col].fillna("")

    # 6. Add ML features
    claims_df = generator.featurize(claims_df)

    # 7. Summary/validation
    if verbose:
        try:
            validate(claims_df, fraud_ratio=fraud_ratio)
        except:
            pass
        print_summary(claims_df)

    return claims_df



def make_train_dataset(
    num_rows: int,
    csv_path: str = "bpjs_train.csv",
    seed: int = 123123,
    fraud_ratio: float = 0.08,
    year: int = 2024,
    verbose: bool = True
) -> pd.DataFrame:
    """
    TRAIN DATASET
    - Ada: fraud_flag (Y binary), fraud_type (Y multiclass)
    - Tujuan: dipakai untuk training Stage 1 & Stage 2
    - Disimpan ke CSV
    """
    claims_df = _generate_base_claims(
        num_rows=num_rows,
        seed=seed,
        fraud_ratio=fraud_ratio,
        year=year,
        verbose=verbose
    )

    # Pastikan kolom target ada
    required_cols = ["fraud_flag", "fraud_type"]
    missing = [c for c in required_cols if c not in claims_df.columns]
    if missing:
        raise ValueError(f"Missing target columns in generated data: {missing}")

    # Simpan ke CSV
    claims_df.to_csv(csv_path, index=False)
    if verbose:
        print(f"\n[TRAIN] Saved train dataset with {len(claims_df):,} rows to: {csv_path}")

    return claims_df


def make_test_dataset(
    num_rows: int,
    csv_path: str = "bpjs_test.csv",
    seed: int = 456456,
    fraud_ratio: float = 0.08,
    year: int = 2024,
    drop_fraud_type: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    TEST DATASET
    - Ada: fraud_flag (untuk evaluasi Stage 1)
    - Tidak ada: fraud_type (simulasi kasus nyata di mana tipe fraud belum diberi label rinci)
    - Disimpan ke CSV
    """
    claims_df = _generate_base_claims(
        num_rows=num_rows,
        seed=seed,
        fraud_ratio=fraud_ratio,
        year=year,
        verbose=verbose
    )

    # Pastikan fraud_flag ada
    if "fraud_flag" not in claims_df.columns:
        raise ValueError("fraud_flag not found in generated data")

    # Hapus fraud_type kalau diminta
    if drop_fraud_type and "fraud_type" in claims_df.columns:
        claims_df = claims_df.drop(columns=["fraud_type"])

    claims_df.to_csv(csv_path, index=False)
    if verbose:
        print(f"\n[TEST] Saved test dataset with {len(claims_df):,} rows to: {csv_path}")
        print("       Columns:", list(claims_df.columns))

    return claims_df


def make_inference_dataset(
    num_rows: int,
    csv_path: str = "bpjs_inference.csv",
    seed: int = 789789,
    year: int = 2024,
    verbose: bool = True
) -> pd.DataFrame:
    """
    INFERENCE DATASET
    - Tidak ada: fraud_flag, fraud_type
    - Simulasi data 'production' yang masuk ke model
    - IMPORTANT: Di sini TIDAK inject fraud_ratio berbasis label ground truth,
      tapi masih boleh ada 'underlying fraud' secara simulasi.
      Untuk kesederhanaan, fungsi ini gunakan inject_fraud hanya untuk
      membuat pola, lalu kolom label dibuang.
    """
    # Bisa pilih apakah ingin tetap memakai generator.inject_fraud
    # supaya pola fraud realistis, tapi label dihapus:
    claims_df = _generate_base_claims(
        num_rows=num_rows,
        seed=seed,
        fraud_ratio=0.20,   # internal only, lalu dibuang
        year=year,
        verbose=verbose
    )

    # Drop kolom label
    drop_cols = []
    for c in ["fraud_flag", "fraud_type", "severity", "evidence_type"]:
        if c in claims_df.columns:
            drop_cols.append(c)
    if drop_cols:
        claims_df = claims_df.drop(columns=drop_cols)

    claims_df.to_csv(csv_path, index=False)
    if verbose:
        print(f"\n[INFERENCE] Saved inference dataset with {len(claims_df):,} rows to: {csv_path}")
        print("           Columns:", list(claims_df.columns))

    return claims_df



# =====================================================================
# Example main usage (small unit-test style)
# =====================================================================

if __name__ == "__main__":
    # Contoh: small sanity test
    print("Generating small train / test / inference datasets...\n")

    train_df = make_train_dataset(
        num_rows=10000,
        csv_path="bpjs_train_small.csv",
        fraud_ratio=0.08,
        verbose=True
    )

    test_df = make_test_dataset(
        num_rows=5000,
        csv_path="bpjs_test_small.csv",
        fraud_ratio=0.08,
        drop_fraud_type=False,
        verbose=True
    )

    inf_df = make_inference_dataset(
        num_rows=2500,
        csv_path="bpjs_inference_small.csv",
        verbose=True
    )

    print("\nDone.")