"""
dataset.datasets
================
High-level helper untuk generate semua split sekaligus.
"""

from pathlib import Path
from typing import Dict

from dataset.data_generator import (
    make_train_dataset,
    make_test_dataset,
    make_inference_dataset,
)


def generate_all_datasets(
    base_dir: str = "./data",
    n_train: int = 50_000,
    n_test: int = 10_000,
    n_infer: int = 5_000,
) -> Dict[str, str]:
    """
    Generate train / test / inference datasets sekaligus.
    """

    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    train_path = str(base / "bpjs_train.csv")
    test_path = str(base / "bpjs_test.csv")
    infer_path = str(base / "bpjs_inference.csv")

    make_train_dataset(num_rows=n_train, csv_path=train_path)
    make_test_dataset(num_rows=n_test, csv_path=test_path)
    make_inference_dataset(num_rows=n_infer, csv_path=infer_path)

    return {
        "train": train_path,
        "test": test_path,
        "inference": infer_path,
    }
