"""
Logging Utilities
=================
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(logs_dir: str = './logs', level: int = logging.INFO):
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(logs_dir) / f"fraud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
