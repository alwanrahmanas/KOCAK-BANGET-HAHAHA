"""
BPJS Fraud Detection System
Production-ready fraud detection with SHAP explanations and RAG integration
"""

__version__ = "1.0.0"
__author__ = "BPJS Fraud Detection Team"

from .models.detection_pipeline import BPJSFraudDetectionPipeline
from .models.inference_engine import BPJSFraudInferenceEngine
from .models.model_artifact_manager import ModelArtifactManager
from .models.shap_manager import SHAPExplainerManager
from .models.threshold_optimizer import ThresholdOptimizer

__all__ = [
    'BPJSFraudDetectionPipeline',
    'BPJSFraudInferenceEngine',
    'ModelArtifactManager',
    'SHAPExplainerManager',
    'ThresholdOptimizer',
]
