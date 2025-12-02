"""
Model artifact management - Save and load trained models
"""

import os
import json
import joblib
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np  
import warnings
warnings.filterwarnings('ignore')

try:
    import cloudpickle
    CLOUDPICKLE_AVAILABLE = True
except ImportError:
    CLOUDPICKLE_AVAILABLE = False
    warnings.warn("cloudpickle not available. Install with: pip install cloudpickle")

# SHAP (dengan error handling)
try:
    import shap
    import cloudpickle
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP/cloudpickle not available in ModelArtifactManager")

class ModelArtifactManager:
    """Manages model artifacts with improved error handling"""

    def __init__(self, artifacts_dir: str = './model_artifacts'):
        self.artifacts_dir = artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)

    def save_pipeline(
        self, 
        pipeline, 
        stage: str = 'binary',
        model_name: str = 'ensemble', 
        metadata: Dict = None,
        shap_explainer: Optional[object] = None
    ):
        """Save complete pipeline including SHAP explainer"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.artifacts_dir, f"{stage}_{model_name}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"SAVING MODEL ARTIFACTS - {stage.upper()} - {model_name.upper()}")
        print(f"{'='*80}")
        print(f"Save directory: {save_dir}")

        try:
            # 1. Save model
            if stage == 'binary':
                model = pipeline.binary_ensemble if model_name == 'ensemble' else pipeline.binary_models.get(model_name)
                optimal_threshold = pipeline.optimal_threshold_stage1
            else:
                model = pipeline.multiclass_ensemble if model_name == 'ensemble' else pipeline.multiclass_models.get(model_name)
                optimal_threshold = None

            if model is None:
                raise ValueError(f"Model '{model_name}' not found")

            model_path = os.path.join(save_dir, 'model.pkl')
            joblib.dump(model, model_path)
            print(f"  ✓ Model saved: {model_path}")

            # 2. Save scaler
            scaler_path = os.path.join(save_dir, 'scaler.pkl')
            joblib.dump(pipeline.scaler, scaler_path)
            print(f"  ✓ Scaler saved")

            # 3. Save label encoders
            encoders_path = os.path.join(save_dir, 'label_encoders.pkl')
            joblib.dump(pipeline.label_encoders, encoders_path)
            print(f"  ✓ Label encoders saved")

            # 4. Save SHAP explainer
            if shap_explainer is not None and SHAP_AVAILABLE:
                explainer_path = os.path.join(save_dir, 'shap_explainer.pkl')
                try:
                    with open(explainer_path, "wb") as f:
                        cloudpickle.dump(shap_explainer, f)
                    print(f"  ✓ SHAP explainer saved")
                except Exception as e:
                    print(f"  ⚠️  Could not save SHAP explainer: {e}")

            # 5. Save feature statistics
            if hasattr(pipeline, 'shap_manager') and pipeline.shap_manager:
                if pipeline.shap_manager.feature_stats:
                    stats_path = os.path.join(save_dir, 'feature_stats.json')
                    with open(stats_path, 'w') as f:
                        json.dump(pipeline.shap_manager.feature_stats, f, indent=2)
                    print(f"  ✓ Feature statistics saved")

            # 6. Save feature config
            features_config = {
                'numerical_features': pipeline.numerical_features,
                'categorical_features': pipeline.categorical_features,
                'boolean_features': pipeline.boolean_features,
                'leakage_features': pipeline.leakage_features
            }
            features_path = os.path.join(save_dir, 'features_config.json')
            with open(features_path, 'w') as f:
                json.dump(features_config, f, indent=2)
            print(f"  ✓ Feature config saved")

            # 7. Save metadata
            if metadata is None:
                metadata = {}

            metadata.update({
                'stage': stage,
                'model_name': model_name,
                'timestamp': timestamp,
                'save_date': datetime.now().isoformat(),
                'artifacts_dir': save_dir,
                'optimal_threshold': float(optimal_threshold) if optimal_threshold else None,
                'shap_available': shap_explainer is not None
            })

            metadata_path = os.path.join(save_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(self._clean_metadata(metadata), f, indent=2)
            print(f"  ✓ Metadata saved")

            if optimal_threshold:
                print(f"  ✓ Optimal threshold: {optimal_threshold:.4f}")

            print(f"\n✅ All artifacts saved to: {save_dir}")
            return save_dir

        except Exception as e:
            print(f"❌ Error saving artifacts: {e}")
            raise

    def _clean_metadata(self, metadata: Dict) -> Dict:
        """Convert numpy types for JSON serialization"""
        clean_dict = {}
        for key, value in metadata.items():
            if isinstance(value, np.integer):
                clean_dict[key] = int(value)
            elif isinstance(value, np.floating):
                clean_dict[key] = float(value)
            elif isinstance(value, np.ndarray):
                clean_dict[key] = value.tolist()
            elif isinstance(value, dict):
                clean_dict[key] = self._clean_metadata(value)
            elif isinstance(value, list):
                clean_dict[key] = [
                    self._clean_metadata(item) if isinstance(item, dict) else item 
                    for item in value
                ]
            else:
                clean_dict[key] = value
        return clean_dict

    def load_pipeline_artifacts(self, artifacts_dir: str) -> Dict:
        """Load all pipeline artifacts"""
        print(f"\n{'='*80}")
        print(f"LOADING MODEL ARTIFACTS")
        print(f"{'='*80}")
        print(f"Load directory: {artifacts_dir}")

        if not os.path.exists(artifacts_dir):
            raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

        artifacts = {}

        try:
            # Load model
            model_path = os.path.join(artifacts_dir, "model.pkl")

            if not os.path.exists(model_path):
                # Try common alternative names inside binary_models/ or multiclass_models/
                candidate_dirs = [
                    artifacts_dir,
                    os.path.join(artifacts_dir, "binary_models"),
                    os.path.join(artifacts_dir, "multiclass_models")
                ]

                candidate_files = [
                    "Ensemble.pkl",
                    "RandomForest.pkl",
                    "LightGBM.pkl",
                    "XGBoost.pkl"
                ]

                found = None

                for cdir in candidate_dirs:
                    for fname in candidate_files:
                        fpath = os.path.join(cdir, fname)
                        if os.path.exists(fpath):
                            found = fpath
                            break
                    if found:
                        break

                if not found:
                    raise FileNotFoundError(
                        f"❌ No model found.\n"
                        f"Checked:\n"
                        f" - {model_path}\n"
                        f" - {artifacts_dir}\\binary_models\\*\n"
                        f" - {artifacts_dir}\\multiclass_models\\*\n"
                    )

                model_path = found

            print(f"  ✓ Model loaded from: {model_path}")
            artifacts["model"] = joblib.load(model_path)


            # Load scaler
            scaler_path = os.path.join(artifacts_dir, 'scaler.pkl')
            artifacts['scaler'] = joblib.load(scaler_path)
            print(f"  ✓ Scaler loaded")

            # Load encoders
            encoders_path = os.path.join(artifacts_dir, 'label_encoders.pkl')
            artifacts['label_encoders'] = joblib.load(encoders_path)
            print(f"  ✓ Label encoders loaded")

            # Load SHAP explainer
            explainer_path = os.path.join(artifacts_dir, 'shap_explainer.pkl')
            if os.path.exists(explainer_path) and SHAP_AVAILABLE:
                try:
                    with open(explainer_path, "rb") as f:
                        artifacts['shap_explainer'] = cloudpickle.load(f)
                    print(f"  ✓ SHAP explainer loaded")
                except Exception as e:
                    print(f"  ⚠️  Could not load SHAP explainer: {e}")
                    artifacts['shap_explainer'] = None
            else:
                artifacts['shap_explainer'] = None

            # Load feature statistics
            stats_path = os.path.join(artifacts_dir, 'feature_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    artifacts['feature_stats'] = json.load(f)
                print(f"  ✓ Feature statistics loaded")
            else:
                artifacts['feature_stats'] = {}

            # Load feature config
            features_path = os.path.join(artifacts_dir, 'features_config.json')
            with open(features_path, 'r') as f:
                artifacts['features_config'] = json.load(f)
            print(f"  ✓ Feature config loaded")

            # Load metadata
            metadata_path = os.path.join(artifacts_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                artifacts['metadata'] = json.load(f)
            print(f"  ✓ Metadata loaded")

            if artifacts['metadata'].get('optimal_threshold'):
                print(f"  ✓ Optimal threshold: {artifacts['metadata']['optimal_threshold']:.4f}")

            if artifacts.get('shap_explainer'):
                print(f"  ✓ Explainability: SHAP available")
            else:
                print(f"  ⚠️  Explainability: Z-score fallback mode")

            print(f"\n✅ All artifacts loaded successfully")
            return artifacts

        except Exception as e:
            print(f"❌ Error loading artifacts: {e}")
            raise

