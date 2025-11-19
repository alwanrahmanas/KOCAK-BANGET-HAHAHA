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
import warnings
warnings.filterwarnings('ignore')

try:
    import cloudpickle
    CLOUDPICKLE_AVAILABLE = True
except ImportError:
    CLOUDPICKLE_AVAILABLE = False
    warnings.warn("cloudpickle not available. Install with: pip install cloudpickle")


class ModelArtifactManager:
    """
    Manage saving and loading of model artifacts
    """
    
    def __init__(self, artifacts_dir: str = './artifacts'):
        """
        Initialize artifact manager
        
        Args:
            artifacts_dir: Base directory for saving artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    def save_pipeline(
        self,
        pipeline,
        stage: str = 'binary',
        model_name: str = 'ensemble',
        metadata: Dict = None,
        shap_explainer = None
    ) -> str:
        """Save complete pipeline"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = Path(self.artifacts_dir) / f"{stage}_{model_name}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print(f"SAVING MODEL ARTIFACTS - {stage.upper()} - {model_name.upper()}")
        print("="*80)
        print(f"Save directory: {save_dir}")
        
        # ========== 1. SAVE MODELS ==========
        if stage == 'binary':
            models_dir = save_dir / 'binary_models'
            models_dir.mkdir(exist_ok=True)
            
            # Save individual models
            if hasattr(pipeline, 'binary_models') and pipeline.binary_models:
                for name, model in pipeline.binary_models.items():
                    model_path = models_dir / f"{name}.pkl"
                    joblib.dump(model, model_path)
                print(f"✓ Saved {len(pipeline.binary_models)} binary models")
            
            # ⭐ FIX: Safe check for ensemble
            if hasattr(pipeline, 'binary_ensemble') and pipeline.binary_ensemble is not None:
                ensemble_path = models_dir / 'Ensemble.pkl'
                joblib.dump(pipeline.binary_ensemble, ensemble_path)
                print(f"✓ Saved binary ensemble model")
            else:
                print(f"⚠️  Binary ensemble not found, skipping")
                
        elif stage == 'multiclass':
            models_dir = save_dir / 'multiclass_models'
            models_dir.mkdir(exist_ok=True)
            
            # Save individual models
            if hasattr(pipeline, 'multiclass_models') and pipeline.multiclass_models:
                for name, model in pipeline.multiclass_models.items():
                    model_path = models_dir / f"{name}.pkl"
                    joblib.dump(model, model_path)
                print(f"✓ Saved {len(pipeline.multiclass_models)} multiclass models")
            
            # ⭐ FIX: Safe check for ensemble
            if hasattr(pipeline, 'multiclass_ensemble') and pipeline.multiclass_ensemble is not None:
                ensemble_path = models_dir / 'Ensemble.pkl'
                joblib.dump(pipeline.multiclass_ensemble, ensemble_path)
                print(f"✓ Saved multiclass ensemble model")
            else:
                print(f"⚠️  Multiclass ensemble not found, skipping")
        
        # ========== 2. SAVE SCALER (FIX KEY NAME) ==========
        scaler_path = save_dir / 'scaler.pkl'
        
        if stage == 'binary':
            if hasattr(pipeline, 'binary_scaler') and pipeline.binary_scaler:
                joblib.dump(pipeline.binary_scaler, scaler_path)
                print(f"✓ Binary scaler saved")
            elif hasattr(pipeline, 'scaler') and pipeline.scaler:
                joblib.dump(pipeline.scaler, scaler_path)
                print(f"✓ Scaler saved")
            else:
                print(f"⚠️  Binary scaler not found")
        else:
            if hasattr(pipeline, 'multiclass_scaler') and pipeline.multiclass_scaler:
                joblib.dump(pipeline.multiclass_scaler, scaler_path)
                print(f"✓ Multiclass scaler saved")
            else:
                print(f"⚠️  Multiclass scaler not found")
        
        # ========== 3. SAVE LABEL ENCODERS ==========
        if hasattr(pipeline, 'label_encoders') and pipeline.label_encoders:
            encoders_path = save_dir / 'label_encoders.pkl'
            joblib.dump(pipeline.label_encoders, encoders_path)
            print(f"✓ Label encoders saved ({len(pipeline.label_encoders)} encoders)")
            
            # ⭐ Verify fraud_type_encoder for Stage 2
            if 'fraud_type_encoder' in pipeline.label_encoders:
                print(f"  ✓ fraud_type_encoder included")
            else:
                print(f"  ⚠️  fraud_type_encoder NOT included")
        else:
            print(f"⚠️  Label encoders not found")
        
        # ========== 4. SAVE SHAP EXPLAINER ==========
        if shap_explainer is not None:
            explainer_path = save_dir / 'shap_explainer.pkl'
            try:
                import cloudpickle
                with open(explainer_path, 'wb') as f:
                    cloudpickle.dump(shap_explainer, f)
                print(f"✓ SHAP explainer saved")
            except Exception as e:
                print(f"⚠️  Could not save SHAP explainer: {e}")
        
        # ========== 5. SAVE FEATURE STATISTICS ==========
        if hasattr(pipeline, 'shap_manager') and pipeline.shap_manager:
            if hasattr(pipeline.shap_manager, 'feature_stats') and pipeline.shap_manager.feature_stats:
                stats_path = save_dir / 'feature_stats.json'
                with open(stats_path, 'w') as f:
                    json.dump(pipeline.shap_manager.feature_stats, f, indent=2)
                print(f"✓ Feature statistics saved")
        
        # ========== 6. SAVE FEATURE CONFIG ==========
        if hasattr(pipeline, 'numerical_features'):
            features_config = {
                'numerical_features': getattr(pipeline, 'numerical_features', []),
                'categorical_features': getattr(pipeline, 'categorical_features', []),
                'boolean_features': getattr(pipeline, 'boolean_features', []),
                'all_features': (
                    getattr(pipeline, 'numerical_features', []) + 
                    getattr(pipeline, 'categorical_features', []) + 
                    getattr(pipeline, 'boolean_features', [])
                ),
                'leakage_features': getattr(pipeline, 'leakage_features', [
                    'fraud_flag', 'fraud_type', 'severity', 'evidence_type',
                    'graph_pattern_id', 'graph_pattern_type'
                ])
            }
            
            features_path = save_dir / 'features_config.json'
            with open(features_path, 'w') as f:
                json.dump(features_config, f, indent=2)
            print(f"✓ Feature config saved")
        
        # ========== 7. SAVE METADATA ==========
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'stage': stage,
            'model_name': model_name,
            'timestamp': timestamp,
            'save_date': datetime.now().isoformat(),
            'artifacts_dir': str(save_dir),
            'shap_available': shap_explainer is not None,
            'has_ensemble': (
                (hasattr(pipeline, 'binary_ensemble') and pipeline.binary_ensemble is not None)
                if stage == 'binary' else
                (hasattr(pipeline, 'multiclass_ensemble') and pipeline.multiclass_ensemble is not None)
            )
        })
        
        metadata_path = save_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"✓ Metadata saved")
        
        print(f"\n✅ All artifacts saved successfully to: {save_dir}")
        return str(save_dir)


    
    def _save_models(self, pipeline: Any, stage: str, save_dir: Path):
        """Save trained models"""
        print("\n1. Saving models...")
        
        if stage == 'binary':
            models = pipeline.binary_models
            model_dir = save_dir / 'binary_models'
        else:
            models = pipeline.multiclass_models
            model_dir = save_dir / 'multiclass_models'
        
        model_dir.mkdir(exist_ok=True)
        
        for model_name, model in models.items():
            model_path = model_dir / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            print(f"   ✓ {model_name} saved: {model_path}")
    
    def _save_scaler(self, pipeline: Any, stage: str, save_dir: Path):
        """Save scaler"""
        print("\n2. Saving scaler...")
        
        scaler = pipeline.binary_scaler if stage == 'binary' else pipeline.multiclass_scaler
        
        if scaler is not None:
            scaler_path = save_dir / 'scaler.pkl'
            joblib.dump(scaler, scaler_path)
            print(f"   ✓ Scaler saved: {scaler_path}")
        else:
            print("   ⚠ No scaler found")
    
    def _save_label_encoders(self, pipeline: Any, save_dir: Path):
        """Save label encoders"""
        print("\n3. Saving label encoders...")
        
        if hasattr(pipeline, 'label_encoders') and pipeline.label_encoders:
            encoders_path = save_dir / 'label_encoders.pkl'
            joblib.dump(pipeline.label_encoders, encoders_path)
            print(f"   ✓ Label encoders saved: {encoders_path}")
        else:
            print("   ⚠ No label encoders found")
    
    def _save_shap_explainer(self, shap_explainer: Any, save_dir: Path):
        """Save SHAP explainer"""
        print("\n4. Saving SHAP explainer...")
        
        if shap_explainer is None:
            print("   ⚠ No SHAP explainer provided")
            return
        
        explainer_path = save_dir / 'shap_explainer.pkl'
        
        try:
            if CLOUDPICKLE_AVAILABLE:
                with open(explainer_path, 'wb') as f:
                    cloudpickle.dump(shap_explainer, f)
                print(f"   ✓ SHAP explainer saved (cloudpickle): {explainer_path}")
            else:
                with open(explainer_path, 'wb') as f:
                    pickle.dump(shap_explainer, f)
                print(f"   ✓ SHAP explainer saved (pickle): {explainer_path}")
        except Exception as e:
            print(f"   ⚠ Could not save SHAP explainer: {e}")
    
    def _save_feature_stats(self, pipeline: Any, save_dir: Path):
        """Save feature statistics"""
        print("\n5. Saving feature statistics...")
        
        if hasattr(pipeline, 'shap_manager') and pipeline.shap_manager is not None:
            if hasattr(pipeline.shap_manager, 'feature_stats') and pipeline.shap_manager.feature_stats:
                stats_path = save_dir / 'feature_stats.json'
                with open(stats_path, 'w') as f:
                    json.dump(pipeline.shap_manager.feature_stats, f, indent=2)
                print(f"   ✓ Feature statistics saved: {stats_path}")
            else:
                print("   ⚠ No feature statistics found")
        else:
            print("   ⚠ No SHAP manager found")
    
    def _save_feature_config(self, pipeline: Any, save_dir: Path):
        """Save feature configuration"""
        print("\n6. Saving feature configuration...")
        
        features_config = {
            'numerical_features': pipeline.numerical_features,
            'categorical_features': pipeline.categorical_features,
            'boolean_features': pipeline.boolean_features,
            'all_features': pipeline.numerical_features + pipeline.categorical_features + pipeline.boolean_features
        }
        
        config_path = save_dir / 'features_config.json'
        with open(config_path, 'w') as f:
            json.dump(features_config, f, indent=2)
        print(f"   ✓ Feature config saved: {config_path}")
    
    def _save_metadata(
        self,
        pipeline: Any,
        stage: str,
        metadata: Dict[str, Any],
        save_dir: Path
    ):
        """Save metadata"""
        print("\n7. Saving metadata...")
        
        meta = {
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'random_state': pipeline.random_state if hasattr(pipeline, 'random_state') else None,
            'enable_shap': pipeline.enable_shap if hasattr(pipeline, 'enable_shap') else False,
        }
        
        if metadata:
            meta.update(metadata)
        
        # Add optimal threshold if available
        if stage == 'binary' and hasattr(pipeline, 'optimal_threshold'):
            meta['optimal_threshold'] = float(pipeline.optimal_threshold)
            meta['threshold_metric'] = getattr(pipeline, 'threshold_metric', 'f1')
        
        metadata_path = save_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"   ✓ Metadata saved: {metadata_path}")
    
    def load_artifacts(self, artifacts_dir: str) -> Dict[str, Any]:
        """
        Load all artifacts from directory
        
        Args:
            artifacts_dir: Path to artifacts directory
        
        Returns:
            Dictionary containing all loaded artifacts
        """
        artifacts_dir = Path(artifacts_dir)
        
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
        
        print(f"\n{'='*80}")
        print(f"LOADING MODEL ARTIFACTS")
        print(f"{'='*80}")
        print(f"Load directory: {artifacts_dir}")
        
        artifacts = {}
        
        # ========== 1. LOAD METADATA ==========
        metadata_path = artifacts_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                artifacts['metadata'] = json.load(f)
            print(f"\n✓ Metadata loaded")
            print(f"  Stage: {artifacts['metadata'].get('stage')}")
            print(f"  Timestamp: {artifacts['metadata'].get('timestamp')}")
        else:
            print(f"\n⚠️  Metadata not found")
            artifacts['metadata'] = {}
        
        # ========== 2. LOAD MODELS ==========
        stage = artifacts['metadata'].get('stage', 'binary')
        model_dir = artifacts_dir / f'{stage}_models'
        
        if model_dir.exists():
            artifacts['models'] = {}
            for model_file in model_dir.glob('*.pkl'):
                model_name = model_file.stem
                artifacts['models'][model_name] = joblib.load(model_file)
                print(f"✓ Model loaded: {model_name}")
        else:
            print(f"⚠️  Model directory not found: {model_dir}")
        
        # ========== 3. LOAD SCALER (FIXED) ==========
        scaler_path = artifacts_dir / 'scaler.pkl'  # ✅ FIXED: artifacts_dir not artifacts_path
        if scaler_path.exists():
            artifacts['scaler'] = joblib.load(scaler_path)
            print(f"✓ Scaler loaded")
        else:
            print(f"⚠️  Scaler file not found: {scaler_path}")
        # ===========================================
        
        # ========== 4. LOAD LABEL ENCODERS ==========
        encoders_path = artifacts_dir / 'label_encoders.pkl'
        if encoders_path.exists():
            artifacts['label_encoders'] = joblib.load(encoders_path)
            print(f"✓ Label encoders loaded")
        else:
            print(f"⚠️  Label encoders not found")
        
        # ========== 5. LOAD SHAP EXPLAINER ==========
        explainer_path = artifacts_dir / 'shap_explainer.pkl'
        if explainer_path.exists():
            try:
                if CLOUDPICKLE_AVAILABLE:
                    with open(explainer_path, 'rb') as f:
                        artifacts['shap_explainer'] = cloudpickle.load(f)
                    print(f"✓ SHAP explainer loaded (cloudpickle)")
                else:
                    with open(explainer_path, 'rb') as f:
                        artifacts['shap_explainer'] = pickle.load(f)
                    print(f"✓ SHAP explainer loaded (pickle)")
            except Exception as e:
                print(f"⚠️  Could not load SHAP explainer: {e}")
        else:
            print(f"⚠️  SHAP explainer not found")
        
        # ========== 6. LOAD FEATURE STATISTICS ==========
        stats_path = artifacts_dir / 'feature_stats.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                artifacts['feature_stats'] = json.load(f)
            print(f"✓ Feature statistics loaded")
        else:
            print(f"⚠️  Feature statistics not found")
        
        # ========== 7. LOAD FEATURE CONFIGURATION ==========
        config_path = artifacts_dir / 'features_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                artifacts['features_config'] = json.load(f)
            print(f"✓ Feature configuration loaded")
        else:
            print(f"⚠️  Feature configuration not found")
        
        print(f"\n✅ All artifacts loaded successfully!")
        print(f"{'='*80}\n")
        
        return artifacts
