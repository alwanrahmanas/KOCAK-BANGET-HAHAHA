"""
BPJS Fraud Detection Pipeline - Training component
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, accuracy_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from models.threshold_optimizer import ThresholdOptimizer
from models.shap_manager import SHAPExplainerManager
from models.model_artifact_manager import ModelArtifactManager
from utils.logger import get_logger

logger = get_logger(__name__)


# ==================== INFERENCE ENGINE ====================

class BPJSFraudInferenceEngine:
    """Production inference engine with evaluation support + SHAP explanations"""

    def __init__(self):
        self.binary_artifacts = None
        self.multiclass_artifacts = None
        self.is_ready = False
        self.shap_manager = None


    def explain_binary_shap(
        self,
        X_scaled: np.ndarray,
        original_df: pd.DataFrame,
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Generate SHAP-based explanations for binary fraud predictions
        
        Args:
            X_scaled: Scaled features (n_samples, n_features)
            original_df: DataFrame with predictions
            top_k: Number of top features to return
        """
        # ========== PERBAIKI CHECK INI ==========
        # Check both attribute and artifacts
        explainer = None
        feature_names = None
        
        if hasattr(self, "binary_shap_explainer") and self.binary_shap_explainer is not None:
            explainer = self.binary_shap_explainer
            feature_names = self.binary_features
        elif self.binary_artifacts and self.binary_artifacts.get('shap_explainer'):
            explainer = self.binary_artifacts['shap_explainer']
            feature_names = (
                self.binary_artifacts['features_config']['numerical_features'] +
                self.binary_artifacts['features_config']['categorical_features'] +
                self.binary_artifacts['features_config']['boolean_features']
            )
        
        if explainer is None:
            print("  ⚠️  SHAP explainer not available, skipping.")
            original_df["shap_top_features"] = [[] for _ in range(len(original_df))]
            original_df["shap_explanation_summary"] = ""
            return original_df
        # ========================================

        print(f"  🔍 Computing SHAP values for {len(X_scaled)} samples...")

        # SHAP values untuk kelas 1 (fraud)
        shap_values = explainer.shap_values(X_scaled)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # kelas fraud

        all_explanations = []
        summary_list = []
        
        for i in range(len(original_df)):
            row_sv = shap_values[i]
            abs_sv = np.abs(row_sv)
            order = np.argsort(abs_sv)[::-1][:top_k]

            top_feats = []
            summary_parts = []
            
            for idx in order:
                feat_name = feature_names[idx]
                feat_value = original_df.iloc[i].get(feat_name)
                shap_val = float(row_sv[idx])
                
                top_feats.append({
                    "feature": feat_name,
                    "value": float(feat_value) if feat_value is not None else None,
                    "shap_value": shap_val,
                    "direction": "↑" if shap_val > 0 else "↓"
                })
                
                # Build summary string
                direction = "↑" if shap_val > 0 else "↓"
                summary_parts.append(f"{feat_name}{direction}")
            
            all_explanations.append(top_feats)
            summary_list.append(", ".join(summary_parts))

        original_df["shap_top_features"] = all_explanations
        original_df["shap_explanation_summary"] = summary_list
        
        print(f"  ✓ SHAP explanations generated")
        
        return original_df



    def load_models(self, binary_artifacts_dir: str, multiclass_artifacts_dir: Optional[str] = None):
        """
        Load trained models for inference with SHAP explainer verification
        
        Args:
            binary_artifacts_dir: Path to binary model artifacts directory
            multiclass_artifacts_dir: Path to multiclass model artifacts directory (optional)
        """
        manager = ModelArtifactManager()

        print("\n" + "="*80)
        print("LOADING MODELS FOR INFERENCE")
        print("="*80)

        # ========== 1. LOAD BINARY ARTIFACTS ==========
        print(f"\nLoading binary model from: {binary_artifacts_dir}")
        self.binary_artifacts = manager.load_artifacts(binary_artifacts_dir)
        
        # Extract binary model
        if 'models' in self.binary_artifacts:
            models = self.binary_artifacts['models']
            if 'Ensemble' in models:
                self.binary_model = models['Ensemble']
                print("  ✓ Using Ensemble model for binary classification")
            else:
                self.binary_model = list(models.values())[0]
                model_name = list(models.keys())[0]
                print(f"  ⚠️  Using {model_name} (Ensemble not found)")
        elif 'model' in self.binary_artifacts:
            self.binary_model = self.binary_artifacts['model']
            print("  ✓ Binary model loaded")
        else:
            raise KeyError("No model found in binary artifacts")
        
        # ========== 2. LOAD MULTICLASS ARTIFACTS (OPTIONAL) ==========
        if multiclass_artifacts_dir:
            print(f"\nLoading multiclass model from: {multiclass_artifacts_dir}")
            self.multiclass_artifacts = manager.load_artifacts(multiclass_artifacts_dir)
            
            # Extract multiclass model
            if 'models' in self.multiclass_artifacts:
                models = self.multiclass_artifacts['models']
                if 'Ensemble' in models:
                    self.multiclass_model = models['Ensemble']
                    print("  ✓ Using Ensemble model for multiclass classification")
                else:
                    self.multiclass_model = list(models.values())[0]
                    model_name = list(models.keys())[0]
                    print(f"  ⚠️  Using {model_name} (Ensemble not found)")
            elif 'model' in self.multiclass_artifacts:
                self.multiclass_model = self.multiclass_artifacts['model']
                print("  ✓ Multiclass model loaded")
        
        # ========== 3. LOAD AND VERIFY SHAP EXPLAINER ==========
        print("\n[SHAP] Verifying explainer availability...")
        
        has_shap_explainer = self.binary_artifacts.get('shap_explainer') is not None
        has_feature_stats = bool(self.binary_artifacts.get('feature_stats'))
        
        print(f"  Binary artifacts contains:")
        print(f"     - shap_explainer: {has_shap_explainer}")
        print(f"     - feature_stats: {has_feature_stats}")
        
        # Load SHAP explainer
        if has_shap_explainer:
            try:
                self.binary_shap_explainer = self.binary_artifacts['shap_explainer']
                self.binary_features = (
                    self.binary_artifacts['features_config']['numerical_features'] +
                    self.binary_artifacts['features_config']['categorical_features'] +
                    self.binary_artifacts['features_config']['boolean_features']
                )
                
                explainer_type = type(self.binary_shap_explainer).__name__
                is_callable = hasattr(self.binary_shap_explainer, 'shap_values')
                
                print(f"  ✓ Binary SHAP explainer loaded successfully")
                print(f"     - Type: {explainer_type}")
                print(f"     - Callable (has shap_values): {is_callable}")
                print(f"     - Features loaded: {len(self.binary_features)}")
                
                if not is_callable:
                    print(f"  ⚠️  WARNING: Explainer missing 'shap_values' method!")
                    self.binary_shap_explainer = None
                        
            except Exception as e:
                print(f"  ⚠️  Error loading SHAP explainer: {e}")
                self.binary_shap_explainer = None
        else:
            self.binary_shap_explainer = None
            print("  ⚠️  Binary SHAP explainer not found in artifacts")
        
        # Load feature statistics
        if has_feature_stats:
            self.feature_stats = self.binary_artifacts['feature_stats']
            print(f"  ✓ Feature statistics loaded")
        else:
            self.feature_stats = {}
            print(f"  ⚠️  Feature statistics not found")
        
        # ========== 4. INITIALIZE SHAP MANAGER ==========
        if has_shap_explainer or has_feature_stats:
            self.shap_manager = SHAPExplainerManager()
            self.shap_manager.binary_explainer = self.binary_artifacts.get('shap_explainer')
            self.shap_manager.feature_stats = self.binary_artifacts.get('feature_stats', {})
            
            # Load multiclass SHAP explainer if available
            if self.multiclass_artifacts and self.multiclass_artifacts.get('shap_explainer'):
                self.shap_manager.multiclass_explainer = self.multiclass_artifacts['shap_explainer']
                print("  ✓ Multiclass SHAP explainer loaded")
            elif self.multiclass_artifacts:
                print("  ℹ️  Multiclass using z-score fallback (SHAP not configured)")
        else:
            self.shap_manager = None
            print("  ⚠️  No SHAP explainer available. Using fallback explanations.")
        
        # ========== 5. SET READY STATE ==========
        self.is_ready = True
        
        # ========== 6. FINAL SUMMARY (ONCE!) ==========
        print("\n" + "="*60)
        if self.binary_shap_explainer:
            print("✅ Inference engine ready with SHAP explanations!")
        else:
            print("⚠️  Inference engine ready (z-score fallback mode)")
        print("="*60)



    def _prepare_features(self, df: pd.DataFrame, artifacts: Dict) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for inference using saved artifacts
        
        ⚠️ Does NOT require y labels (fraud_flag, fraud_type)
        This is for INFERENCE mode only
        
        Args:
            df: Input DataFrame
            artifacts: Loaded model artifacts
        
        Returns:
            X: Prepared feature matrix
            available_features: List of feature names
        """
        df = df.copy()
        
        features_config = artifacts.get('features_config', {})
        label_encoders = artifacts.get('label_encoders', {})
        
        if not features_config:
            raise ValueError("features_config not found in artifacts")
        
        # ========== SAFE LEAKAGE FEATURES HANDLING ==========
        # Get leakage features from config (with fallback)
        leakage_features = features_config.get('leakage_features', [
            'fraud_flag', 'fraud_type', 'severity', 'evidence_type',
            'graph_pattern_id', 'graph_pattern_type'
        ])
        
        leakage_cols = [col for col in leakage_features if col in df.columns]
        
        if leakage_cols:
            df = df.drop(columns=leakage_cols)
            if len(leakage_cols) > 0:
                print(f"  ℹ️  Dropped {len(leakage_cols)} leakage columns: {leakage_cols[:3]}{'...' if len(leakage_cols) > 3 else ''}")
        # ===================================================
        
        # Get feature lists
        numerical_features = features_config.get('numerical_features', [])
        categorical_features = features_config.get('categorical_features', [])
        boolean_features = features_config.get('boolean_features', [])
        
        # Encode categorical features
        for col in categorical_features:
            if col in df.columns:
                if col in label_encoders:
                    le = label_encoders[col]
                    # Handle unseen categories gracefully
                    df[col] = df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    print(f"  ⚠️  Warning: No encoder found for '{col}', treating as numeric")
        
        # Convert boolean features
        for col in boolean_features:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        # Select all features
        all_features = numerical_features + categorical_features + boolean_features
        available_features = [col for col in all_features if col in df.columns]
        
        # Check for missing features
        missing_features = set(all_features) - set(available_features)
        if missing_features:
            print(f"  ⚠️  Warning: {len(missing_features)} features missing from input data")
            if len(missing_features) <= 5:
                print(f"     Missing: {list(missing_features)}")
            else:
                print(f"     Missing: {list(missing_features)[:5]} ... (+{len(missing_features)-5} more)")
        
        # Extract features
        X = df[available_features].copy()
        
        # Handle missing values
        # For numerical: fill with median
        numeric_cols = [col for col in available_features if col in numerical_features]
        if numeric_cols:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # For categorical/boolean: fill with mode or -1
        cat_bool_cols = [col for col in available_features if col not in numeric_cols]
        if cat_bool_cols:
            X[cat_bool_cols] = X[cat_bool_cols].fillna(-1)
        
        print(f"  ✓ Prepared {len(available_features)} features for inference")
        
        return X, available_features


    def predict(
        self, 
        df: pd.DataFrame, 
        primary_key: str = 'claim_id',
        return_probabilities: bool = True,
        use_optimal_threshold: bool = True, 
        evaluate: bool = False,
        generate_explanations: bool = False, 
        explanation_top_k: int = 5,
        merge_with_input: bool = True
    ) -> Dict:
        """
        Make predictions with optional evaluation + local explanations + data merging
        
        Parameters:
        -----------
        df : DataFrame
            Input data for prediction
        primary_key : str
            Column name to use as primary key for joining (default: 'claim_id')
        return_probabilities : bool
            Return prediction probabilities
        use_optimal_threshold : bool
            Use saved optimal threshold (default: True)
        evaluate : bool
            Perform evaluation if ground truth available
        generate_explanations : bool
            Generate SHAP-based local explanations for fraud cases
        explanation_top_k : int
            Number of top contributing features to return in explanation
        merge_with_input : bool
            Merge predictions with original input data via primary key (default: True)
        
        Returns:
        --------
        Dict with:
            - 'predictions': DataFrame with predictions (merged with original if merge_with_input=True)
            - 'evaluation': Dict with metrics (only if evaluate=True and labels exist)
        """
        if not self.is_ready:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # ========== VALIDATE PRIMARY KEY ==========
        if primary_key not in df.columns:
            raise ValueError(
                f"Primary key '{primary_key}' not found in input DataFrame.\n"
                f"Available columns: {list(df.columns)}"
            )
        
        # ========== STORE ORIGINAL DATA FOR MERGING ==========
        df_original = df.copy()
        
        print(f"\n{'='*80}")
        print(f"RUNNING INFERENCE ON {len(df)} CLAIMS")
        if generate_explanations:
            print(f"WITH LOCAL EXPLANATIONS FOR RAG INTEGRATION")
        if merge_with_input:
            print(f"WITH AUTOMATIC DATA MERGING (primary_key='{primary_key}')")
        print(f"{'='*80}")
        
        # Check ground truth availability
        has_fraud_flag = 'fraud_flag' in df.columns
        has_fraud_type = 'fraud_type' in df.columns
        
        # Determine mode
        if evaluate:
            if not has_fraud_flag:
                raise ValueError(
                    "Evaluation mode requires 'fraud_flag' column in dataset.\n"
                    "Either:\n"
                    "  1. Add 'fraud_flag' column to your data, OR\n"
                    "  2. Set evaluate=False for pure inference mode"
                )
            print("📊 Mode: INFERENCE WITH EVALUATION")
            print(f"   Ground truth available: fraud_flag={has_fraud_flag}, fraud_type={has_fraud_type}")
        else:
            print("🔮 Mode: PURE INFERENCE (No Evaluation)")
            if has_fraud_flag:
                print("   Note: Ground truth detected but will be ignored (evaluate=False)")
        
        # Initialize results with primary key
        results = pd.DataFrame({primary_key: df[primary_key].values})
        
        # Store ground truth if available
        if has_fraud_flag:
            y_true_binary = df['fraud_flag'].values
            if evaluate:
                results['actual_fraud'] = y_true_binary
        
        if has_fraud_type and evaluate:
            results['actual_fraud_type'] = df['fraud_type'].values
        
        # ===== STAGE 1: Binary =====
        print("\n[Stage 1] Binary Fraud Detection...")
        
        X_binary, features = self._prepare_features(df, self.binary_artifacts)
        
        # ========== GET SCALER (SAFE) ==========
        if 'scaler' in self.binary_artifacts:
            scaler = self.binary_artifacts['scaler']
        elif 'binary_scaler' in self.binary_artifacts:
            scaler = self.binary_artifacts['binary_scaler']
        else:
            raise KeyError(
                f"Scaler not found in binary_artifacts!\n"
                f"Available keys: {list(self.binary_artifacts.keys())}"
            )
        
        X_binary_scaled = scaler.transform(X_binary)
        # ========================================
        
        X_binary_scaled_df = pd.DataFrame(
            X_binary_scaled, 
            columns=features, 
            index=X_binary.index
        )
        
        # ========== GET BINARY MODEL ==========
        if hasattr(self, 'binary_model') and self.binary_model is not None:
            model_binary = self.binary_model
            print(f"  ✓ Using pre-loaded binary model")
        elif 'models' in self.binary_artifacts:
            models = self.binary_artifacts['models']
            if 'Ensemble' in models:
                model_binary = models['Ensemble']
                print(f"  ✓ Using Ensemble model")
            else:
                model_binary = list(models.values())[0]
                model_name = list(models.keys())[0]
                print(f"  ⚠️  Using {model_name} (Ensemble not found)")
        elif 'model' in self.binary_artifacts:
            model_binary = self.binary_artifacts['model']
            print(f"  ✓ Using binary model from artifacts")
        else:
            raise KeyError(
                f"No model found in binary_artifacts.\n"
                f"Available keys: {list(self.binary_artifacts.keys())}"
            )
        # =====================================
        
        # Predict probabilities
        fraud_proba = model_binary.predict_proba(X_binary_scaled)[:, 1]
        
        # Use optimal threshold or default
        if use_optimal_threshold and self.binary_artifacts.get('metadata', {}).get('optimal_threshold'):
            fraud_threshold = self.binary_artifacts['metadata']['optimal_threshold']
            print(f"  Using optimal threshold: {fraud_threshold:.4f}")
        else:
            fraud_threshold = 0.5
            print(f"  Using default threshold: {fraud_threshold:.4f}")
        
        fraud_pred = (fraud_proba >= fraud_threshold).astype(int)
        
        results['predicted_fraud'] = fraud_pred
        results['fraud_probability'] = fraud_proba
        results['fraud_label'] = results['predicted_fraud'].map({0: 'BENIGN', 1: 'FRAUD'})
        results['threshold_used'] = fraud_threshold
        
        fraud_count = fraud_pred.sum()
        print(f"  ✓ Predicted {fraud_count} fraud cases ({fraud_count/len(df)*100:.2f}%)")
        
        # ===== GENERATE SHAP EXPLANATIONS =====
        if generate_explanations and fraud_count > 0:
            print(f"\n[Explanations] Generating SHAP explanations for {fraud_count} fraud cases...")
            
            if hasattr(self, 'binary_shap_explainer') and self.binary_shap_explainer is not None:
                results = self.explain_binary_shap(
                    X_binary_scaled,
                    results,
                    top_k=explanation_top_k
                )
                print(f"  ✓ SHAP explanations generated")
            else:
                print(f"  ⚠️  SHAP explainer not available, skipping")
        
        # ===== STAGE 2: Multi-Class =====
        if self.multiclass_artifacts and fraud_count > 0:
            print("\n[Stage 2] Fraud Type Classification...")
            
            fraud_mask = fraud_pred == 1
            X_fraud = X_binary_scaled_df[fraud_mask]
            
            # Get multiclass model
            if hasattr(self, 'multiclass_model') and self.multiclass_model is not None:
                model_multiclass = self.multiclass_model
            elif 'models' in self.multiclass_artifacts:
                models = self.multiclass_artifacts['models']
                model_multiclass = models.get('Ensemble', list(models.values())[0])
            elif 'model' in self.multiclass_artifacts:
                model_multiclass = self.multiclass_artifacts['model']
            else:
                raise KeyError("No model found in multiclass_artifacts")
            
            fraud_type_pred_encoded = model_multiclass.predict(X_fraud)
            
            # ========== GET FRAUD TYPE ENCODER ==========
            fraud_type_encoder = None
            
            # Try binary artifacts first
            if 'label_encoders' in self.binary_artifacts:
                if 'fraud_type_encoder' in self.binary_artifacts['label_encoders']:
                    fraud_type_encoder = self.binary_artifacts['label_encoders']['fraud_type_encoder']
                    print(f"  ✓ Fraud type encoder found in binary artifacts")
                    print(f"     Classes: {list(fraud_type_encoder.classes_)}")
            
            # Try multiclass artifacts as fallback
            if fraud_type_encoder is None and 'label_encoders' in self.multiclass_artifacts:
                if 'fraud_type_encoder' in self.multiclass_artifacts['label_encoders']:
                    fraud_type_encoder = self.multiclass_artifacts['label_encoders']['fraud_type_encoder']
                    print(f"  ✓ Fraud type encoder found in multiclass artifacts")
                    print(f"     Classes: {list(fraud_type_encoder.classes_)}")
            
            # Decode fraud types
            if fraud_type_encoder is not None:
                try:
                    fraud_type_names = fraud_type_encoder.inverse_transform(
                        fraud_type_pred_encoded.astype(int)
                    )
                    print(f"  ✓ Decoded {len(fraud_type_names)} fraud type predictions")
                except Exception as e:
                    print(f"  ⚠️  Error decoding: {e}")
                    fraud_type_names = fraud_type_pred_encoded
            else:
                print(f"  ⚠️  Fraud type encoder NOT FOUND!")
                print(f"     Available in binary artifacts:")
                if 'label_encoders' in self.binary_artifacts:
                    print(f"       label_encoders keys: {list(self.binary_artifacts['label_encoders'].keys())}")
                else:
                    print(f"       label_encoders: NOT PRESENT")
                print(f"     Using raw encoded predictions (numbers)")
                fraud_type_names = fraud_type_pred_encoded
            # ==========================================
            
            results.loc[fraud_mask, 'predicted_fraud_type'] = fraud_type_names
            
            # Add probabilities if encoder available
            if return_probabilities and fraud_type_encoder is not None:
                fraud_type_proba = model_multiclass.predict_proba(X_fraud)
                fraud_types = fraud_type_encoder.classes_
                
                for i, fraud_type in enumerate(fraud_types):
                    results.loc[fraud_mask, f'prob_{fraud_type}'] = fraud_type_proba[:, i]
            
            print(f"  ✓ Classified {fraud_count} fraud claims")
        
        results['predicted_fraud_type'] = results.get('predicted_fraud_type', 'benign').fillna('benign')
        
        # ========== MERGE WITH ORIGINAL INPUT DATA ==========
        if merge_with_input:
            print(f"\n{'='*80}")
            print("MERGING PREDICTIONS WITH ORIGINAL INPUT DATA")
            print(f"{'='*80}")
            
            print(f"  Primary key: {primary_key}")
            print(f"  Original data columns: {len(df_original.columns)}")
            print(f"  Prediction columns: {len(results.columns)}")
            
            pred_cols = [col for col in results.columns if col not in df_original.columns or col == primary_key]
            
            try:
                final_results = df_original.merge(
                    results[pred_cols],
                    on=primary_key,
                    how='left'
                )
                
                prediction_cols = [col for col in pred_cols if col != primary_key]
                original_cols = [col for col in df_original.columns if col != primary_key]
                
                column_order = [primary_key] + prediction_cols + original_cols
                final_results = final_results[column_order]
                
                print(f"  ✓ Merge successful!")
                print(f"  Final columns: {len(final_results.columns)}")
                print(f"  Column order: {primary_key}, predictions, original_data")
                
                sample_pred_cols = prediction_cols[:5]
                sample_orig_cols = original_cols[:5]
                print(f"\n  Sample prediction columns: {sample_pred_cols}")
                print(f"  Sample original columns: {sample_orig_cols}")
                
                results = final_results
                
            except Exception as e:
                print(f"  ⚠️  Merge failed: {e}")
                print(f"  Returning predictions only (without original data)")
        
        # ===== EVALUATION =====
        evaluation_results = None
        
        if evaluate and has_fraud_flag:
            print(f"\n{'='*80}")
            print("EVALUATION MODE")
            print(f"{'='*80}")
            
            evaluation_results = self._evaluate_predictions(
                results, y_true_binary, has_fraud_type, fraud_threshold
            )
        
        print(f"\n✅ Inference completed")
        print(f"{'='*80}\n")
        
        return {
            'predictions': results,
            'evaluation': evaluation_results
        }




    def _generate_explanations(
    self, 
    results: pd.DataFrame, 
    df_original: pd.DataFrame,
    X_binary: pd.DataFrame, 
    X_binary_scaled: pd.DataFrame,
    features: List[str], 
    fraud_pred: np.ndarray, 
    top_k: int = 5
) -> pd.DataFrame:
        """
        SIMPLIFIED: Wrapper around explain_binary_shap with backward compatibility
        
        This method maintains compatibility with existing code that calls
        _generate_explanations while delegating to the main explain_binary_shap method.
        """
        # Just call explain_binary_shap - it already handles everything
        results = self.explain_binary_shap(
            X_binary_scaled.values,
            results,
            top_k=top_k
        )
        
        # Add backward-compatible columns if needed
        fraud_mask = fraud_pred == 1
        
        if "shap_top_features" in results.columns and fraud_mask.sum() > 0:
            # Create explanation_summary from shap_top_features for RAG compatibility
            def format_for_rag(row):
                if row['predicted_fraud'] == 0:
                    return ""
                
                top_feats = row.get('shap_top_features', [])
                if not top_feats:
                    return "No explanation available"
                
                # Format for human readability
                parts = []
                for feat in top_feats[:3]:
                    fname = self._get_readable_feature_name(feat['feature'])
                    direction = feat['direction']
                    value = feat.get('value', 0)
                    
                    # Handle different value types
                    if isinstance(value, (int, float)) and abs(value) > 1000:
                        value_str = f"{value:,.0f}"
                    elif isinstance(value, (int, float)):
                        value_str = f"{value:.2f}"
                    else:
                        value_str = str(value)
                    
                    parts.append(f"{fname}{direction} ({value_str})")
                
                return " | ".join(parts)
            
            # Add formatted columns for RAG/UI consumption
            results.loc[fraud_mask, 'explanation_summary'] = results[fraud_mask].apply(
                format_for_rag, axis=1
            )
            
            # Keep raw JSON for detailed analysis
            results.loc[fraud_mask, 'explanation_json'] = results.loc[fraud_mask, 'shap_top_features'].apply(
                lambda x: json.dumps(x, default=str) if x else "{}"
            )
            
            # Create simple top_features string (for backward compatibility)
            results.loc[fraud_mask, 'top_features'] = results.loc[fraud_mask, 'shap_explanation_summary']
        
        return results


    def _get_readable_feature_name(self, feature_name: str) -> str:
        """Convert technical feature names to human-readable names"""
        name_mapping = {
            'claim_ratio': 'Claim Ratio',
            'selisih_klaim': 'Claim Difference',
            'billed_amount': 'Billed Amount',
            'paid_amount': 'Paid Amount',
            'drug_cost': 'Drug Cost',
            'procedure_cost': 'Procedure Cost',
            'lama_dirawat': 'Length of Stay',
            'age': 'Patient Age',
            'tarif_inacbg': 'INA-CBG Tariff',
            'drug_ratio': 'Drug Ratio',
            'procedure_ratio': 'Procedure Ratio',
            'clinical_pathway_deviation_score': 'Clinical Deviation',
            'visit_count_30d': '30-Day Visit Count',
            'provider_monthly_claims': 'Provider Monthly Claims'
        }
        return name_mapping.get(feature_name, feature_name.replace('_', ' ').title())

        """
        Make predictions with optional evaluation

        Parameters:
        -----------
        df : DataFrame
            Input data for prediction
            - For pure inference: Only X features needed (no fraud_flag, fraud_type)
            - For evaluation: Must include fraud_flag (and optionally fraud_type)
        return_probabilities : bool
            Return prediction probabilities
        use_optimal_threshold : bool
            Use saved optimal threshold (default: True)
        evaluate : bool
            Perform evaluation if ground truth available
            - Automatically detects if 'fraud_flag' column exists
            - If False: Pure inference mode (ignores ground truth even if present)
            - If True: Requires 'fraud_flag' column, otherwise raises error

        Returns:
        --------
        Dict with:
            - 'predictions': DataFrame with predictions
            - 'evaluation': Dict with metrics (only if evaluate=True and labels exist)
        """
        if not self.is_ready:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        print(f"\n{'='*80}")
        print(f"RUNNING INFERENCE ON {len(df)} CLAIMS")
        print(f"{'='*80}")

        # Check ground truth availability
        has_fraud_flag = 'fraud_flag' in df.columns
        has_fraud_type = 'fraud_type' in df.columns

        # Determine mode
        if evaluate:
            if not has_fraud_flag:
                raise ValueError(
                    "Evaluation mode requires 'fraud_flag' column in dataset.\n"
                    "Either:\n"
                    "  1. Add 'fraud_flag' column to your data, OR\n"
                    "  2. Set evaluate=False for pure inference mode"
                )
            print("📊 Mode: INFERENCE WITH EVALUATION")
            print(f"   Ground truth available: fraud_flag={has_fraud_flag}, fraud_type={has_fraud_type}")
        else:
            print("🔮 Mode: PURE INFERENCE (No Evaluation)")
            if has_fraud_flag:
                print("   Note: Ground truth detected but will be ignored (evaluate=False)")

        results = df[['claim_id']].copy() if 'claim_id' in df.columns else pd.DataFrame(index=df.index)

        # Store ground truth if available (for evaluation later)
        if has_fraud_flag:
            y_true_binary = df['fraud_flag'].values
            if evaluate:  # Only add to results if evaluation mode
                results['actual_fraud'] = y_true_binary

        if has_fraud_type and evaluate:
            results['actual_fraud_type'] = df['fraud_type'].values

        # ===== STAGE 1: Binary =====
        print("\n[Stage 1] Binary Fraud Detection...")

        X_binary, features = self._prepare_features(df, self.binary_artifacts)
        X_binary_scaled = self.binary_artifacts['scaler'].transform(X_binary)
        X_binary_scaled = pd.DataFrame(X_binary_scaled, columns=features)

        model_binary = self.binary_artifacts['model']
        fraud_proba = model_binary.predict_proba(X_binary_scaled)[:, 1]

        # Use optimal threshold or default
        if use_optimal_threshold and self.binary_artifacts['metadata'].get('optimal_threshold'):
            fraud_threshold = self.binary_artifacts['metadata']['optimal_threshold']
            print(f"  Using optimal threshold: {fraud_threshold:.4f}")
        else:
            fraud_threshold = 0.5
            print(f"  Using default threshold: {fraud_threshold:.4f}")

        fraud_pred = (fraud_proba >= fraud_threshold).astype(int)

        results['predicted_fraud'] = fraud_pred
        results['fraud_probability'] = fraud_proba
        results['fraud_label'] = results['predicted_fraud'].map({0: 'BENIGN', 1: 'FRAUD'})
        results['threshold_used'] = fraud_threshold

        fraud_count = fraud_pred.sum()
        print(f"  ✓ Predicted {fraud_count} fraud cases ({fraud_count/len(df)*100:.2f}%)")

        # Generate SHAP explanations immediately after binary prediction
        if generate_explanations and fraud_count > 0:
            if hasattr(self, 'binary_shap_explainer') and self.binary_shap_explainer is not None:
                print(f"  🔍 Generating SHAP explanations for {fraud_count} fraud cases...")
                results = self.explain_binary_shap(
                    X_binary_scaled,  # Convert to numpy array
                    results,
                    top_k=explanation_top_k
                )
                print(f"  ✓ SHAP explanations added to results")
            else:
                print(f"  ⚠️  SHAP explainer not available, skipping explanation generation")
        # ========================================
        # ===== STAGE 2: Multi-Class =====
        if self.multiclass_artifacts and fraud_count > 0:
            print("\n[Stage 2] Fraud Type Classification...")

            fraud_mask = fraud_pred == 1
            X_fraud = X_binary_scaled[fraud_mask]

            model_multiclass = self.multiclass_artifacts['model']
            fraud_type_pred_encoded = model_multiclass.predict(X_fraud)

            fraud_type_encoder = self.binary_artifacts['label_encoders']['fraud_type_encoder']
            fraud_type_names = fraud_type_encoder.inverse_transform(fraud_type_pred_encoded)

            results.loc[fraud_mask, 'predicted_fraud_type'] = fraud_type_names

            if return_probabilities:
                fraud_type_proba = model_multiclass.predict_proba(X_fraud)
                fraud_types = fraud_type_encoder.classes_

                for i, fraud_type in enumerate(fraud_types):
                    results.loc[fraud_mask, f'prob_{fraud_type}'] = fraud_type_proba[:, i]

            print(f"  ✓ Classified {fraud_count} fraud claims")

        results['predicted_fraud_type'] = results.get('predicted_fraud_type', 'benign').fillna('benign')

        # ===== EVALUATION =====
        evaluation_results = None

        if evaluate and has_fraud_flag:
            print(f"\n{'='*80}")
            print("EVALUATION MODE")
            print(f"{'='*80}")

            evaluation_results = self._evaluate_predictions(
                results, y_true_binary, has_fraud_type, fraud_threshold
            )

        print(f"\n✅ Inference completed")

        return {
            'predictions': results,
            'evaluation': evaluation_results
        }

    def _evaluate_predictions(self, results: pd.DataFrame, y_true_binary: np.ndarray,
                             has_fraud_type: bool, threshold: float) -> Dict:
        """Evaluate predictions against ground truth"""
        evaluation = {}

        # ===== STAGE 1 EVAL =====
        print("\n[Evaluation] Stage 1 - Binary")

        y_pred_binary = results['predicted_fraud'].values
        y_proba_binary = results['fraud_probability'].values

        binary_metrics = {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
            'roc_auc': roc_auc_score(y_true_binary, y_proba_binary),
            'matthews_corrcoef': matthews_corrcoef(y_true_binary, y_pred_binary),
            'cohen_kappa': cohen_kappa_score(y_true_binary, y_pred_binary),
            'threshold': threshold
        }

        cm = confusion_matrix(y_true_binary, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()

        binary_metrics['confusion_matrix'] = {
            'true_negative': int(tn), 'false_positive': int(fp),
            'false_negative': int(fn), 'true_positive': int(tp)
        }

        binary_metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        binary_metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        binary_metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        binary_metrics['classification_report'] = classification_report(
            y_true_binary, y_pred_binary, target_names=['Benign', 'Fraud'], output_dict=True
        )

        evaluation['stage1_binary'] = binary_metrics

        print(f"  Accuracy:  {binary_metrics['accuracy']:.4f}")
        print(f"  Precision: {binary_metrics['precision']:.4f}")
        print(f"  Recall:    {binary_metrics['recall']:.4f}")
        print(f"  F1-Score:  {binary_metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {binary_metrics['roc_auc']:.4f}")
        print(f"  MCC:       {binary_metrics['matthews_corrcoef']:.4f}")
        print(f"\n  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        # ===== STAGE 2 EVAL =====
        if has_fraud_type and 'predicted_fraud_type' in results.columns:
            print("\n[Evaluation] Stage 2 - Multiclass")

            fraud_mask = results['actual_fraud'] == 1

            if fraud_mask.sum() > 0:
                y_true_type = results.loc[fraud_mask, 'actual_fraud_type'].values
                y_pred_type = results.loc[fraud_mask, 'predicted_fraud_type'].values

                unique_types = np.unique(np.concatenate([y_true_type, y_pred_type]))
                type_to_idx = {t: i for i, t in enumerate(unique_types)}

                y_true_type_encoded = np.array([type_to_idx[t] for t in y_true_type])
                y_pred_type_encoded = np.array([type_to_idx[t] for t in y_pred_type])

                multiclass_metrics = {
                    'accuracy': accuracy_score(y_true_type_encoded, y_pred_type_encoded),
                    'precision_macro': precision_score(y_true_type_encoded, y_pred_type_encoded,
                                                       average='macro', zero_division=0),
                    'recall_macro': recall_score(y_true_type_encoded, y_pred_type_encoded,
                                                 average='macro', zero_division=0),
                    'f1_score_macro': f1_score(y_true_type_encoded, y_pred_type_encoded,
                                               average='macro', zero_division=0),
                    'f1_score_weighted': f1_score(y_true_type_encoded, y_pred_type_encoded,
                                                  average='weighted', zero_division=0)
                }

                cm_multiclass = confusion_matrix(y_true_type, y_pred_type, labels=unique_types)
                multiclass_metrics['confusion_matrix'] = cm_multiclass.tolist()
                multiclass_metrics['class_labels'] = unique_types.tolist()

                multiclass_metrics['classification_report'] = classification_report(
                    y_true_type, y_pred_type, output_dict=True, zero_division=0
                )

                evaluation['stage2_multiclass'] = multiclass_metrics

                print(f"  Accuracy:        {multiclass_metrics['accuracy']:.4f}")
                print(f"  F1 (macro):      {multiclass_metrics['f1_score_macro']:.4f}")
                print(f"  F1 (weighted):   {multiclass_metrics['f1_score_weighted']:.4f}")

        # Summary
        evaluation['summary'] = {
            'total_samples': len(results),
            'actual_fraud_count': int(y_true_binary.sum()),
            'predicted_fraud_count': int(y_pred_binary.sum()),
            'actual_fraud_rate': float(y_true_binary.mean()),
            'predicted_fraud_rate': float(y_pred_binary.mean()),
            'correctly_classified': int((y_true_binary == y_pred_binary).sum()),
            'misclassified': int((y_true_binary != y_pred_binary).sum())
        }

        return evaluation

    def predict_single(self, claim_data: Dict, evaluate: bool = False) -> Dict:
        """Predict single claim"""
        df = pd.DataFrame([claim_data])
        result = self.predict(df, evaluate=evaluate)

        prediction_dict = result['predictions'].iloc[0].to_dict()
        if result['evaluation']:
            prediction_dict['evaluation'] = result['evaluation']

        return prediction_dict