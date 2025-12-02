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


class BPJSFraudInferenceEngine:
    """
    🔧 FIXED: Production inference engine with robust error handling
    """

    def __init__(self):
        self.binary_artifacts = None
        self.multiclass_artifacts = None
        self.is_ready = False
        self.shap_manager = None
        self.binary_shap_explainer = None
        self.binary_features = None
        self.feature_stats = {}

    def load_models(
        self, 
        binary_artifacts_dir: str = r"C:\Users\US3R\OneDrive\Dokumen\Data_Science\Project\MLInference\artifacts\binary_ensemble_20251119_190009",
        multiclass_artifacts_dir: Optional[str] = r"C:\Users\US3R\OneDrive\Dokumen\Data_Science\Project\MLInference\artifacts\multiclass_ensemble_20251119_190010"
    ):
        """
        Load trained model artifacts from absolute directories
        """
        manager = ModelArtifactManager()

        print("\n" + "="*80)
        print("LOADING MODELS FOR INFERENCE")
        print("="*80)

        try:
            # Load binary artifacts
            if binary_artifacts_dir is None:
                raise ValueError("binary_artifacts_dir is required")
            print(f"[Binary] Loading: {binary_artifacts_dir}")
            self.binary_artifacts = manager.load_pipeline_artifacts(binary_artifacts_dir)

            # Load multiclass
            if multiclass_artifacts_dir:
                print(f"[Multiclass] Loading: {multiclass_artifacts_dir}")
                self.multiclass_artifacts = manager.load_pipeline_artifacts(multiclass_artifacts_dir)
            else:
                print("[Multiclass] Not provided — multiclass disabled")

            # SHAP
            self._setup_shap_explainer()

            self.is_ready = True
            print("\n" + "="*60)
            print("✅ Inference engine ready!")
            print("="*60)

        except Exception as e:
            print(f"\n❌ Failed to load models: {e}")
            raise


    def _setup_shap_explainer(self):
        """Setup SHAP explainer with error handling"""
        print("\n[SHAP] Verifying explainer...")

        has_shap = self.binary_artifacts.get('shap_explainer') is not None
        has_stats = bool(self.binary_artifacts.get('feature_stats'))

        print(f"  📦 SHAP explainer available: {has_shap}")
        print(f"  📦 Feature statistics available: {has_stats}")

        if has_shap:
            try:
                self.binary_shap_explainer = self.binary_artifacts['shap_explainer']
                self.binary_features = (
                    self.binary_artifacts['features_config']['numerical_features'] +
                    self.binary_artifacts['features_config']['categorical_features'] +
                    self.binary_artifacts['features_config']['boolean_features']
                )

                # Verify explainer is usable
                if hasattr(self.binary_shap_explainer, 'shap_values'):
                    print(f"  ✅ SHAP explainer loaded and ready")
                    print(f"     Features: {len(self.binary_features)}")
                else:
                    print(f"  ⚠️  Explainer missing required methods")
                    self.binary_shap_explainer = None

            except Exception as e:
                print(f"  ❌ SHAP explainer load failed: {e}")
                self.binary_shap_explainer = None

        if has_stats:
            self.feature_stats = self.binary_artifacts.get('feature_stats', {})
            print(f"  ✅ Feature statistics loaded")

        # Initialize SHAP manager
        if has_shap or has_stats:
            self.shap_manager = SHAPExplainerManager()
            self.shap_manager.binary_explainer = self.binary_shap_explainer
            self.shap_manager.feature_stats = self.feature_stats
            self.shap_manager.is_ready = self.binary_shap_explainer is not None

    def _prepare_features(
        self, 
        df: pd.DataFrame, 
        artifacts: Dict
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        🔧 FIXED: Prepare features for inference
        """
        df = df.copy()

        features_config = artifacts['features_config']
        label_encoders = artifacts['label_encoders']

        # Drop leakage features if present
        leakage_cols = [
            col for col in features_config.get('leakage_features', [])
            if col in df.columns
        ]
        if leakage_cols:
            df = df.drop(columns=leakage_cols)

        # Encode categorical
        for col in features_config['categorical_features']:
            if col in df.columns and col in label_encoders:
                le = label_encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # Convert boolean
        for col in features_config['boolean_features']:
            if col in df.columns:
                df[col] = df[col].astype(int)

        # Select features
        all_features = (
            features_config['numerical_features'] +
            features_config['categorical_features'] +
            features_config['boolean_features']
        )
        
        available_features = [col for col in all_features if col in df.columns]
        missing_features = [col for col in all_features if col not in df.columns]

        if missing_features:
            print(f"  ⚠️  Missing {len(missing_features)} features")
            if len(missing_features) <= 10:
                print(f"     {missing_features}")

        X = df[available_features].copy()
        
        # Handle NaN
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        X = X.fillna(0)

        return X, available_features
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

        if not self.is_ready:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Keep original data for merging
        df_original = df.copy()

        print(f"\n{'='*80}")
        print(f"RUNNING INFERENCE ON {len(df)} CLAIMS")
        print(f"Primary key: {primary_key}")
        if generate_explanations:
            print(f"WITH LOCAL EXPLANATIONS FOR RAG INTEGRATION")
        if merge_with_input:
            print(f"WITH AUTOMATIC MERGING ENABLED")
        print(f"{'='*80}")

        # Validate primary key
        if primary_key not in df.columns:
            raise ValueError(
                f"Primary key '{primary_key}' not found in input DataFrame.\n"
                f"Available columns: {list(df.columns)}"
            )

        # Keep track of ground truth if exists
        has_fraud_flag = 'fraud_flag' in df.columns
        has_fraud_type = 'fraud_type' in df.columns

        results = pd.DataFrame({primary_key: df[primary_key].values})

        if has_fraud_flag:
            y_true_binary = df['fraud_flag'].values
            if evaluate:
                results['actual_fraud'] = y_true_binary

        if has_fraud_type and evaluate:
            results['actual_fraud_type'] = df['fraud_type'].values

        # ===== Stage 1: Binary Fraud Detection =====
        print("\n[Stage 1] Binary Fraud Detection...")

        X_binary, features = self._prepare_features(df, self.binary_artifacts)

        scaler = (
            self.binary_artifacts['scaler']
            if 'scaler' in self.binary_artifacts
            else self.binary_artifacts['binary_scaler']
        )
        print("X_binary.columns:", list(X_binary.columns))
        print("X_binary.shape:", X_binary.shape)

        X_binary_scaled = scaler.transform(X_binary)
        X_binary_scaled_df = pd.DataFrame(X_binary_scaled, columns=features, index=X_binary.index)
        logger.error(f"X_binary columns: {list(X_binary.columns)}")
        logger.error(f"X_binary shape: {X_binary.shape}")

        
        model_binary = (
            self.binary_artifacts.get('model') or
            self.binary_artifacts.get('models', {}).get('Ensemble')
        )
        fraud_proba = model_binary.predict_proba(X_binary_scaled)[:, 1]

        # Threshold
        threshold = (
            self.binary_artifacts.get('metadata', {}).get('optimal_threshold', 0.5)
            if use_optimal_threshold
            else 0.5
        )
        print(f"  Using threshold: {threshold:.4f}")

        fraud_pred = (fraud_proba >= threshold).astype(int)

        results['predicted_fraud'] = fraud_pred
        results['fraud_probability'] = fraud_proba
        results['fraud_label'] = results['predicted_fraud'].map({0: 'BENIGN', 1: 'FRAUD'})
        results['threshold_used'] = threshold

        fraud_count = fraud_pred.sum()
        print(f"  ✓ Found {fraud_count} fraud cases")

        # ===== Stage 2: Multiclass Fraud Type =====
        if self.multiclass_artifacts and fraud_count > 0:
            print("\n[Stage 2] Fraud Type Classification...")

            fraud_mask = fraud_pred == 1
            X_fraud = X_binary_scaled_df[fraud_mask]

            if hasattr(self, "multiclass_model") and self.multiclass_model is not None:
                model_m = self.multiclass_model
            elif "model" in self.multiclass_artifacts:
                model_m = self.multiclass_artifacts["model"]
            elif "models" in self.multiclass_artifacts:
                model_m = list(self.multiclass_artifacts["models"].values())[0]
            else:
                raise RuntimeError("No multiclass model found in multiclass_artifacts.")

            fraud_type_pred = model_m.predict(X_fraud)

            encoder = (
                self.binary_artifacts.get('label_encoders', {}).get('fraud_type_encoder') or
                self.multiclass_artifacts.get('label_encoders', {}).get('fraud_type_encoder')
            )

            if encoder is not None:
                decoded = encoder.inverse_transform(fraud_type_pred)
                results.loc[fraud_mask, 'predicted_fraud_type'] = decoded
            else:
                results.loc[fraud_mask, 'predicted_fraud_type'] = fraud_type_pred

            if return_probabilities:
                type_proba = model_m.predict_proba(X_fraud)
                for i, ftype in enumerate(encoder.classes_):
                    results.loc[fraud_mask, f'prob_{ftype}'] = type_proba[:, i]

                if 'predicted_fraud_type' not in results.columns:
                    results['predicted_fraud_type'] = 'benign'
                else:
                    results['predicted_fraud_type'] = results['predicted_fraud_type'].fillna('benign')
                    print(f"  ✓ Classified fraud types for {fraud_count} cases")   


        # ===== SHAP Explanation =====
        if generate_explanations and fraud_count > 0:
            print("\n[Explanations] Generating SHAP explanations...")
            results = self.explain_binary_shap(
                X_binary_scaled, results, top_k=explanation_top_k
            )

        # ===== MERGE WITH ORIGINAL DF =====
        if merge_with_input:
            print("\n[MERGING DATA]")

            try:
                merged = df_original.merge(results, on=primary_key, how='left')
                results = merged
                print("  ✓ Merge completed successfully")
            except Exception as e:
                print(f"  ⚠ Merge failed: {e}")

        print("\nInference finished\n")
        return {
            "predictions": results,
            "evaluation": None
        }


    def _generate_explanations(
        self,
        results: pd.DataFrame,
        X_scaled: np.ndarray,
        features: List[str],
        fraud_mask: np.ndarray,
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        🔧 FIXED: Generate explanations with robust error handling
        """
        fraud_indices = np.where(fraud_mask)[0]

        if len(fraud_indices) == 0:
            return results

        # Initialize explanation columns
        results['shap_explanation_summary'] = ''
        results['shap_top_features'] = ''

        # Use SHAP if available
        if self.binary_shap_explainer and self.shap_manager:
            try:
                print(f"  🔍 Computing SHAP values...")

                # Get SHAP values for all fraud cases
                X_fraud = X_scaled[fraud_mask]
                shap_values = self.binary_shap_explainer.shap_values(X_fraud)

                # Handle different output formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Fraud class

                # Generate explanations for each fraud case
                for i, idx in enumerate(fraud_indices):
                    try:
                        shap_vals = shap_values[i] if len(shap_values.shape) > 1 else shap_values
                        
                        # Get top features
                        abs_shap = np.abs(shap_vals)
                        top_indices = np.argsort(abs_shap)[-top_k:][::-1]

                        top_features = []
                        for feat_idx in top_indices:
                            feature_name = features[feat_idx]
                            shap_value = float(shap_vals[feat_idx])
                            feature_value = float(X_scaled[idx, feat_idx])

                            direction = "↑" if shap_value > 0 else "↓"

                            top_features.append({
                                'feature': feature_name,
                                'value': feature_value,
                                'shap_value': shap_value,
                                'direction': direction
                            })

                        summary = ", ".join([
                            f"{f['feature']}{f['direction']}" 
                            for f in top_features[:3]
                        ])

                        results.at[idx, 'shap_explanation_summary'] = summary
                        results.at[idx, 'shap_top_features'] = str(top_features)

                    except Exception as e:
                        print(f"  ⚠️  Failed to explain case {idx}: {e}")

                print(f"  ✓ SHAP explanations generated")

            except Exception as e:
                print(f"  ⚠️  SHAP computation failed: {e}")
                print(f"  ℹ️  Falling back to z-score explanations")
                results = self._generate_fallback_explanations(
                    results, X_scaled, features, fraud_indices, top_k
                )

        else:
            # Use fallback explanations
            print(f"  ℹ️  Using z-score explanations (SHAP not available)")
            results = self._generate_fallback_explanations(
                results, X_scaled, features, fraud_indices, top_k
            )

        return results

    def _generate_fallback_explanations(
        self,
        results: pd.DataFrame,
        X_scaled: np.ndarray,
        features: List[str],
        fraud_indices: np.ndarray,
        top_k: int = 5
    ) -> pd.DataFrame:
        """Generate z-score based explanations"""
        for idx in fraud_indices:
            try:
                explanations = []

                for i, feat_name in enumerate(features):
                    feat_val = float(X_scaled[idx, i])
                    feat_mean = self.feature_stats.get('mean', {}).get(feat_name, 0)
                    feat_std = self.feature_stats.get('std', {}).get(feat_name, 1)

                    z_score = (feat_val - feat_mean) / feat_std if feat_std > 0 else 0

                    if abs(z_score) > 1:  # Significant deviation
                        explanations.append({
                            'feature': feat_name,
                            'value': feat_val,
                            'z_score': float(z_score),
                            'direction': "↑" if z_score > 0 else "↓"
                        })

                explanations.sort(key=lambda x: abs(x['z_score']), reverse=True)
                top_features = explanations[:top_k]

                summary = ", ".join([
                    f"{f['feature']}{f['direction']}" 
                    for f in top_features[:3]
                ])

                results.at[idx, 'shap_explanation_summary'] = summary
                results.at[idx, 'shap_top_features'] = str(top_features)

            except Exception as e:
                print(f"  ⚠️  Failed fallback explanation for {idx}: {e}")

        return results

    def _evaluate_predictions(
        self,
        results: pd.DataFrame,
        y_true_binary: np.ndarray,
        has_fraud_type: bool,
        threshold: float
    ) -> Dict:
        """Evaluate predictions against ground truth"""
        evaluation = {}

        # Binary evaluation
        print("\n[Evaluation] Stage 1 - Binary")

        y_pred_binary = results['predicted_fraud'].values
        y_proba_binary = results['fraud_probability'].values

        try:
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
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            }

            binary_metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

            evaluation['stage1_binary'] = binary_metrics

            print(f"  Accuracy:  {binary_metrics['accuracy']:.4f}")
            print(f"  Precision: {binary_metrics['precision']:.4f}")
            print(f"  Recall:    {binary_metrics['recall']:.4f}")
            print(f"  F1-Score:  {binary_metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:   {binary_metrics['roc_auc']:.4f}")
            print(f"\n  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        except Exception as e:
            print(f"  ⚠️  Binary evaluation failed: {e}")

        # Multiclass evaluation
        if has_fraud_type and 'predicted_fraud_type' in results.columns:
            print("\n[Evaluation] Stage 2 - Multiclass")

            try:
                fraud_mask = results['actual_fraud'] == 1

                if fraud_mask.sum() > 0:
                    y_true_type = results.loc[fraud_mask, 'actual_fraud_type'].values
                    y_pred_type = results.loc[fraud_mask, 'predicted_fraud_type'].values

                    multiclass_metrics = {
                        'accuracy': accuracy_score(y_true_type, y_pred_type),
                        'precision_macro': precision_score(
                            y_true_type, y_pred_type, 
                            average='macro', zero_division=0
                        ),
                        'recall_macro': recall_score(
                            y_true_type, y_pred_type, 
                            average='macro', zero_division=0
                        ),
                        'f1_score_macro': f1_score(
                            y_true_type, y_pred_type, 
                            average='macro', zero_division=0
                        ),
                        'f1_score_weighted': f1_score(
                            y_true_type, y_pred_type, 
                            average='weighted', zero_division=0
                        )
                    }

                    evaluation['stage2_multiclass'] = multiclass_metrics

                    print(f"  Accuracy:      {multiclass_metrics['accuracy']:.4f}")
                    print(f"  F1 (macro):    {multiclass_metrics['f1_score_macro']:.4f}")
                    print(f"  F1 (weighted): {multiclass_metrics['f1_score_weighted']:.4f}")

            except Exception as e:
                print(f"  ⚠️  Multiclass evaluation failed: {e}")

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

    def predict_single(
        self, 
        claim_data: Dict,
        generate_explanations: bool = True
    ) -> Dict:
        """Predict single claim"""
        df = pd.DataFrame([claim_data])
        result = self.predict(
            df, 
            evaluate=False,
            generate_explanations=generate_explanations
        )

        return result['predictions'].iloc[0].to_dict()