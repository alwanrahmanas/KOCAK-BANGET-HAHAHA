"""
SHAP Explainer Manager
======================
Manages SHAP explainers with robust error handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    import cloudpickle
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  Warning: SHAP not installed. Install with: pip install shap cloudpickle")


class SHAPExplainerManager:
    """
    🔧 FIXED: Robust SHAP explainer with proper error handling
    """

    def __init__(self):
        self.binary_explainer = None
        self.multiclass_explainer = None
        self.feature_stats = {}
        self.is_ready = False

    def create_explainer(
        self,
        model,
        X_background: pd.DataFrame,
        feature_names: List[str] = None,
        max_samples: int = 100,
        model_type: str = "tree"
    ):
        """
        Create SHAP explainer with robust error handling
        """
        if not SHAP_AVAILABLE:
            print("  ⚠️  SHAP not available - using fallback explanations")
            return None

        try:
            # Normalize background data
            if isinstance(X_background, pd.DataFrame):
                df_bg = X_background.copy()
            else:
                df_bg = pd.DataFrame(X_background, columns=feature_names)

            # Sample background if too large
            if len(df_bg) > max_samples:
                df_bg = df_bg.sample(n=max_samples, random_state=42)
                print(f"  📊 Sampled {max_samples} background samples for SHAP")

            # Validate background data
            if df_bg.empty:
                print("  ❌ Background data is empty - cannot create explainer")
                return None

            # Check for NaN/inf values
            if df_bg.isnull().any().any() or np.isinf(df_bg).any().any():
                print("  ⚠️  Background data contains NaN/inf - cleaning...")
                df_bg = df_bg.fillna(df_bg.median())
                df_bg = df_bg.replace([np.inf, -np.inf], df_bg.max())

            print(f"  🔧 Creating SHAP TreeExplainer...")
            print(f"     Background shape: {df_bg.shape}")
            print(f"     Model type: {type(model).__name__}")

            # Create explainer with SAFE settings
            explainer = shap.TreeExplainer(
                model,
                data=df_bg,
                feature_names=feature_names,
                feature_perturbation='interventional',  # ✅ SAFE mode
                model_output='raw'  # Use raw model output
            )

            self.binary_explainer = explainer
            self.is_ready = True
            
            print(f"  ✅ SHAP explainer created successfully")
            return explainer

        except Exception as e:
            print(f"  ❌ SHAP explainer creation failed: {e}")
            print(f"     Falling back to z-score explanations")
            self.is_ready = False
            return None

    def compute_feature_stats(self, X_train: pd.DataFrame, feature_names: List[str]):
        """Compute feature statistics for reference"""
        try:
            self.feature_stats = {
                'mean': X_train.mean().to_dict(),
                'std': X_train.std().to_dict(),
                'median': X_train.median().to_dict(),
                'q25': X_train.quantile(0.25).to_dict(),
                'q75': X_train.quantile(0.75).to_dict(),
                'min': X_train.min().to_dict(),
                'max': X_train.max().to_dict()
            }
            print(f"  ✓ Feature statistics computed for {len(feature_names)} features")
        except Exception as e:
            print(f"  ⚠️  Feature stats computation failed: {e}")
            self.feature_stats = {}

    def get_local_explanation(
        self, 
        explainer,
        X_instance: np.ndarray,
        feature_names: List[str],
        X_original: pd.Series = None,
        top_k: int = 5
    ) -> Dict:
        """
        Generate local explanation with robust error handling
        """
        if explainer is None:
            return self._fallback_explanation(X_instance, feature_names, X_original, top_k)

        try:
            # Ensure proper shape
            if len(X_instance.shape) == 1:
                X_instance = X_instance.reshape(1, -1)

            # Get SHAP values
            shap_values = explainer.shap_values(X_instance)

            # Handle different output formats
            if isinstance(shap_values, list):
                shap_vals = shap_values[1]  # Fraud class
            else:
                shap_vals = shap_values

            # Flatten if needed
            if len(shap_vals.shape) > 1:
                shap_vals = shap_vals.flatten()

            # Get top features
            abs_shap = np.abs(shap_vals)
            top_indices = np.argsort(abs_shap)[-top_k:][::-1]

            top_features = []
            for idx in top_indices:
                feature_name = feature_names[idx]
                shap_value = float(shap_vals[idx])
                
                # Get original value if available
                if X_original is not None:
                    feature_value = float(X_original.iloc[idx])
                else:
                    feature_value = float(X_instance[0, idx])

                direction = "↑" if shap_value > 0 else "↓"
                impact = self._classify_impact(abs_shap[idx], abs_shap)

                # Get feature statistics
                feat_stats = self.feature_stats.get('mean', {})
                feat_mean = feat_stats.get(feature_name, 0)
                feat_std = self.feature_stats.get('std', {}).get(feature_name, 1)

                z_score = (feature_value - feat_mean) / feat_std if feat_std > 0 else 0

                top_features.append({
                    'feature': feature_name,
                    'value': feature_value,
                    'shap_value': shap_value,
                    'direction': direction,
                    'impact': impact,
                    'z_score': float(z_score),
                    'vs_mean': f"{feature_value:.2f} vs avg {feat_mean:.2f}"
                })

            summary = ", ".join([f"{f['feature']}{f['direction']}" for f in top_features[:3]])

            return {
                'top_features': top_features,
                'summary': summary,
                'explanation_type': 'shap'
            }

        except Exception as e:
            print(f"  ⚠️  SHAP explanation failed: {e}")
            return self._fallback_explanation(X_instance, feature_names, X_original, top_k)

    def _classify_impact(self, abs_shap_value: float, all_abs_shap: np.ndarray) -> str:
        """Classify SHAP impact strength"""
        try:
            percentile = (abs_shap_value >= all_abs_shap).mean() * 100
            if percentile >= 90:
                return "very_strong"
            elif percentile >= 75:
                return "strong"
            elif percentile >= 50:
                return "moderate"
            else:
                return "weak"
        except:
            return "unknown"

    def _fallback_explanation(
        self, 
        X_instance: np.ndarray, 
        feature_names: List[str],
        X_original: pd.Series = None,
        top_k: int = 5
    ) -> Dict:
        """Fallback to z-score based explanation"""
        explanations = []

        for idx, feat_name in enumerate(feature_names):
            if X_original is not None:
                feat_val = float(X_original.iloc[idx])
            else:
                feat_val = float(X_instance[0, idx]) if len(X_instance.shape) > 1 else float(X_instance[idx])

            feat_mean = self.feature_stats.get('mean', {}).get(feat_name, 0)
            feat_std = self.feature_stats.get('std', {}).get(feat_name, 1)

            z_score = (feat_val - feat_mean) / feat_std if feat_std > 0 else 0

            explanations.append({
                'feature': feat_name,
                'value': feat_val,
                'z_score': float(z_score),
                'direction': "↑" if z_score > 0 else "↓",
                'impact': 'strong' if abs(z_score) > 2 else 'moderate' if abs(z_score) > 1 else 'weak'
            })

        explanations.sort(key=lambda x: abs(x['z_score']), reverse=True)
        top_features = explanations[:top_k]

        summary = ", ".join([f"{f['feature']}{f['direction']}" for f in top_features[:3]])

        return {
            'top_features': top_features,
            'summary': summary,
            'explanation_type': 'zscore_fallback'
        }
