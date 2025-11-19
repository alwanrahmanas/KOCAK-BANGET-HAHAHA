import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')
# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

class SHAPExplainerManager:
    """
    Manages SHAP explainers for local explanation generation
    Provides feature-level contributions for RAG integration
    """

    def __init__(self):
        self.binary_explainer = None
        self.multiclass_explainer = None
        self.feature_stats = {}
        self.is_ready = False
        self.shap_manager = None


    def create_explainer(
        self,
        model,
        X_background,
        feature_names=None,
        max_samples=100,
        model_type="auto",
    ):
        # pastikan background dalam bentuk DataFrame (untuk sampling)
        if isinstance(X_background, pd.DataFrame):
            df_bg = X_background
        else:
            df_bg = pd.DataFrame(X_background, columns=feature_names)

        if max_samples is not None and len(df_bg) > max_samples:
            X_bg_sample = df_bg.sample(n=max_samples, random_state=42)
        else:
            X_bg_sample = df_bg

        # Pilih jenis explainer
        # Untuk LightGBM/XGBoost/RandomForest: pakai TreeExplainer
        if model_type in ("tree", "binary"):
            self.explainer = shap.TreeExplainer(model)
        elif model_type == "kernel":
            # kalau benar-benar butuh KernelExplainer untuk model non-tree
            self.explainer = shap.KernelExplainer(
                model.predict_proba, X_bg_sample.to_numpy()
            )
        else:  # "auto"
            # coba dulu TreeExplainer, fallback ke KernelExplainer kalau gagal
            try:
                self.explainer = shap.TreeExplainer(model)
            except Exception:
                self.explainer = shap.KernelExplainer(
                    model.predict_proba, X_bg_sample.to_numpy()
                )

        return self.explainer

    def compute_feature_stats(self, X_train, feature_names):
        # pastikan jadi DataFrame untuk mean/std per fitur
        if isinstance(X_train, pd.DataFrame):
            df = X_train
        else:
            df = pd.DataFrame(X_train, columns=feature_names)

        self.feature_stats = {
            "mean": df.mean().to_dict(),
            "std": df.std().to_dict(),
            "min": df.min().to_dict(),
            "max": df.max().to_dict(),
        }

    def get_local_explanation(self, model, explainer, X_instance: np.ndarray,
                             feature_names: List[str], X_original: pd.Series,
                             class_idx: int = 1, top_k: int = 5) -> Dict:
        """
        Generate local explanation for a single instance

        Parameters:
        -----------
        model : trained model
        explainer : SHAP explainer
        X_instance : scaled feature vector (1D array)
        feature_names : list of feature names
        X_original : original unscaled values (for display)
        class_idx : class index for multiclass (1 for fraud in binary)
        top_k : number of top features to return

        Returns:
        --------
        Dict with local explanation details
        """
        if explainer is None:
            return self._fallback_explanation(X_instance, feature_names, X_original, top_k)

        try:
            # Get SHAP values
            # shaped = explainer(X_instance.reshape(1, -1))
            shap_values = explainer.shap_values(X_instance.reshape(1, -1))

            # Handle different SHAP output formats
            if isinstance(shap_values, list):  # Multiclass or binary with list output
                shap_vals = shap_values[class_idx]
            else:  # Binary with single array
                shap_vals = shap_values

            # Flatten if needed
            if len(shap_vals.shape) > 1:
                shap_vals = shap_vals.flatten()

            # Get top features by absolute SHAP value
            abs_shap = np.abs(shap_vals)
            top_indices = np.argsort(abs_shap)[-top_k:][::-1]

            top_features = []
            for idx in top_indices:
                feature_name = feature_names[idx]
                shap_value = float(shap_vals[idx])
                feature_value = float(X_original.iloc[idx])

                # Determine impact direction and magnitude
                direction = "↑" if shap_value > 0 else "↓"
                impact_strength = self._classify_impact(abs_shap[idx], abs_shap)

                # Get feature statistics for context
                feat_mean = self.feature_stats.get('mean', {}).get(feature_name, 0)
                feat_std = self.feature_stats.get('std', {}).get(feature_name, 1)

                # Calculate z-score
                z_score = (feature_value - feat_mean) / feat_std if feat_std > 0 else 0

                top_features.append({
                    'feature': feature_name,
                    'value': feature_value,
                    'shap_value': shap_value,
                    'direction': direction,
                    'impact': impact_strength,
                    'z_score': float(z_score),
                    'vs_mean': f"{feature_value:.2f} vs avg {feat_mean:.2f}"
                })

            # Create summary text
            summary_text = ", ".join([
                f"{f['feature']}{f['direction']}" for f in top_features[:3]
            ])

            return {
                'top_features': top_features,
                'summary': summary_text,
                'explanation_type': 'shap'
            }

        except Exception as e:
            print(f"  ⚠️  SHAP explanation failed: {e}")
            return self._fallback_explanation(X_instance, feature_names, X_original, top_k)

    def _classify_impact(self, abs_shap_value: float, all_abs_shap: np.ndarray) -> str:
        """Classify SHAP impact as strong/moderate/weak"""
        percentile = (abs_shap_value >= all_abs_shap).mean() * 100

        if percentile >= 90:
            return "very_strong"
        elif percentile >= 75:
            return "strong"
        elif percentile >= 50:
            return "moderate"
        else:
            return "weak"

    def _fallback_explanation(self, X_instance: np.ndarray, feature_names: List[str],
                             X_original: pd.Series, top_k: int = 5) -> Dict:
        """
        Fallback explanation when SHAP is not available
        Uses z-score based feature importance
        """
        explanations = []

        for idx, feat_name in enumerate(feature_names):
            feat_val = float(X_original.iloc[idx])
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

        # Sort by absolute z-score
        explanations.sort(key=lambda x: abs(x['z_score']), reverse=True)
        top_features = explanations[:top_k]

        summary_text = ", ".join([
            f"{f['feature']}{f['direction']}" for f in top_features[:3]
        ])

        return {
            'top_features': top_features,
            'summary': summary_text,
            'explanation_type': 'zscore_fallback'
        }


