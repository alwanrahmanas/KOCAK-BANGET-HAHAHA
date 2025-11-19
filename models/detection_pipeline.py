"""
BPJS Fraud Detection Pipeline - Training component
"""
from datetime import datetime
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
from utils.logger import get_logger

logger = get_logger(__name__)

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, 
    roc_curve, classification_report
)
import seaborn as sns   # untuk heatmap multiclass, tambahkan ke requirements jika perlu

# Setelah optimal_threshold sudah didapat (paling bawah sebelum return)
import joblib
import os

def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score

def save_binary_report_to_file(y_true, y_prob, threshold=0.5, output_dir=None, prefix="binary"):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    acc = accuracy_score(y_true, y_pred)

    report_text = (
        f"=== Binary Classification Report ===\n"
        f"Threshold: {threshold:.4f}\n"
        f"Accuracy: {acc:.4f}\n"
        f"Confusion Matrix:\n{cm}\n\n"
        f"Classification Report:\n{report}\n"
    )
    print(report_text)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{prefix}_classification_report.txt")
        with open(filepath, "w") as f:
            f.write(report_text)
        print(f"[Saved] Binary classification report to: {filepath}")

        # Save ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        roc_path = os.path.join(output_dir, f"{prefix}_roc_curve.png")
        plt.savefig(roc_path)
        plt.close()
        print(f"[Saved] ROC curve to: {roc_path}")

def save_multiclass_report_to_file(y_true, y_pred, classes, output_dir=None, prefix="multiclass"):
    acc = accuracy_score(y_true, y_pred)
    encoded_classes = np.arange(len(classes))
    cm = confusion_matrix(y_true, y_pred, labels=encoded_classes)

    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    
    report_text = (
        f"=== Multiclass Classification Report ===\n"
        f"Accuracy: {acc:.4f}\n"
        f"Confusion Matrix:\n{cm}\n\n"
        f"Classification Report:\n{report}\n"
    )
    print(report_text)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{prefix}_classification_report.txt")
        with open(filepath, "w") as f:
            f.write(report_text)
        print(f"[Saved] Multiclass classification report to: {filepath}")

        # Save confusion matrix heatmap
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        cm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"[Saved] Confusion matrix heatmap to: {cm_path}")

    


def report_multiclass_performance(y_true, y_pred, classes=None, output_dir=None, prefix="multiclass"):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    report = classification_report(y_true, y_pred, labels=classes, digits=3, zero_division=0)
    print(f"\n=== Multiclass Classification Report ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{report}")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{prefix}_classification_report.txt"), 'w') as f:
            f.write(f"Accuracy : {acc:.4f}\n\n")
            f.write(str(cm) + "\n\n")
            f.write(report)
        # Heatmap
        if classes is not None:
            plt.figure(figsize=(10,8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_confusion_matrix.png"))
            plt.close()


class BPJSFraudDetectionPipeline:
    """
    Two-stage fraud detection pipeline with SHAP explainability
    """
    
    def __init__(
        self,
        random_state: int = 42,
        enable_shap: bool = True,
        config: Dict[str, Any] = None
    ):
        """
        Initialize detection pipeline
        
        Args:
            random_state: Random seed for reproducibility
            enable_shap: Enable SHAP explainer creation
            config: Configuration dictionary
        """
        self.random_state = random_state
        self.enable_shap = enable_shap
        self.config = config or {}
        
        # Feature definitions
        self.numerical_features = [
            'age', 'lama_dirawat', 'billed_amount', 'paid_amount',
            'drug_cost', 'procedure_cost', 'visit_count_30d', 'tarif_inacbg',
            'selisih_klaim', 'time_diff_prev_claim', 'rolling_avg_cost_30d',
            'provider_monthly_claims', 'nik_hash_reuse_count',
            'clinical_pathway_deviation_score', 'claim_ratio', 'drug_ratio',
            'procedure_ratio', 'provider_claim_share', 'claim_month'
        ]
        
        self.categorical_features = [
            'sex', 'faskes_level', 'jenis_pelayanan', 'room_class'
        ]
        
        self.boolean_features = [
            'kapitasi_flag', 'referral_flag', 'referral_to_same_facility'
        ]
        
        # Leakage features (columns that should not be used for prediction)
        self.leakage_features = [
            'fraud_flag',           # Target: binary classification
            'fraud_type',           # Target: multiclass classification
            'severity',             # Derived from fraud labels
            'evidence_type',        # Derived from fraud labels
            'graph_pattern_id',     # Generated during fraud injection
            'graph_pattern_type'    # Generated during fraud injection
        ]
        
        # Models storage
        self.binary_models = {}
        self.multiclass_models = {}
        self.binary_scaler = None
        self.multiclass_scaler = None
        self.label_encoders = {}
        
        # SHAP components
        self.shap_manager = SHAPExplainerManager() if enable_shap else None
        
        # Threshold optimization
        self.optimal_threshold = 0.5
        self.threshold_metric = 'f1'
        
        # Logging
        logger.info("BPJSFraudDetectionPipeline initialized")
        logger.info(f"  Random state: {random_state}")
        logger.info(f"  SHAP enabled: {enable_shap}")
        logger.info(f"  Features: {len(self.numerical_features + self.categorical_features + self.boolean_features)}")
        logger.info(f"  Leakage features: {len(self.leakage_features)}")

    
    def train_stage1_binary(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        tune_threshold: bool = True,
        threshold_metric: str = 'f1',
        create_explainer: bool = True
    ) -> Dict[str, Any]:
        """
        Train Stage 1: Binary classification (fraud vs benign)
        
        Args:
            df: Training DataFrame
            test_size: Test split ratio
            tune_threshold: Whether to tune classification threshold
            threshold_metric: Metric for threshold optimization
            create_explainer: Whether to create SHAP explainer
        
        Returns:
            Dictionary with training results and SHAP explainer
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 1: BINARY CLASSIFICATION (Fraud vs Benign)")
        logger.info("="*80)
        
        # ========== 1. VALIDATE REQUIRED COLUMNS ==========
        if 'fraud_flag' not in df.columns:
            raise ValueError("Column 'fraud_flag' is required for training")
        
        # ========== 2. PREPARE FEATURES ==========
        X, y = self._prepare_features_binary(df)
        
        # Get feature names (will need for SHAP)
        all_features = (
            self.numerical_features + 
            self.categorical_features + 
            self.boolean_features
        )
        
        # ========== 3. SPLIT DATA ==========
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        logger.info(f"\nDataset split:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Test: {len(X_test)} samples")
        logger.info(f"  Fraud rate (train): {y_train.mean():.2%}")
        logger.info(f"  Fraud rate (test): {y_test.mean():.2%}")
        
        # ========== 4. SCALE FEATURES ==========
        self.binary_scaler = StandardScaler()
        X_train_scaled = self.binary_scaler.fit_transform(X_train)
        X_test_scaled = self.binary_scaler.transform(X_test)
        
        # ========== 5. TRAIN MODELS ==========
        logger.info("\nTraining models...")
        self._train_binary_models(X_train_scaled, y_train)
        
        # ========== 6. EVALUATE MODELS ==========
        results = self._evaluate_binary_models(
            X_train_scaled, X_test_scaled,
            y_train, y_test
        )
        
        # ========== 7. THRESHOLD TUNING ==========
        optimal_threshold = 0.5
        threshold_results = None
        
        if tune_threshold:
            logger.info(f"\n{'='*80}")
            logger.info(f"THRESHOLD OPTIMIZATION (metric: {threshold_metric})")
            logger.info(f"{'='*80}")
            
            self.threshold_metric = threshold_metric
            
            # Get ensemble predictions
            if 'Ensemble' in self.binary_models:
                model_for_threshold = self.binary_models['Ensemble']
            else:
                # Fallback to first available model
                model_for_threshold = list(self.binary_models.values())[0]
            
            y_proba_test = model_for_threshold.predict_proba(X_test_scaled)[:, 1]
            
            # Optimize threshold
            optimizer = ThresholdOptimizer()
            optimal_threshold, threshold_results = optimizer.optimize(
                y_test, y_proba_test, metric=threshold_metric
            )
            
            self.optimal_threshold = optimal_threshold
            
            logger.info(f"  Optimal threshold: {optimal_threshold:.4f}")
            logger.info(f"  Best {threshold_metric}: {threshold_results['best_score']:.4f}")
        
        # report_binary_performance(
        #     y_true=y_test,
        #     y_prob=y_proba_test,
        #     threshold=optimal_threshold,
        #     output_dir=path_to_save_reports,
        #     prefix="binary"
        # )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_binary_report_to_file(
            y_true=y_test,
            y_prob=y_proba_test,
            threshold=optimal_threshold,
            output_dir=f"artifacts/binary_report_{ts}",
            prefix="binary"
        )
        # ========== 8. CREATE SHAP EXPLAINER ==========
        shap_explainer = None
        
        if create_explainer and self.enable_shap:
            logger.info(f"\n{'='*80}")
            logger.info("CREATING SHAP EXPLAINER")
            logger.info(f"{'='*80}")
            
            try:
                import shap
                
                # Select model for SHAP (prefer tree-based)
                if 'XGBoost' in self.binary_models:
                    model_for_shap = self.binary_models['XGBoost']
                    model_name = 'XGBoost'
                elif 'LightGBM' in self.binary_models:
                    model_for_shap = self.binary_models['LightGBM']
                    model_name = 'LightGBM'
                elif 'RandomForest' in self.binary_models:
                    model_for_shap = self.binary_models['RandomForest']
                    model_name = 'RandomForest'
                else:
                    raise ValueError("No tree-based model available for SHAP")
                
                logger.info(f"  Using model: {model_name}")
                
                # Sample background data (for speed)
                background_size = min(100, len(X_train_scaled))
                background_indices = np.random.choice(
                    len(X_train_scaled),
                    size=background_size,
                    replace=False
                )
                X_background = X_train_scaled[background_indices]
                
                logger.info(f"  Background samples: {background_size}")
                logger.info(f"  Features: {len(all_features)}")
                
                # Create TreeExplainer
                shap_explainer = shap.TreeExplainer(
                    model_for_shap,
                    data=X_background,
                    feature_names=all_features,
                    model_output='probability'  # For binary classification
                )
                
                logger.info(f"  ✓ SHAP TreeExplainer created successfully")
                
                # Compute feature statistics (for fallback explanations)
                if self.shap_manager:
                    self.shap_manager.compute_feature_stats(X_train_scaled, all_features)
                    self.shap_manager.binary_explainer = shap_explainer
                    logger.info(f"  ✓ Feature statistics computed")
                
            except ImportError:
                logger.warning("  ⚠️  SHAP library not installed")
                logger.warning("     Install with: pip install shap")
                shap_explainer = None
            except Exception as e:
                logger.warning(f"  ⚠️  Could not create SHAP explainer: {e}")
                logger.warning(f"     Continuing without SHAP support")
                shap_explainer = None
        else:
            if not self.enable_shap:
                logger.info("\nSHAP explainer creation disabled (enable_shap=False)")
            elif not create_explainer:
                logger.info("\nSHAP explainer creation skipped (create_explainer=False)")
        
        # ========== 9. COMPILE RESULTS ==========
        final_results = {
            'metrics': results.get('metrics', {}),
            'full_results': results,
            'optimal_threshold': optimal_threshold,
            'threshold_optimization': threshold_results,
            'shap_explainer': shap_explainer,  # ⭐ CRITICAL: Include explainer
            'X_test': X_test,
            'y_test': y_test,
            'X_test_scaled': X_test_scaled,
            'feature_names': all_features
        }
        
        logger.info(f"\n{'='*80}")
        logger.info("✅ STAGE 1 TRAINING COMPLETED!")
        logger.info(f"{'='*80}")
        logger.info(f"  Models trained: {len(self.binary_models)}")
        logger.info(f"  Optimal threshold: {optimal_threshold:.4f}")
        logger.info(f"  SHAP explainer: {'✓ Created' if shap_explainer else '✗ Not available'}")
        # print(report_binary_performance)
        return final_results

    
    def train_stage2_multiclass(
        self,
        X_multiclass,
        y_multiclass,
        test_size=0.2,
        random_state=42,
        use_stratify=True,
        min_samples_per_class=2,
    ):
        """
        Stage 2: Multiclass classification untuk tipe fraud.
        - X_multiclass: fitur numerik untuk kasus fraud (np.array atau DataFrame)
        - y_multiclass: label string, misalnya 'upcoding_diagnosis', 'phantom_billing', dll.
        """

        logger.info("=" * 80)
        logger.info("STAGE 2: MULTICLASS CLASSIFICATION (Fraud Types)")
        logger.info("=" * 80)
        logger.info("")

        # 1. Statistik distribusi label (string asli)
        unique, counts = np.unique(y_multiclass, return_counts=True)
        total = counts.sum()
        logger.info("Fraud cases: %d", total)
        logger.info("Fraud types distribution:")
        for label, c in zip(unique, counts):
            logger.info("  %s: %d (%.1f%%)", label, c, 100.0 * c / total)

        # 2. Tangani kelas yang terlalu jarang (misal hanya 1 sampel)
        counts_dict = {label: c for label, c in zip(unique, counts)}
        rare_classes = [lbl for lbl, c in counts_dict.items() if c < min_samples_per_class]

        if rare_classes:
            logger.warning(
                "Classes with < %d samples will be removed from Stage 2: %s",
                min_samples_per_class,
                rare_classes,
            )
            mask = ~np.isin(y_multiclass, rare_classes)
            X_multiclass = X_multiclass[mask]
            y_multiclass = y_multiclass[mask]

            unique, counts = np.unique(y_multiclass, return_counts=True)
            total = counts.sum()
            logger.info("After filtering rare classes:")
            logger.info("Fraud cases: %d", total)
            logger.info("Fraud types distribution:")
            for label, c in zip(unique, counts):
                logger.info("  %s: %d (%.1f%%)", label, c, 100.0 * c / total)

        # 3. Encode label string -> integer (0..K-1) untuk XGBoost dan model lain
        # Create encoder
        self.label_encoder_multiclass = LabelEncoder()
        y_encoded = self.label_encoder_multiclass.fit_transform(y_multiclass)

        # ⭐ ADD THIS: Save to label_encoders dict (for persistence)
        self.label_encoders['fraud_type_encoder'] = self.label_encoder_multiclass
        
        logger.info(f"\n✓ Fraud type encoder created and saved")
        logger.info(f"  Number of classes: {len(self.label_encoder_multiclass.classes_)}")
        logger.info(f"  Classes: {list(self.label_encoder_multiclass.classes_)}")


        # 4. Split train/test (opsional stratify jika masih memungkinkan)
        stratify_arg = y_encoded if use_stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X_multiclass,
            y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg,
        )

        # 5. Scaling fitur
        # 5. Scaling fitur
        self.multiclass_scaler = StandardScaler()
        X_train_scaled = self.multiclass_scaler.fit_transform(X_train)
        X_test_scaled = self.multiclass_scaler.transform(X_test)


        # 6. Latih model-model multiclass
        # 6. Latih model-model multiclass
        self._train_multiclass_models(X_train_scaled, y_train)

        # 7. Evaluasi dasar (menggunakan label asli untuk interpretasi)
        results = {'metrics': {}}
        class_names = self.label_encoder_multiclass.classes_

        for name, model in self.multiclass_models.items():
            y_pred = model.predict(X_test_scaled)
            report = classification_report(
                y_test,
                y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
            cm = confusion_matrix(y_test, y_pred)

            # simpan classification_report & confusion_matrix jika mau
            results[name] = {
                "classification_report": report,
                "confusion_matrix": cm,
            }

            # summary metrics (accuracy & F1) ke dalam results['metrics']
            metrics_summary = {
                "accuracy": report["accuracy"],
                "f1_score": report["weighted avg"]["f1-score"],
                "f1_macro": report["macro avg"]["f1-score"],
            }
            results['metrics'][name] = metrics_summary

            logger.info("=== Multiclass model: %s ===", name)
            logger.info(
                "Macro F1: %.4f | Weighted F1: %.4f",
                report["macro avg"]["f1-score"],
                report["weighted avg"]["f1-score"],
            )

        

        # 8. (Opsional) SHAP explainer untuk salah satu model multiclass (mis. XGBoost)
        if "XGBoost" in self.multiclass_models and self.shap_manager is not None:
            model_xgb = self.multiclass_models["XGBoost"]
            feature_names = (
                self.numerical_features +
                self.categorical_features +
                self.boolean_features
            )

            self.shap_manager.create_explainer(
                model=model_xgb,
                X_background=X_train_scaled,
                feature_names=feature_names,
                max_samples=100,
                model_type="tree",
            )
        # In your train_stage2_multiclass method
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir=f"artifacts/multiclass_reports_{ts}"

        for name, model in self.multiclass_models.items():
            y_pred = model.predict(X_test_scaled)
            
            report = classification_report(
                y_test,
                y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
            
            # ========== DEBUG: Print report structure ==========
            print("\n" + "="*60)
            print(f"DEBUG: Classification Report for {name}")
            print("="*60)
            print("Report keys:", report.keys())
            print("Has 'accuracy' key:", 'accuracy' in report)
            if 'accuracy' in report:
                print(f"Accuracy value: {report['accuracy']}")
                print(f"Accuracy type: {type(report['accuracy'])}")
            
            print("\nHas 'weighted avg' key:", 'weighted avg' in report)
            if 'weighted avg' in report:
                print("Weighted avg keys:", report['weighted avg'].keys())
                print(f"F1-score: {report['weighted avg']['f1-score']}")
            
            print("="*60 + "\n")
        

            # output_dir=f"artifacts/binary_reports_{ts}"

        output_dir = f"artifacts/multiclass_reports_{ts}"
        y_pred_ensemble = self.multiclass_models["Ensemble"].predict(X_test_scaled)

        save_multiclass_report_to_file(
            y_true=y_test,
            y_pred=y_pred_ensemble,
            classes=self.label_encoder_multiclass.classes_,
            output_dir=output_dir,
            prefix="multiclass"
        )


        return {
            "metrics": results['metrics'],  # ⭐ Extract metrics to top level
            "models": results,  # Full results for reference
            "label_encoder": self.label_encoder_multiclass,
            # Optional debug info:
            "X_test": X_test,
            "y_test": y_test,
            "X_test_scaled": X_test_scaled
        }


    
    def _prepare_features_binary(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for binary classification"""
        df = df.copy()
        
        # Drop leakage columns
        drop_cols = ['fraud_flag', 'fraud_type', 'severity', 'evidence_type']
        for col in drop_cols:
            if col in df.columns and col != 'fraud_flag':
                df = df.drop(columns=[col])
        
        # Encode categorical features
        for feat in self.categorical_features:
            if feat in df.columns:
                if feat not in self.label_encoders:
                    self.label_encoders[feat] = LabelEncoder()
                    df[feat] = self.label_encoders[feat].fit_transform(df[feat].astype(str))
                else:
                    df[feat] = self.label_encoders[feat].transform(df[feat].astype(str))
        
        # Select features
        all_features = self.numerical_features + self.categorical_features + self.boolean_features
        available_features = [f for f in all_features if f in df.columns]
        
        X = df[available_features].values
        y = df['fraud_flag'].values
        
        return X, y
    
    def _prepare_features_multiclass(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for multiclass classification"""
        df = df.copy()
        
        # Drop leakage columns
        drop_cols = ['fraud_flag', 'fraud_type', 'severity', 'evidence_type']
        for col in drop_cols:
            if col in df.columns and col != 'fraud_type':
                df = df.drop(columns=[col])
        
        # Encode categorical features (use existing encoders from binary stage)
        for feat in self.categorical_features:
            if feat in df.columns:
                if feat in self.label_encoders:
                    df[feat] = self.label_encoders[feat].transform(df[feat].astype(str))
        
        # Select features
        all_features = self.numerical_features + self.categorical_features + self.boolean_features
        available_features = [f for f in all_features if f in df.columns]
        
        X = df[available_features].values
        y = df['fraud_type'].values
        
        return X, y
    
    def _train_binary_models(self, X_train: np.ndarray, y_train: np.ndarray,save_path: str = None):
        """Train binary classification models"""
        # XGBoost
        self.binary_models['XGBoost'] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        self.binary_models['XGBoost'].fit(X_train, y_train)
        logger.info("  ✓ XGBoost trained")
        
        # LightGBM
        self.binary_models['LightGBM'] = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            verbose=-1
        )
        self.binary_models['LightGBM'].fit(X_train, y_train)
        logger.info("  ✓ LightGBM trained")
        
        # Random Forest
        self.binary_models['RandomForest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state
        )
        self.binary_models['RandomForest'].fit(X_train, y_train)
        logger.info("  ✓ RandomForest trained")
        
        # Ensemble (Voting)
        self.binary_models['Ensemble'] = VotingClassifier(
            estimators=[
                ('xgb', self.binary_models['XGBoost']),
                ('lgbm', self.binary_models['LightGBM']),
                ('rf', self.binary_models['RandomForest'])
            ],
            voting='soft'
        )
        self.binary_models['Ensemble'].fit(X_train, y_train)
        logger.info("  ✓ Ensemble trained")
       
    def _train_multiclass_models(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train multiclass classification models"""
        # XGBoost
        self.multiclass_models['XGBoost'] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            eval_metric='mlogloss'
        )
        self.multiclass_models['XGBoost'].fit(X_train, y_train)
        logger.info("  ✓ XGBoost trained")
        
        # LightGBM
        self.multiclass_models['LightGBM'] = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            verbose=-1
        )
        self.multiclass_models['LightGBM'].fit(X_train, y_train)
        logger.info("  ✓ LightGBM trained")
        
        # Random Forest
        self.multiclass_models['RandomForest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state
        )
        self.multiclass_models['RandomForest'].fit(X_train, y_train)
        logger.info("  ✓ RandomForest trained")
        
        # Ensemble
        self.multiclass_models['Ensemble'] = VotingClassifier(
            estimators=[
                ('xgb', self.multiclass_models['XGBoost']),
                ('lgbm', self.multiclass_models['LightGBM']),
                ('rf', self.multiclass_models['RandomForest'])
            ],
            voting='soft'
        )
        self.multiclass_models['Ensemble'].fit(X_train, y_train)
        logger.info("  ✓ Ensemble trained")
    
    def _evaluate_binary_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate binary models"""
        logger.info("\nEvaluation Results:")
        logger.info("-" * 80)
        
        results = {'metrics': {}}
        
        for model_name, model in self.binary_models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_proba),
                'mcc': matthews_corrcoef(y_test, y_pred),
                'kappa': cohen_kappa_score(y_test, y_pred)
            }
            
            results['metrics'][model_name] = metrics
            
            logger.info(f"\n{model_name}:")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return results
    
    def _evaluate_multiclass_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate multiclass models"""
        logger.info("\nEvaluation Results:")
        logger.info("-" * 80)
        
        results = {'metrics': {}}
        
        # Get class names
        if hasattr(self, 'label_encoder_multiclass') and self.label_encoder_multiclass:
            class_names = self.label_encoder_multiclass.classes_
        else:
            class_names = None
        
        for model_name, model in self.multiclass_models.items():
            y_pred = model.predict(X_test)
            
            # Generate classification report
            if class_names is not None:
                report = classification_report(
                    y_test,
                    y_pred,
                    target_names=class_names,
                    output_dict=True,
                    zero_division=0
                )
            else:
                report = classification_report(
                    y_test,
                    y_pred,
                    output_dict=True,
                    zero_division=0
                )
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # ========== SAFE METRIC EXTRACTION ==========
            # Extract accuracy (top-level key in classification_report)
            accuracy = report.get('accuracy', 0.0)
            
            # Extract F1 scores from aggregated metrics
            weighted_avg = report.get('weighted avg', {})
            macro_avg = report.get('macro avg', {})
            
            f1_weighted = weighted_avg.get('f1-score', 0.0)
            f1_macro = macro_avg.get('f1-score', 0.0)
            
            # Store metrics summary
            metrics_summary = {
                'accuracy': float(accuracy),
                'f1_score': float(f1_weighted),     # weighted avg
                'f1_macro': float(f1_macro)         # macro avg
            }
            
            results['metrics'][model_name] = metrics_summary
            
            # Store full report and confusion matrix
            results[model_name] = {
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'metrics_summary': metrics_summary
            }
            
            # Log results
            logger.info(f"\n{model_name}:")
            logger.info(f"  Accuracy:        {accuracy:.4f}")
            logger.info(f"  F1 (Weighted):   {f1_weighted:.4f}")
            logger.info(f"  F1 (Macro):      {f1_macro:.4f}")
        
        return results

    # def _save_feature_config(self, save_dir: Path):
    #     """Save feature configuration"""
    #     print("\n6. Saving feature configuration...")
        
    #     features_config = {
    #         'numerical_features': self.numerical_features,
    #         'categorical_features': self.categorical_features,
    #         'boolean_features': self.boolean_features,
    #         'all_features': self.numerical_features + self.categorical_features + self.boolean_features,
    #         'leakage_features': ['fraud_flag', 'fraud_type', 'severity', 'evidence_type']  # ⭐ ADD THIS
    #     }
        
    #     config_path = save_dir / 'features_config.json'
    #     with open(config_path, 'w') as f:
    #         json.dump(features_config, f, indent=2)
    #     print(f"   ✓ Feature config saved: {config_path}")
