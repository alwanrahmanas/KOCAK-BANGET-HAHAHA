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

# SHAP (dengan error handling)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  Warning: SHAP not installed. Install with: pip install shap")

# Import internal modules
from models.threshold_optimizer import ThresholdOptimizer
from models.shap_manager import SHAPExplainerManager

class BPJSFraudDetectionPipeline:
    """
    🔧 FIXED: Two-Stage ML Pipeline with robust training
    """

    def __init__(self, random_state: int = 42, enable_shap: bool = True):
        self.random_state = random_state
        self.enable_shap = enable_shap and SHAP_AVAILABLE
        self.label_encoders = {}
        self.scaler = StandardScaler()

        self.optimal_threshold_stage1 = 0.5
        self.optimal_threshold_stage2 = 0.5

        # SHAP manager
        self.shap_manager = SHAPExplainerManager() if self.enable_shap else None

        # Models
        self.binary_models = {
            'XGBoost': XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, 
                random_state=random_state, eval_metric='logloss',
                use_label_encoder=False
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                random_state=random_state, verbose=-1
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=12,
                min_samples_split=10, min_samples_leaf=4,
                random_state=random_state, n_jobs=-1
            )
        }

        self.multiclass_models = {
            'XGBoost': XGBClassifier(
                n_estimators=200, max_depth=7, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, 
                random_state=random_state, eval_metric='mlogloss',
                objective='multi:softmax', use_label_encoder=False
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=200, max_depth=7, learning_rate=0.1,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                random_state=random_state, verbose=-1, objective='multiclass'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=15,
                min_samples_split=8, min_samples_leaf=3,
                random_state=random_state, n_jobs=-1
            )
        }

        self.binary_ensemble = None
        self.multiclass_ensemble = None

        # 🔧 FIXED: Feature lists (matching your dataset)
        self.numerical_features = [
            'age', 'lama_dirawat', 'billed_amount', 'paid_amount',
            'drug_cost', 'procedure_cost', 'visit_count_30d',
            'tarif_inacbg', 'selisih_klaim', 'time_diff_prev_claim',
            'rolling_avg_cost_30d', 'provider_monthly_claims',
            'nik_hash_reuse_count', 'clinical_pathway_deviation_score',
            'claim_ratio', 'drug_ratio', 'procedure_ratio',
            'provider_claim_share', 'claim_month', 'claim_quarter',
            # Add history features
            'total_klaim_5x', 'rerata_billed_5x', 'std_claim_ratio_5x',
            'rerata_lama_dirawat_5x', 'total_rs_unique_visited_5x',
            'total_diagnosis_unique_5x', 'obat_match_score',
            # Add red flags
            'phantom_suspect_score', 'phantom_node_flag'
        ]

        self.categorical_features = [
            'sex', 'faskes_level', 'jenis_pelayanan', 'room_class'
        ]

        self.boolean_features = [
            'kapitasi_flag', 'referral_flag', 'referral_to_same_facility',
            'visit_suspicious_flag', 'obat_mismatch_flag', 
            'billing_spike_flag', 'high_deviation_flag'
        ]

        self.leakage_features = [
            'fraud_type', 'severity', 'evidence_type', 'graph_pattern_id'
        ]
        self.enable_shap = enable_shap and SHAP_AVAILABLE  # ✅ Sekarang SHAP_AVAILABLE sudah didefinisikan
        
        if enable_shap and not SHAP_AVAILABLE:
            print("⚠️  SHAP requested but not available. Continuing without SHAP.")

    def prepare_data(
        self, 
        df: pd.DataFrame, 
        stage: str = 'binary',
        for_training: bool = True
    ) -> Tuple:
        """
        🔧 FIXED: Prepare data with consistent feature handling
        
        Key fixes:
        1. Better missing feature handling
        2. Consistent encoding across train/inference
        3. Proper NaN handling
        4. Feature validation
        """
        df = df.copy()
        
        # Drop leakage features
        df_clean = df.drop(columns=self.leakage_features, errors='ignore')

        # Encode categorical
        for col in self.categorical_features:
            if col in df_clean.columns:
                if col not in self.label_encoders:
                    if not for_training:
                        raise ValueError(
                            f"Label encoder for '{col}' not found. "
                            "Load trained model first."
                        )
                    # Fit encoder
                    self.label_encoders[col] = LabelEncoder()
                    df_clean[col] = self.label_encoders[col].fit_transform(
                        df_clean[col].astype(str)
                    )
                else:
                    # Transform with handling for unseen categories
                    le = self.label_encoders[col]
                    df_clean[col] = df_clean[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        # Convert boolean
        for col in self.boolean_features:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(int)

        # Select features
        feature_cols = (
            self.numerical_features + 
            self.categorical_features + 
            self.boolean_features
        )
        
        # Check which features are available
        available_features = [col for col in feature_cols if col in df_clean.columns]
        missing_features = [col for col in feature_cols if col not in df_clean.columns]

        if missing_features:
            print(f"  ⚠️  Missing {len(missing_features)} features:")
            print(f"     {missing_features[:10]}...")
            
            if for_training:
                print(f"  ℹ️  Will exclude missing features from training")
            else:
                print(f"  ⚠️  Some expected features are missing - predictions may be affected")

        X = df_clean[available_features].copy()
        
        # Handle NaN values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Fill remaining NaN with 0
        X = X.fillna(0)

        # Target
        if for_training:
            if stage == 'binary':
                if 'fraud_flag' not in df.columns:
                    raise ValueError("Training requires 'fraud_flag' column")
                y = df['fraud_flag'].values

            elif stage == 'multiclass':
                if 'fraud_flag' not in df.columns or 'fraud_type' not in df.columns:
                    raise ValueError("Training requires 'fraud_flag' and 'fraud_type'")

                fraud_mask = df['fraud_flag'] == 1
                X = X[fraud_mask]

                if 'fraud_type_encoder' not in self.label_encoders:
                    self.label_encoders['fraud_type_encoder'] = LabelEncoder()
                    y = self.label_encoders['fraud_type_encoder'].fit_transform(
                        df.loc[fraud_mask, 'fraud_type']
                    )
                else:
                    y = self.label_encoders['fraud_type_encoder'].transform(
                        df.loc[fraud_mask, 'fraud_type']
                    )
        else:
            y = None

        return X, y, available_features

    def train_stage1_binary(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        tune_threshold: bool = True, 
        threshold_metric: str = 'f1',
        create_explainer: bool = True
    ) -> Dict:
        """
        🔧 FIXED: Train Stage 1 with robust error handling
        """
        print("\n" + "="*80)
        print("STAGE 1: BINARY FRAUD DETECTION")
        print("="*80)

        if 'fraud_flag' not in df.columns:
            raise ValueError("Training requires 'fraud_flag' column")

        # Prepare data
        X, y, features = self.prepare_data(df, stage='binary', for_training=True)

        print(f"\n📊 Dataset:")
        print(f"   Features: {len(features)}")
        print(f"   Samples: {len(X)}")
        print(f"   Fraud rate: {y.mean():.2%}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        print(X_train).columns
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

        results = {
            'models': {}, 'predictions': {}, 'metrics': {},
            'X_test': X_test_scaled, 'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'features': features, 'threshold_analysis': {}
        }

        print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

        # Train individual models
        for name, model in self.binary_models.items():
            print(f"\n[{name}] Training...")
            
            try:
                model.fit(X_train_scaled, y_train)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

                if tune_threshold:
                    optimal_threshold, best_score, df_threshold_metrics = \
                        ThresholdOptimizer.find_optimal_threshold(
                            y_test, y_pred_proba, metric=threshold_metric
                        )
                    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
                    
                    results['threshold_analysis'][name] = {
                        'optimal_threshold': optimal_threshold,
                        'best_score': best_score,
                        'metrics_df': df_threshold_metrics
                    }
                    print(f"  ✅ Optimal Threshold: {optimal_threshold:.3f}")
                else:
                    y_pred_optimal = model.predict(X_test_scaled)
                    optimal_threshold = 0.5

                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred_optimal),
                    'precision': precision_score(y_test, y_pred_optimal, zero_division=0),
                    'recall': recall_score(y_test, y_pred_optimal, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred_optimal, zero_division=0),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'confusion_matrix': confusion_matrix(y_test, y_pred_optimal),
                    'optimal_threshold': optimal_threshold
                }

                results['models'][name] = model
                results['predictions'][name] = {
                    'y_pred': y_pred_optimal, 
                    'y_pred_proba': y_pred_proba,
                    'threshold': optimal_threshold
                }
                results['metrics'][name] = metrics

                print(f"  Acc: {metrics['accuracy']:.4f} | "
                      f"F1: {metrics['f1_score']:.4f} | "
                      f"AUC: {metrics['roc_auc']:.4f}")

            except Exception as e:
                print(f"  ❌ Training failed: {e}")

        # Train ensemble
        print(f"\n[Ensemble] Training...")
        
        try:
            self.binary_ensemble = VotingClassifier(
                estimators=[(n, m) for n, m in self.binary_models.items()], 
                voting='soft'
            )
            self.binary_ensemble.fit(X_train_scaled, y_train)
            y_pred_proba_ensemble = self.binary_ensemble.predict_proba(X_test_scaled)[:, 1]

            if tune_threshold:
                optimal_threshold_ensemble, best_score_ensemble, df_threshold_metrics_ensemble = \
                    ThresholdOptimizer.find_optimal_threshold(
                        y_test, y_pred_proba_ensemble, metric=threshold_metric
                    )
                y_pred_ensemble = (y_pred_proba_ensemble >= optimal_threshold_ensemble).astype(int)
                self.optimal_threshold_stage1 = optimal_threshold_ensemble
                
                results['threshold_analysis']['Ensemble'] = {
                    'optimal_threshold': optimal_threshold_ensemble,
                    'best_score': best_score_ensemble,
                    'metrics_df': df_threshold_metrics_ensemble
                }
                print(f"  ✅ Optimal Threshold: {optimal_threshold_ensemble:.3f}")
            else:
                y_pred_ensemble = self.binary_ensemble.predict(X_test_scaled)
                optimal_threshold_ensemble = 0.5

            metrics_ensemble = {
                'accuracy': accuracy_score(y_test, y_pred_ensemble),
                'precision': precision_score(y_test, y_pred_ensemble, zero_division=0),
                'recall': recall_score(y_test, y_pred_ensemble, zero_division=0),
                'f1_score': f1_score(y_test, y_pred_ensemble, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba_ensemble),
                'confusion_matrix': confusion_matrix(y_test, y_pred_ensemble),
                'optimal_threshold': optimal_threshold_ensemble
            }

            results['models']['Ensemble'] = self.binary_ensemble
            results['predictions']['Ensemble'] = {
                'y_pred': y_pred_ensemble,
                'y_pred_proba': y_pred_proba_ensemble,
                'threshold': optimal_threshold_ensemble
            }
            results['metrics']['Ensemble'] = metrics_ensemble

            print(f"  Acc: {metrics_ensemble['accuracy']:.4f} | "
                  f"F1: {metrics_ensemble['f1_score']:.4f} | "
                  f"AUC: {metrics_ensemble['roc_auc']:.4f}")

        except Exception as e:
            print(f"  ❌ Ensemble training failed: {e}")

        # Create SHAP explainer
        if create_explainer and self.enable_shap and self.shap_manager:
            print(f"\n[SHAP] Creating explainer...")
            
            try:
                self.shap_manager.compute_feature_stats(X_train_scaled, features)
                
                # Use LightGBM as base model (faster than ensemble)
                base_model = self.binary_models['LightGBM']
                explainer = self.shap_manager.create_explainer(
                    base_model,
                    X_train_scaled,
                    feature_names=features,
                    max_samples=100,
                    model_type='tree'
                )
                
                results['shap_explainer'] = explainer
                
                if explainer:
                    print(f"  ✅ SHAP explainer ready")
                else:
                    print(f"  ⚠️  SHAP explainer creation failed - using fallback")

            except Exception as e:
                print(f"  ⚠️  SHAP setup failed: {e}")
                results['shap_explainer'] = None

        return results

    def train_stage2_multiclass(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        create_explainer: bool = False
    ) -> Dict:
        """Train Stage 2: Fraud Type Classification"""
        print("\n" + "="*80)
        print("STAGE 2: FRAUD TYPE CLASSIFICATION")
        print("="*80)

        if 'fraud_flag' not in df.columns or 'fraud_type' not in df.columns:
            raise ValueError("Training requires 'fraud_flag' and 'fraud_type'")

        X, y, features = self.prepare_data(df, stage='multiclass', for_training=True)
        unique_classes = np.unique(y)
        
        print(f"\nFraud types: {len(unique_classes)}")

        if len(unique_classes) < 2:
            print("⚠️  Less than 2 fraud types - skipping Stage 2")
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )

        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

        results = {
            'models': {}, 'predictions': {}, 'metrics': {},
            'X_test': X_test_scaled, 'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'features': features,
            'fraud_types': self.label_encoders['fraud_type_encoder'].classes_
        }

        print(f"Train: {len(X_train)} | Test: {len(X_test)}")

        # Train models
        for name, model in self.multiclass_models.items():
            print(f"\n[{name}] Training...")
            
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)

                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }

                results['models'][name] = model
                results['predictions'][name] = {
                    'y_pred': y_pred, 
                    'y_pred_proba': y_pred_proba
                }
                results['metrics'][name] = metrics

                print(f"  Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

            except Exception as e:
                print(f"  ❌ Training failed: {e}")

        # Train ensemble
        print(f"\n[Ensemble] Training...")
        
        try:
            self.multiclass_ensemble = VotingClassifier(
                estimators=[(n, m) for n, m in self.multiclass_models.items()],
                voting='soft'
            )
            self.multiclass_ensemble.fit(X_train_scaled, y_train)
            y_pred_ensemble = self.multiclass_ensemble.predict(X_test_scaled)
            y_pred_proba_ensemble = self.multiclass_ensemble.predict_proba(X_test_scaled)

            metrics_ensemble = {
                'accuracy': accuracy_score(y_test, y_pred_ensemble),
                'precision': precision_score(y_test, y_pred_ensemble, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_ensemble, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred_ensemble, average='weighted', zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred_ensemble)
            }

            results['models']['Ensemble'] = self.multiclass_ensemble
            results['predictions']['Ensemble'] = {
                'y_pred': y_pred_ensemble,
                'y_pred_proba': y_pred_proba_ensemble
            }
            results['metrics']['Ensemble'] = metrics_ensemble

            print(f"  Acc: {metrics_ensemble['accuracy']:.4f} | "
                  f"F1: {metrics_ensemble['f1_score']:.4f}")

        except Exception as e:
            print(f"  ❌ Ensemble training failed: {e}")

        return results
