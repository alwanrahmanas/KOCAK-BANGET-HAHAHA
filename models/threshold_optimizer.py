"""
Threshold Optimizer
===================
Optimizes classification thresholds for better performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score


class ThresholdOptimizer:
    """🔧 FIXED: Robust threshold optimization"""

    @staticmethod
    def find_optimal_threshold(
        y_true, 
        y_pred_proba, 
        metric='f1', 
        beta=1.0,
        min_threshold=0.1,
        max_threshold=0.9,
        step=0.01
    ):
        """
        Find optimal threshold with better error handling
        """
        # Validate inputs
        if len(y_true) == 0 or len(y_pred_proba) == 0:
            print("  ⚠️  Empty inputs - using default threshold 0.5")
            return 0.5, 0.0, pd.DataFrame()

        if len(np.unique(y_true)) < 2:
            print("  ⚠️  Only one class in y_true - using default threshold 0.5")
            return 0.5, 0.0, pd.DataFrame()

        try:
            thresholds = np.arange(min_threshold, max_threshold, step)
            scores = []
            metrics_list = []

            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)

                # Handle edge case: all predictions same class
                if len(np.unique(y_pred)) < 2:
                    continue

                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                if metric == 'f1':
                    score = f1
                elif metric == 'f2':
                    score = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
                elif metric == 'f0.5':
                    score = (1.25 * precision * recall) / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0
                elif metric == 'balanced':
                    score = (precision + recall) / 2
                else:
                    score = f1

                scores.append(score)
                metrics_list.append({
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'score': score
                })

            if len(scores) == 0:
                print("  ⚠️  No valid thresholds found - using default 0.5")
                return 0.5, 0.0, pd.DataFrame()

            best_idx = np.argmax(scores)
            optimal_threshold = thresholds[best_idx]
            best_score = scores[best_idx]

            return optimal_threshold, best_score, pd.DataFrame(metrics_list)

        except Exception as e:
            print(f"  ❌ Threshold optimization failed: {e}")
            print(f"     Using default threshold 0.5")
            return 0.5, 0.0, pd.DataFrame()

    @staticmethod
    def plot_threshold_analysis(df_metrics, optimal_threshold, save_path=None):
        """Plot threshold analysis"""
        if df_metrics.empty:
            print("  ⚠️  No metrics to plot")
            return

        try:
            import matplotlib
            matplotlib.use('Agg')  # ✅ Non-interactive backend untuk CLI
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # Plot 1: Metrics vs threshold
            axes[0].plot(df_metrics['threshold'], df_metrics['precision'], 
                        label='Precision', linewidth=2)
            axes[0].plot(df_metrics['threshold'], df_metrics['recall'], 
                        label='Recall', linewidth=2)
            axes[0].plot(df_metrics['threshold'], df_metrics['f1_score'], 
                        label='F1-Score', linewidth=2, linestyle='--')
            axes[0].axvline(optimal_threshold, color='red', linestyle=':', 
                        label=f'Optimal = {optimal_threshold:.3f}')
            axes[0].set_xlabel('Threshold')
            axes[0].set_ylabel('Score')
            axes[0].set_title('Metrics vs Classification Threshold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            # Plot 2: Precision-Recall curve
            axes[1].plot(df_metrics['recall'], df_metrics['precision'], linewidth=2)
            
            # Find optimal point
            optimal_row = df_metrics[df_metrics['threshold'] == optimal_threshold].iloc[0]
            axes[1].scatter(optimal_row['recall'], optimal_row['precision'],
                        color='red', s=100, zorder=5, 
                        label=f'Optimal (t={optimal_threshold:.3f})')
            axes[1].set_xlabel('Recall')
            axes[1].set_ylabel('Precision')
            axes[1].set_title('Precision-Recall Curve')
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  ✅ Threshold plot saved: {save_path}")
            
            plt.close(fig)  # ✅ Close figure tanpa show()
            
        except Exception as e:
            print(f"  ⚠️  Plotting failed: {e}")

