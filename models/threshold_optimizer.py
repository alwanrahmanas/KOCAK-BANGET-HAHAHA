"""
Threshold optimization for binary classification
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class ThresholdOptimizer:
    """
    Optimize classification threshold for different metrics
    """
    
    def __init__(self, min_threshold: float = 0.1, max_threshold: float = 0.9, step: float = 0.01):
        """
        Initialize threshold optimizer
        
        Args:
            min_threshold: Minimum threshold to test
            max_threshold: Maximum threshold to test
            step: Step size for threshold search
        """
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.step = step
        self.thresholds = np.arange(min_threshold, max_threshold + step, step)
    
    def optimize(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, Dict]:
        """
        Find optimal threshold for given metric
        
        Args:
            y_true: True labels (0/1)
            y_proba: Predicted probabilities (0.0-1.0)
            metric: Metric to optimize ('f1', 'f2', 'f0.5', 'balanced')
        
        Returns:
            optimal_threshold: Best threshold value
            results: Dictionary with optimization results
        """
        best_score = 0
        best_threshold = 0.5
        threshold_scores = []
        
        for threshold in self.thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'f2':
                # F2 score (recall weighted 2x)
                beta = 2
                score = ((1 + beta**2) * precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
            elif metric == 'f0.5':
                # F0.5 score (precision weighted 2x)
                beta = 0.5
                score = ((1 + beta**2) * precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
            elif metric == 'balanced':
                # Balanced metric (average of precision and recall)
                score = (precision + recall) / 2
            else:
                raise ValueError(f"Unknown metric: {metric}. Use 'f1', 'f2', 'f0.5', or 'balanced'")
            
            threshold_scores.append({
                'threshold': float(threshold),
                'score': float(score),
                'precision': float(precision),
                'recall': float(recall)
            })
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        results = {
            'optimal_threshold': float(best_threshold),
            'best_score': float(best_score),
            'metric': metric,
            'all_scores': threshold_scores
        }
        
        return best_threshold, results
    
    def plot_threshold_curve(
        self,
        results: Dict,
        save_path: str = None
    ):
        """
        Plot threshold vs metrics curve
        
        Args:
            results: Results from optimize()
            save_path: Path to save plot (optional)
        """
        scores_data = results['all_scores']
        thresholds = [s['threshold'] for s in scores_data]
        scores = [s['score'] for s in scores_data]
        precisions = [s['precision'] for s in scores_data]
        recalls = [s['recall'] for s in scores_data]
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(thresholds, scores, 'b-', label=f'{results["metric"].upper()} Score', linewidth=2)
        plt.plot(thresholds, precisions, 'g--', label='Precision', linewidth=1.5)
        plt.plot(thresholds, recalls, 'r--', label='Recall', linewidth=1.5)
        
        # Mark optimal threshold
        plt.axvline(
            results['optimal_threshold'],
            color='orange',
            linestyle=':',
            linewidth=2,
            label=f'Optimal ({results["optimal_threshold"]:.3f})'
        )
        
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(f'Threshold Optimization - {results["metric"].upper()}', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
