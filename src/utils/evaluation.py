"""Evaluation utilities for fraud detection models."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score, f1_score, precision_score, 
    recall_score, accuracy_score, matthews_corrcoef
)
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation utility class."""
    
    def __init__(self, model, model_name: str = "Model"):
        """Initialize evaluator with model.
        
        Args:
            model: Trained model with predict and predict_proba methods
            model_name: Name of the model for reporting
        """
        self.model = model
        self.model_name = model_name
        
    def evaluate_comprehensive(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Perform comprehensive model evaluation.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_all_metrics(y_test, y_pred, y_pred_proba)
        
        # Add model info
        metrics['model_name'] = self.model_name
        metrics['test_size'] = len(y_test)
        
        return metrics
    
    def _calculate_all_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                              y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate all evaluation metrics."""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Probabilistic metrics
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            roc_auc = 0.5  # Default for edge cases
            
        try:
            pr_auc = average_precision_score(y_true, y_pred_proba)
        except ValueError:
            pr_auc = sum(y_true) / len(y_true)  # Baseline
        
        # Additional metrics
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        else:
            tn = fp = fn = tp = 0
            specificity = npv = fpr = fnr = 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'mcc': mcc,
            'specificity': specificity,
            'npv': npv,
            'fpr': fpr,
            'fnr': fnr,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
    
    def plot_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series, 
                            normalize: bool = False, figsize: Tuple[int, int] = (8, 6)):
        """Plot confusion matrix.
        
        Args:
            X_test: Test features
            y_test: Test labels
            normalize: Whether to normalize the confusion matrix
            figsize: Figure size
        """
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = f'{self.model_name} - Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = f'{self.model_name} - Confusion Matrix'
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        plt.title(title, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, X_test: pd.DataFrame, y_test: pd.Series, 
                      figsize: Tuple[int, int] = (8, 6)):
        """Plot ROC curve.
        
        Args:
            X_test: Test features
            y_test: Test labels
            figsize: Figure size
        """
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'{self.model_name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.model_name} - ROC Curve', fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(self, X_test: pd.DataFrame, y_test: pd.Series, 
                                   figsize: Tuple[int, int] = (8, 6)):
        """Plot Precision-Recall curve.
        
        Args:
            X_test: Test features
            y_test: Test labels
            figsize: Figure size
        """
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        baseline = sum(y_test) / len(y_test)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='darkorange', lw=2, 
                label=f'{self.model_name} (AUC = {pr_auc:.3f})')
        plt.axhline(y=baseline, color='navy', linestyle='--', alpha=0.5, 
                   label=f'Baseline ({baseline:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{self.model_name} - Precision-Recall Curve', fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], top_n: int = 15, 
                               figsize: Tuple[int, int] = (10, 8)):
        """Plot feature importance if available.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to show
            figsize: Figure size
        """
        if not hasattr(self.model, 'feature_importances_'):
            print(f"Model {self.model_name} does not have feature_importances_ attribute")
            return
        
        importance = self.model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(feature_imp_df)), feature_imp_df['importance'])
        plt.yticks(range(len(feature_imp_df)), feature_imp_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'{self.model_name} - Feature Importance', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5, 
                      scoring: str = 'f1') -> Dict[str, float]:
        """Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with cross-validation results
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        return {
            'cv_scores': scores.tolist(),
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_min': scores.min(),
            'cv_max': scores.max()
        }

class BusinessImpactCalculator:
    """Calculate business impact metrics for fraud detection models."""
    
    def __init__(self, avg_transaction_value: float = 100, 
                 investigation_cost: float = 25,
                 fraud_loss_multiplier: float = 1.0):
        """Initialize business impact calculator.
        
        Args:
            avg_transaction_value: Average value of transactions
            investigation_cost: Cost of investigating each flagged transaction
            fraud_loss_multiplier: Multiplier for fraud losses (e.g., chargebacks)
        """
        self.avg_transaction_value = avg_transaction_value
        self.investigation_cost = investigation_cost
        self.fraud_loss_multiplier = fraud_loss_multiplier
    
    def calculate_impact(self, tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
        """Calculate business impact metrics.
        
        Args:
            tp: True positives
            fp: False positives
            tn: True negatives
            fn: False negatives
            
        Returns:
            Dictionary with business impact metrics
        """
        # Financial calculations
        fraud_prevented = tp * self.avg_transaction_value
        fraud_losses = fn * self.avg_transaction_value * self.fraud_loss_multiplier
        investigation_costs = fp * self.investigation_cost
        
        # Net benefit
        net_benefit = fraud_prevented - fraud_losses - investigation_costs
        
        # Operational metrics
        total_transactions = tp + fp + tn + fn
        customer_friction_rate = fp / total_transactions if total_transactions > 0 else 0
        fraud_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # ROI calculation
        total_costs = investigation_costs
        roi = (net_benefit / total_costs) if total_costs > 0 else 0
        
        return {
            'fraud_prevented_value': fraud_prevented,
            'fraud_losses': fraud_losses,
            'investigation_costs': investigation_costs,
            'net_benefit': net_benefit,
            'customer_friction_rate': customer_friction_rate,
            'fraud_detection_rate': fraud_detection_rate,
            'roi': roi,
            'cost_per_transaction': total_costs / total_transactions if total_transactions > 0 else 0
        }

def compare_models(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, 
                  metrics: List[str] = None) -> pd.DataFrame:
    """Compare multiple models on the same test set.
    
    Args:
        models: Dictionary of {model_name: model} pairs
        X_test: Test features
        y_test: Test labels
        metrics: List of metrics to include in comparison
        
    Returns:
        DataFrame with comparison results
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
    
    results = []
    
    for model_name, model in models.items():
        evaluator = ModelEvaluator(model, model_name)
        eval_results = evaluator.evaluate_comprehensive(X_test, y_test)
        
        # Extract requested metrics
        model_results = {'model': model_name}
        for metric in metrics:
            model_results[metric] = eval_results.get(metric, np.nan)
        
        results.append(model_results)
    
    return pd.DataFrame(results)

def plot_model_comparison(comparison_df: pd.DataFrame, metrics: List[str] = None, 
                         figsize: Tuple[int, int] = (15, 10)):
    """Plot model comparison results.
    
    Args:
        comparison_df: DataFrame from compare_models function
        metrics: List of metrics to plot
        figsize: Figure size
    """
    if metrics is None:
        metrics = [col for col in comparison_df.columns if col != 'model']
    
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i] if n_metrics > 1 else axes
        
        comparison_df.plot(x='model', y=metric, kind='bar', ax=ax, 
                          color='skyblue', legend=False)
        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def threshold_optimization(model, X_test: pd.DataFrame, y_test: pd.Series, 
                          metric: str = 'f1', thresholds: np.ndarray = None) -> Dict[str, Any]:
    """Optimize classification threshold for a given metric.
    
    Args:
        model: Trained model with predict_proba method
        X_test: Test features
        y_test: Test labels
        metric: Metric to optimize ('f1', 'precision', 'recall')
        thresholds: Array of thresholds to test
        
    Returns:
        Dictionary with optimization results
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.01)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    scores = []
    metrics_data = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        if metric == 'f1':
            score = f1_score(y_test, y_pred_thresh, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_test, y_pred_thresh, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_test, y_pred_thresh, zero_division=0)
        else:
            score = f1_score(y_test, y_pred_thresh, zero_division=0)
        
        scores.append(score)
        
        # Store all metrics for this threshold
        metrics_data.append({
            'threshold': threshold,
            'f1': f1_score(y_test, y_pred_thresh, zero_division=0),
            'precision': precision_score(y_test, y_pred_thresh, zero_division=0),
            'recall': recall_score(y_test, y_pred_thresh, zero_division=0),
            'accuracy': accuracy_score(y_test, y_pred_thresh)
        })
    
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]
    
    return {
        'optimal_threshold': optimal_threshold,
        'optimal_score': optimal_score,
        'all_scores': scores,
        'thresholds': thresholds.tolist(),
        'metrics_data': pd.DataFrame(metrics_data)
    }

def generate_classification_report_df(y_true: pd.Series, y_pred: np.ndarray, 
                                     target_names: List[str] = None) -> pd.DataFrame:
    """Generate classification report as DataFrame.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names for the classes
        
    Returns:
        DataFrame with classification report
    """
    if target_names is None:
        target_names = ['Legitimate', 'Fraud']
    
    report = classification_report(y_true, y_pred, target_names=target_names, 
                                 output_dict=True, zero_division=0)
    
    # Convert to DataFrame
    df = pd.DataFrame(report).transpose()
    
    return df