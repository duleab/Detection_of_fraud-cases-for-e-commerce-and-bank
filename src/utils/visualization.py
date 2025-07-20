"""Visualization utilities for fraud detection project."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional, Any
import warnings
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

class FraudVisualization:
    """Comprehensive visualization utility for fraud detection analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'seaborn-v0_8'):
        """Initialize visualization utility.
        
        Args:
            figsize: Default figure size
            style: Matplotlib style
        """
        self.figsize = figsize
        plt.style.use(style)
        
    def plot_class_distribution(self, df: pd.DataFrame, target_col: str, 
                               title: str = "Class Distribution", 
                               figsize: Optional[Tuple[int, int]] = None):
        """Plot class distribution with statistics.
        
        Args:
            df: DataFrame containing the data
            target_col: Name of the target column
            title: Plot title
            figsize: Figure size override
        """
        figsize = figsize or self.figsize
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Count plot
        class_counts = df[target_col].value_counts()
        axes[0].bar(class_counts.index, class_counts.values, 
                   color=['lightblue', 'lightcoral'])
        axes[0].set_title('Class Counts')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        
        # Add count labels
        for i, v in enumerate(class_counts.values):
            axes[0].text(i, v + max(class_counts.values) * 0.01, str(v), 
                        ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.2f%%',
                   colors=['lightblue', 'lightcoral'], startangle=90)
        axes[1].set_title('Class Proportion')
        
        plt.suptitle(title, fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        total = len(df)
        fraud_count = class_counts.get(1, 0)
        fraud_rate = fraud_count / total * 100
        
        print(f"Dataset Statistics:")
        print(f"Total samples: {total:,}")
        print(f"Fraud cases: {fraud_count:,} ({fraud_rate:.2f}%)")
        print(f"Legitimate cases: {total - fraud_count:,} ({100 - fraud_rate:.2f}%)")
        print(f"Imbalance ratio: 1:{total // fraud_count if fraud_count > 0 else 'N/A'}")
    
    def plot_feature_distributions(self, df: pd.DataFrame, features: List[str], 
                                  target_col: str, ncols: int = 3, 
                                  figsize: Optional[Tuple[int, int]] = None):
        """Plot feature distributions by class.
        
        Args:
            df: DataFrame containing the data
            features: List of feature names to plot
            target_col: Name of the target column
            ncols: Number of columns in subplot grid
            figsize: Figure size override
        """
        n_features = len(features)
        nrows = (n_features + ncols - 1) // ncols
        
        figsize = figsize or (ncols * 5, nrows * 4)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        if nrows == 1:
            axes = [axes] if ncols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(features):
            ax = axes[i] if n_features > 1 else axes
            
            # Plot distributions for each class
            for class_val in df[target_col].unique():
                class_data = df[df[target_col] == class_val][feature]
                label = 'Fraud' if class_val == 1 else 'Legitimate'
                ax.hist(class_data, alpha=0.7, label=label, bins=30, density=True)
            
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, features: List[str] = None, 
                               method: str = 'pearson', figsize: Optional[Tuple[int, int]] = None):
        """Plot correlation matrix heatmap.
        
        Args:
            df: DataFrame containing the data
            features: List of features to include (None for all numeric)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            figsize: Figure size override
        """
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        figsize = figsize or (min(len(features), 20), min(len(features), 20))
        
        corr_matrix = df[features].corr(method=method)
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title(f'{method.title()} Correlation Matrix', fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_time_series_analysis(self, df: pd.DataFrame, date_col: str, 
                                 target_col: str, freq: str = 'D',
                                 figsize: Optional[Tuple[int, int]] = None):
        """Plot time series analysis of fraud patterns.
        
        Args:
            df: DataFrame containing the data
            date_col: Name of the date column
            target_col: Name of the target column
            freq: Frequency for aggregation ('D', 'W', 'M')
            figsize: Figure size override
        """
        figsize = figsize or (15, 10)
        
        # Convert to datetime
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        # Aggregate by time period
        time_agg = df_copy.groupby(pd.Grouper(key=date_col, freq=freq)).agg({
            target_col: ['count', 'sum', 'mean']
        }).round(4)
        
        time_agg.columns = ['total_transactions', 'fraud_count', 'fraud_rate']
        time_agg = time_agg.reset_index()
        
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Total transactions over time
        axes[0].plot(time_agg[date_col], time_agg['total_transactions'], 
                    marker='o', linewidth=2)
        axes[0].set_title('Total Transactions Over Time')
        axes[0].set_ylabel('Count')
        axes[0].grid(True, alpha=0.3)
        
        # Fraud count over time
        axes[1].plot(time_agg[date_col], time_agg['fraud_count'], 
                    marker='o', linewidth=2, color='red')
        axes[1].set_title('Fraud Cases Over Time')
        axes[1].set_ylabel('Fraud Count')
        axes[1].grid(True, alpha=0.3)
        
        # Fraud rate over time
        axes[2].plot(time_agg[date_col], time_agg['fraud_rate'], 
                    marker='o', linewidth=2, color='orange')
        axes[2].set_title('Fraud Rate Over Time')
        axes[2].set_ylabel('Fraud Rate')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return time_agg
    
    def plot_geographical_analysis(self, df: pd.DataFrame, country_col: str, 
                                  target_col: str, top_n: int = 20,
                                  figsize: Optional[Tuple[int, int]] = None):
        """Plot geographical fraud analysis.
        
        Args:
            df: DataFrame containing the data
            country_col: Name of the country column
            target_col: Name of the target column
            top_n: Number of top countries to show
            figsize: Figure size override
        """
        figsize = figsize or (15, 10)
        
        # Calculate country statistics
        country_stats = df.groupby(country_col).agg({
            target_col: ['count', 'sum', 'mean']
        }).round(4)
        
        country_stats.columns = ['total_transactions', 'fraud_count', 'fraud_rate']
        country_stats = country_stats.reset_index().sort_values('total_transactions', ascending=False)
        
        # Get top countries by transaction volume
        top_countries = country_stats.head(top_n)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Transaction volume by country
        axes[0, 0].barh(range(len(top_countries)), top_countries['total_transactions'])
        axes[0, 0].set_yticks(range(len(top_countries)))
        axes[0, 0].set_yticklabels(top_countries[country_col])
        axes[0, 0].set_title('Transaction Volume by Country')
        axes[0, 0].set_xlabel('Total Transactions')
        
        # Fraud count by country
        axes[0, 1].barh(range(len(top_countries)), top_countries['fraud_count'], color='red')
        axes[0, 1].set_yticks(range(len(top_countries)))
        axes[0, 1].set_yticklabels(top_countries[country_col])
        axes[0, 1].set_title('Fraud Count by Country')
        axes[0, 1].set_xlabel('Fraud Cases')
        
        # Fraud rate by country
        fraud_rate_sorted = country_stats[country_stats['total_transactions'] >= 10].sort_values('fraud_rate', ascending=False).head(top_n)
        axes[1, 0].barh(range(len(fraud_rate_sorted)), fraud_rate_sorted['fraud_rate'], color='orange')
        axes[1, 0].set_yticks(range(len(fraud_rate_sorted)))
        axes[1, 0].set_yticklabels(fraud_rate_sorted[country_col])
        axes[1, 0].set_title('Fraud Rate by Country (min 10 transactions)')
        axes[1, 0].set_xlabel('Fraud Rate')
        
        # Scatter plot: volume vs fraud rate
        axes[1, 1].scatter(country_stats['total_transactions'], country_stats['fraud_rate'], alpha=0.6)
        axes[1, 1].set_xlabel('Total Transactions')
        axes[1, 1].set_ylabel('Fraud Rate')
        axes[1, 1].set_title('Transaction Volume vs Fraud Rate')
        axes[1, 1].set_xscale('log')
        
        plt.tight_layout()
        plt.show()
        
        return country_stats
    
    def plot_model_performance_comparison(self, results_df: pd.DataFrame, 
                                        metrics: List[str] = None,
                                        figsize: Optional[Tuple[int, int]] = None):
        """Plot model performance comparison.
        
        Args:
            results_df: DataFrame with model results
            metrics: List of metrics to plot
            figsize: Figure size override
        """
        if metrics is None:
            metrics = ['f1_score', 'precision', 'recall', 'roc_auc']
        
        figsize = figsize or (15, 10)
        
        n_metrics = len(metrics)
        ncols = 2
        nrows = (n_metrics + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1:
            axes = [axes] if ncols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i] if n_metrics > 1 else axes
            
            # Bar plot
            results_df.plot(x='model', y=metric, kind='bar', ax=ax, 
                           color='skyblue', legend=False)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_xlabel('Model')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for j, v in enumerate(results_df[metric]):
                ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_pr_curves_comparison(self, models_results: Dict[str, Dict], 
                                     figsize: Optional[Tuple[int, int]] = None):
        """Plot ROC and PR curves for multiple models.
        
        Args:
            models_results: Dict with model_name: {y_true, y_pred_proba} pairs
            figsize: Figure size override
        """
        figsize = figsize or (15, 6)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))
        
        for i, (model_name, results) in enumerate(models_results.items()):
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = np.trapz(tpr, fpr)
            ax1.plot(fpr, tpr, color=colors[i], lw=2, 
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
            
            # PR Curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = np.trapz(precision, recall)
            ax2.plot(recall, precision, color=colors[i], lw=2, 
                    label=f'{model_name} (AUC = {pr_auc:.3f})')
        
        # ROC plot formatting
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves Comparison', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PR plot formatting
        baseline = sum(list(models_results.values())[0]['y_true']) / len(list(models_results.values())[0]['y_true'])
        ax2.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
                   label=f'Baseline ({baseline:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves Comparison', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices_comparison(self, models_results: Dict[str, Dict], 
                                          normalize: bool = False,
                                          figsize: Optional[Tuple[int, int]] = None):
        """Plot confusion matrices for multiple models.
        
        Args:
            models_results: Dict with model_name: {y_true, y_pred} pairs
            normalize: Whether to normalize confusion matrices
            figsize: Figure size override
        """
        n_models = len(models_results)
        ncols = min(3, n_models)
        nrows = (n_models + ncols - 1) // ncols
        
        figsize = figsize or (ncols * 5, nrows * 4)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1:
            axes = [axes] if ncols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (model_name, results) in enumerate(models_results.items()):
            ax = axes[i] if n_models > 1 else axes
            
            y_true = results['y_true']
            y_pred = results['y_pred']
            
            cm = confusion_matrix(y_true, y_pred)
            
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                fmt = '.2%'
            else:
                fmt = 'd'
            
            sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                       xticklabels=['Legitimate', 'Fraud'],
                       yticklabels=['Legitimate', 'Fraud'])
            ax.set_title(f'{model_name}', fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        title = 'Normalized Confusion Matrices' if normalize else 'Confusion Matrices'
        plt.suptitle(title, fontweight='bold', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance_comparison(self, importance_results: Dict[str, pd.DataFrame], 
                                          top_n: int = 15,
                                          figsize: Optional[Tuple[int, int]] = None):
        """Plot feature importance comparison across models.
        
        Args:
            importance_results: Dict with model_name: importance_df pairs
            top_n: Number of top features to show
            figsize: Figure size override
        """
        figsize = figsize or (15, 10)
        
        # Combine all importance results
        combined_df = pd.concat([
            df.assign(model=model_name) for model_name, df in importance_results.items()
        ], ignore_index=True)
        
        # Get top features across all models
        avg_importance = combined_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        top_features = avg_importance.head(top_n).index
        
        # Filter for top features
        plot_df = combined_df[combined_df['feature'].isin(top_features)]
        
        # Create pivot table for heatmap
        pivot_df = plot_df.pivot(index='feature', columns='model', values='importance')
        pivot_df = pivot_df.reindex(top_features)
        
        # Plot heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Feature Importance'})
        plt.title('Feature Importance Comparison Across Models', fontweight='bold', fontsize=14)
        plt.xlabel('Model')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
        
        return avg_importance
    
    def plot_business_impact_analysis(self, business_results: pd.DataFrame,
                                     figsize: Optional[Tuple[int, int]] = None):
        """Plot business impact analysis results.
        
        Args:
            business_results: DataFrame with business impact metrics
            figsize: Figure size override
        """
        figsize = figsize or (16, 12)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Net benefit comparison
        business_results.plot(x='model', y='net_benefit', kind='bar', ax=axes[0,0], color='green')
        axes[0,0].set_title('Net Business Benefit by Model', fontweight='bold')
        axes[0,0].set_ylabel('Net Benefit ($)')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # ROI comparison
        business_results.plot(x='model', y='roi', kind='bar', ax=axes[0,1], color='blue')
        axes[0,1].set_title('Return on Investment by Model', fontweight='bold')
        axes[0,1].set_ylabel('ROI')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Customer friction vs fraud detection
        scatter = axes[1,0].scatter(business_results['customer_friction_rate'], 
                                   business_results['fraud_detection_rate'],
                                   c=business_results['net_benefit'], 
                                   cmap='viridis', s=100)
        axes[1,0].set_xlabel('Customer Friction Rate (False Positive Rate)')
        axes[1,0].set_ylabel('Fraud Detection Rate (Recall)')
        axes[1,0].set_title('Customer Experience vs Fraud Detection', fontweight='bold')
        plt.colorbar(scatter, ax=axes[1,0], label='Net Benefit ($)')
        
        # Cost breakdown
        cost_columns = ['fraud_prevented_value', 'fraud_losses', 'investigation_costs']
        if all(col in business_results.columns for col in cost_columns):
            cost_data = business_results[['model'] + cost_columns].set_index('model')
            cost_data.plot(kind='bar', stacked=True, ax=axes[1,1])
            axes[1,1].set_title('Cost Breakdown by Model', fontweight='bold')
            axes[1,1].set_ylabel('Amount ($)')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].legend(['Fraud Prevented', 'Fraud Losses', 'Investigation Costs'])
        
        plt.tight_layout()
        plt.show()

def create_interactive_dashboard(df: pd.DataFrame, target_col: str, 
                               features: List[str]) -> go.Figure:
    """Create interactive dashboard using Plotly.
    
    Args:
        df: DataFrame containing the data
        target_col: Name of the target column
        features: List of features to include
        
    Returns:
        Plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Class Distribution', 'Feature Correlation', 
                       'Feature Distribution', 'Time Series'),
        specs=[[{"type": "pie"}, {"type": "heatmap"}],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )
    
    # Class distribution pie chart
    class_counts = df[target_col].value_counts()
    fig.add_trace(
        go.Pie(labels=['Legitimate', 'Fraud'], values=class_counts.values,
               name="Class Distribution"),
        row=1, col=1
    )
    
    # Feature correlation heatmap
    corr_matrix = df[features].corr()
    fig.add_trace(
        go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                  colorscale='RdBu', name="Correlation"),
        row=1, col=2
    )
    
    # Feature distribution histogram
    for class_val in df[target_col].unique():
        class_data = df[df[target_col] == class_val][features[0]]
        label = 'Fraud' if class_val == 1 else 'Legitimate'
        fig.add_trace(
            go.Histogram(x=class_data, name=label, opacity=0.7),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title_text="Fraud Detection Dashboard",
        showlegend=True,
        height=800
    )
    
    return fig

def save_plots_to_pdf(plot_functions: List[callable], filename: str = "fraud_analysis_plots.pdf"):
    """Save multiple plots to a PDF file.
    
    Args:
        plot_functions: List of functions that generate plots
        filename: Output PDF filename
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages(filename) as pdf:
        for plot_func in plot_functions:
            plot_func()
            pdf.savefig(bbox_inches='tight')
            plt.close()
    
    print(f"Plots saved to {filename}")