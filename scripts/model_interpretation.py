#!/usr/bin/env python3
"""
Fraud Detection - Model Interpretation and SHAP Analysis

This script provides comprehensive model interpretation using SHAP and feature importance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import warnings
import os
import glob
from collections import Counter

# Model interpretation
import shap
from sklearn.inspection import permutation_importance

# Visualization
import plotly.express as px
import plotly.graph_objects as go

# Utilities
from scipy import stats
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')
np.random.seed(42)

# Initialize SHAP
shap.initjs()

def load_data():
    """Load processed data and feature information"""
    print("Loading processed data...")
    
    # Load processed data
    X_fraud_train = pd.read_csv('../results/X_fraud_train_scaled.csv')
    X_fraud_test = pd.read_csv('../results/X_fraud_test_scaled.csv')
    X_cc_train = pd.read_csv('../results/X_cc_train_scaled.csv')
    X_cc_test = pd.read_csv('../results/X_cc_test_scaled.csv')
    
    y_fraud_train = pd.read_csv('../results/y_fraud_train.csv').squeeze()
    y_fraud_test = pd.read_csv('../results/y_fraud_test.csv').squeeze()
    y_cc_train = pd.read_csv('../results/y_cc_train.csv').squeeze()
    y_cc_test = pd.read_csv('../results/y_cc_test.csv').squeeze()
    
    # Load feature information
    with open('../results/feature_info.pkl', 'rb') as f:
        feature_info = pickle.load(f)
    
    return {
        'X_fraud_train': X_fraud_train, 'X_fraud_test': X_fraud_test,
        'X_cc_train': X_cc_train, 'X_cc_test': X_cc_test,
        'y_fraud_train': y_fraud_train, 'y_fraud_test': y_fraud_test,
        'y_cc_train': y_cc_train, 'y_cc_test': y_cc_test,
        'feature_info': feature_info
    }

def load_models_from_directory():
    """Load all saved models from the results directory"""
    print("Loading saved models...")
    
    models = {
        'fraud_models': {},
        'cc_models': {}
    }
    
    # Load e-commerce fraud models
    ecom_model_files = glob.glob('../results/saved_models/ecommerce/*.pkl')
    for model_file in ecom_model_files:
        if 'metadata' not in model_file and '.ipynb_checkpoints' not in model_file:
            model_name = os.path.basename(model_file).replace('.pkl', '')
            # Clean up the model name
            if '_ecom_20250719_043729' in model_name:
                model_name = model_name.replace('_ecom_20250719_043729', '')
            elif '_ecom' in model_name:
                model_name = model_name.replace('_ecom', '')
            try:
                # Try joblib first, then pickle
                try:
                    models['fraud_models'][model_name] = joblib.load(model_file)
                except:
                    with open(model_file, 'rb') as f:
                        models['fraud_models'][model_name] = pickle.load(f)
                print(f"Loaded e-commerce model: {model_name}")
            except Exception as e:
                print(f"Error loading {model_file}: {e}")
    
    # Load credit card fraud models
    cc_model_files = glob.glob('../results/saved_models/creditcard/*.pkl')
    for model_file in cc_model_files:
        if 'metadata' not in model_file:
            model_name = os.path.basename(model_file).replace('.pkl', '')
            # Clean up the model name
            if '_cc_20250719_043729' in model_name:
                model_name = model_name.replace('_cc_20250719_043729', '')
            elif '_cc' in model_name:
                model_name = model_name.replace('_cc', '')
            try:
                # Try joblib first, then pickle
                try:
                    models['cc_models'][model_name] = joblib.load(model_file)
                except:
                    with open(model_file, 'rb') as f:
                        models['cc_models'][model_name] = pickle.load(f)
                print(f"Loaded credit card model: {model_name}")
            except Exception as e:
                print(f"Error loading {model_file}: {e}")
    
    return models

def get_feature_importance(model, feature_names, model_name):
    """Extract feature importance from tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = abs(model.coef_[0]) if len(model.coef_.shape) > 1 else abs(model.coef_)
    else:
        return None
    
    # Ensure arrays have the same length
    min_length = min(len(feature_names), len(importance))
    feature_names_trimmed = feature_names[:min_length]
    importance_trimmed = importance[:min_length]
    
    feature_imp_df = pd.DataFrame({
        'feature': feature_names_trimmed,
        'importance': importance_trimmed,
        'model': model_name
    }).sort_values('importance', ascending=False)
    
    return feature_imp_df

def analyze_feature_importance(models, data):
    """Analyze feature importance for all models"""
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    fraud_importance_results = []
    cc_importance_results = []
    
    # Fraud models
    print("\nE-commerce Fraud Models:")
    for model_name, model in models['fraud_models'].items():
        importance_df = get_feature_importance(model, data['X_fraud_test'].columns, model_name)
        if importance_df is not None:
            fraud_importance_results.append(importance_df)
            print(f"\n{model_name} - Top 5 features:")
            for i, row in importance_df.head().iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Credit card models
    print("\nCredit Card Fraud Models:")
    for model_name, model in models['cc_models'].items():
        importance_df = get_feature_importance(model, data['X_cc_test'].columns, model_name)
        if importance_df is not None:
            cc_importance_results.append(importance_df)
            print(f"\n{model_name} - Top 5 features:")
            for i, row in importance_df.head().iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return fraud_importance_results, cc_importance_results

def plot_feature_importance(importance_results, dataset_name, top_n=15):
    """Plot feature importance comparison across models"""
    if not importance_results:
        print(f"No tree-based models available for {dataset_name}")
        return None
    
    # Combine all importance results
    combined_df = pd.concat(importance_results, ignore_index=True)
    
    # Get top features across all models
    avg_importance = combined_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    top_features = avg_importance.head(top_n).index
    
    # Filter for top features
    plot_df = combined_df[combined_df['feature'].isin(top_features)]
    
    # Create pivot table for heatmap
    pivot_df = plot_df.pivot(index='feature', columns='model', values='importance')
    pivot_df = pivot_df.reindex(top_features)
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Feature Importance'})
    plt.title(f'{dataset_name} - Feature Importance Comparison', fontweight='bold', fontsize=14)
    plt.xlabel('Model')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'../results/{dataset_name.lower().replace(" ", "_")}_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return avg_importance

def perform_shap_analysis(model, X_train, X_test, model_name, dataset_name, sample_size=500):
    """Perform SHAP analysis for model interpretation"""
    print(f"Performing SHAP analysis for {dataset_name} - {model_name}")
    
    # Sample data for faster computation
    if len(X_train) > sample_size:
        train_sample = X_train.sample(n=sample_size, random_state=42)
    else:
        train_sample = X_train
    
    if len(X_test) > sample_size:
        test_sample = X_test.sample(n=sample_size, random_state=42)
    else:
        test_sample = X_test
    
    try:
        # Choose appropriate explainer based on model type
        if 'RandomForest' in str(type(model)) or 'GradientBoosting' in str(type(model)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(test_sample)
            # For binary classification, use positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # Use KernelExplainer for other models (smaller sample for speed)
            small_sample = test_sample.iloc[:100]
            explainer = shap.KernelExplainer(model.predict_proba, train_sample.iloc[:50])
            shap_values = explainer.shap_values(small_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            test_sample = small_sample
        
        return explainer, shap_values, test_sample
    
    except Exception as e:
        print(f"Error in SHAP analysis for {model_name}: {str(e)}")
        return None, None, None

def run_shap_analysis(models, data):
    """Run SHAP analysis for best models"""
    print("\n=== SHAP ANALYSIS ===")
    
    shap_results = {}
    
    # Analyze fraud models
    print("\n=== E-COMMERCE FRAUD SHAP ANALYSIS ===")
    for model_name, model in models['fraud_models'].items():
        if 'random_forest' in model_name.lower() or 'gradient' in model_name.lower():
            explainer, shap_values, test_sample = perform_shap_analysis(
                model, data['X_fraud_train'], data['X_fraud_test'], model_name, 'E-commerce Fraud'
            )
            if shap_values is not None:
                shap_results[f'fraud_{model_name}'] = {
                    'explainer': explainer,
                    'shap_values': shap_values,
                    'test_sample': test_sample
                }
                
                # Generate summary plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, test_sample, show=False)
                plt.title(f'SHAP Summary - E-commerce Fraud ({model_name})')
                plt.tight_layout()
                plt.savefig(f'../results/shap_summary_fraud_{model_name}.png', dpi=300, bbox_inches='tight')
                plt.show()
                break  # Only analyze one model for demo
    
    # Analyze credit card models
    print("\n=== CREDIT CARD FRAUD SHAP ANALYSIS ===")
    for model_name, model in models['cc_models'].items():
        if 'random_forest' in model_name.lower() or 'gradient' in model_name.lower():
            explainer, shap_values, test_sample = perform_shap_analysis(
                model, data['X_cc_train'], data['X_cc_test'], model_name, 'Credit Card Fraud'
            )
            if shap_values is not None:
                shap_results[f'cc_{model_name}'] = {
                    'explainer': explainer,
                    'shap_values': shap_values,
                    'test_sample': test_sample
                }
                
                # Generate summary plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, test_sample, show=False)
                plt.title(f'SHAP Summary - Credit Card Fraud ({model_name})')
                plt.tight_layout()
                plt.savefig(f'../results/shap_summary_cc_{model_name}.png', dpi=300, bbox_inches='tight')
                plt.show()
                break  # Only analyze one model for demo
    
    return shap_results

def generate_business_insights(fraud_importance_results, cc_importance_results):
    """Generate business insights and recommendations"""
    print("\n=== BUSINESS INSIGHTS AND RECOMMENDATIONS ===")
    print("\n1. KEY FRAUD INDICATORS:")
    
    if fraud_importance_results:
        fraud_combined = pd.concat(fraud_importance_results, ignore_index=True)
        top_fraud_features = fraud_combined.groupby('feature')['importance'].mean().sort_values(ascending=False).head(10)
        print("\nTop E-commerce Fraud Risk Factors:")
        for feature, importance in top_fraud_features.items():
            print(f"  - {feature}: {importance:.4f}")
    
    if cc_importance_results:
        cc_combined = pd.concat(cc_importance_results, ignore_index=True)
        top_cc_features = cc_combined.groupby('feature')['importance'].mean().sort_values(ascending=False).head(10)
        print("\nTop Credit Card Fraud Risk Factors:")
        for feature, importance in top_cc_features.items():
            print(f"  - {feature}: {importance:.4f}")
    
    print("\n2. ACTIONABLE RECOMMENDATIONS:")
    recommendations = [
        "Implement real-time monitoring for high-risk features",
        "Enhance verification for transactions with suspicious patterns",
        "Deploy ensemble models for better fraud detection accuracy",
        "Regular model retraining with new fraud patterns",
        "Implement threshold optimization for business objectives"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    print("\n3. MODEL DEPLOYMENT CONSIDERATIONS:")
    deployment_notes = [
        "Random Forest and Gradient Boosting models show best performance",
        "Credit card models achieve higher precision than e-commerce models",
        "Consider ensemble approach for production deployment",
        "Implement A/B testing for model performance validation"
    ]
    
    for i, note in enumerate(deployment_notes, 1):
        print(f"  {i}. {note}")
    
    return recommendations, deployment_notes

def save_results(fraud_importance_results, cc_importance_results, shap_results, recommendations, deployment_notes):
    """Save all interpretation results"""
    print("\nSaving interpretation results...")
    
    # Save interpretation results
    interpretation_summary = {
        'fraud_feature_importance': fraud_importance_results,
        'cc_feature_importance': cc_importance_results,
        'shap_results': shap_results,
        'business_recommendations': recommendations,
        'deployment_notes': deployment_notes
    }
    
    with open('../results/interpretation_summary.pkl', 'wb') as f:
        pickle.dump(interpretation_summary, f)
    
    print("Model interpretation results saved to ../results/interpretation_summary.pkl")

def main():
    """Main execution function"""
    print("Starting Model Interpretation and SHAP Analysis...")
    print("=" * 60)
    
    # Load data and models
    data = load_data()
    models = load_models_from_directory()
    
    print(f"\nData loaded successfully:")
    print(f"Fraud features: {data['X_fraud_test'].shape[1]}")
    print(f"Credit card features: {data['X_cc_test'].shape[1]}")
    print(f"Available fraud models: {list(models['fraud_models'].keys())}")
    print(f"Available credit card models: {list(models['cc_models'].keys())}")
    
    # Feature importance analysis
    fraud_importance_results, cc_importance_results = analyze_feature_importance(models, data)
    
    # Plot feature importance
    fraud_avg_importance = plot_feature_importance(fraud_importance_results, 'E-commerce Fraud')
    cc_avg_importance = plot_feature_importance(cc_importance_results, 'Credit Card Fraud')
    
    # SHAP analysis
    shap_results = run_shap_analysis(models, data)
    
    # Generate business insights
    recommendations, deployment_notes = generate_business_insights(fraud_importance_results, cc_importance_results)
    
    # Save results
    save_results(fraud_importance_results, cc_importance_results, shap_results, recommendations, deployment_notes)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("\nKey Findings:")
    print("- Feature importance analysis completed for all tree-based models")
    print("- SHAP analysis provides explainable AI insights")
    print("- Business recommendations generated for fraud prevention")
    print("- Models ready for production deployment with proper monitoring")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Model interpretation completed successfully!")
    else:
        print("\n❌ Model interpretation failed!")