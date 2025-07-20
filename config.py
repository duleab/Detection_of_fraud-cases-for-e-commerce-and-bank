"""Configuration file for fraud detection project."""

import os
from pathlib import Path
from typing import Dict, List, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "Data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"

# Data file paths
FRAUD_DATA_PATH = DATA_DIR / "Fraud_Data.csv"
CREDIT_DATA_PATH = DATA_DIR / "creditcard.csv"
IP_COUNTRY_PATH = DATA_DIR / "IpAddress_to_Country.csv"

# Create directories if they don't exist
for directory in [RESULTS_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_STATE = 42

# Data processing configuration
DATA_CONFIG = {
    'missing_value_threshold': 0.5,  # Drop columns with >50% missing values
    'outlier_method': 'iqr',  # 'iqr', 'zscore', 'isolation'
    'outlier_action': 'cap',  # 'cap', 'remove', 'transform'
    'scaling_method': 'standard',  # 'standard', 'minmax', 'robust'
    'encoding_method': 'label',  # 'label', 'onehot', 'target'
    'test_size': 0.2,
    'validation_size': 0.2,
    'stratify': True
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'create_time_features': True,
    'create_user_features': True,
    'create_device_features': True,
    'create_interaction_features': True,
    'max_interactions': 10,
    'datetime_columns': ['signup_time', 'purchase_time'],
    'categorical_columns': ['source', 'browser', 'sex', 'country'],
    'numerical_columns': ['purchase_value', 'age']
}

# Model training configuration
MODEL_CONFIG = {
    'cv_folds': 5,
    'scoring_metric': 'f1',
    'hyperparameter_tuning': True,
    'n_iter_search': 20,  # For RandomizedSearchCV
    'n_jobs': -1,
    'verbose': 1
}

# Models to train
MODELS_TO_TRAIN = [
    'logistic_regression',
    'random_forest',
    'gradient_boosting',
    'xgboost',
    'lightgbm',
    'svm',
    'naive_bayes',
    'knn'
]

# Sampling strategies for imbalanced data
SAMPLING_STRATEGIES = [
    'none',
    'smote',
    'adasyn',
    'borderline_smote',
    'random_undersample',
    'smote_tomek'
]

# Evaluation metrics
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'roc_auc',
    'pr_auc',
    'specificity',
    'npv'  # Negative Predictive Value
]

# Business impact configuration
BUSINESS_CONFIG = {
    'average_fraud_amount': 100,  # Average amount per fraud case
    'investigation_cost_per_case': 15,  # Cost to investigate each flagged case
    'false_positive_cost': 5,  # Cost of inconveniencing legitimate customers
    'fraud_prevention_rate': 0.8,  # Percentage of fraud that can be prevented when detected
    'customer_lifetime_value': 500,  # Average customer lifetime value
    'churn_rate_from_false_positive': 0.1  # Percentage of customers who churn due to false positives
}

# Visualization configuration
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'style': 'seaborn-v0_8',
    'color_palette': 'husl',
    'dpi': 300,
    'save_format': 'png',
    'font_size': 12,
    'title_size': 14,
    'label_size': 10
}

# SHAP configuration for model interpretation
SHAP_CONFIG = {
    'max_display_features': 20,
    'sample_size': 1000,  # Sample size for SHAP calculations
    'plot_types': ['summary', 'waterfall', 'force', 'dependence'],
    'save_plots': True
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': RESULTS_DIR / 'fraud_detection.log'
}

# Model hyperparameter grids (detailed)
HYPERPARAMETER_GRIDS = {
    'logistic_regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000]
    },
    'random_forest': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    },
    'gradient_boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'xgboost': {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    },
    'lightgbm': {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5],
        'num_leaves': [31, 50, 100]
    },
    'svm': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'degree': [2, 3, 4]  # For polynomial kernel
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2]  # For minkowski metric
    },
    'naive_bayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    },
    'mlp': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh', 'logistic'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [500, 1000]
    }
}

# Feature importance methods
FEATURE_IMPORTANCE_METHODS = [
    'built_in',  # Model's built-in feature importance
    'permutation',  # Permutation importance
    'shap',  # SHAP values
    'correlation'  # Correlation with target
]

# Threshold optimization configuration
THRESHOLD_CONFIG = {
    'method': 'f1',  # 'f1', 'precision', 'recall', 'business_value'
    'thresholds': None,  # Auto-generate if None
    'n_thresholds': 100
}

# Cross-validation configuration
CV_CONFIG = {
    'method': 'stratified_kfold',  # 'stratified_kfold', 'time_series_split'
    'n_splits': 5,
    'shuffle': True,
    'random_state': RANDOM_STATE
}

# Ensemble configuration
ENSEMBLE_CONFIG = {
    'methods': ['voting', 'stacking'],
    'voting_type': 'soft',  # 'hard', 'soft'
    'stacking_final_estimator': 'logistic_regression',
    'stacking_cv': 5
}

# Data validation rules
DATA_VALIDATION_RULES = {
    'fraud_data': {
        'required_columns': ['user_id', 'signup_time', 'purchase_time', 'purchase_value', 
                           'device_id', 'source', 'browser', 'sex', 'age', 'ip_address', 'class'],
        'target_column': 'class',
        'date_columns': ['signup_time', 'purchase_time'],
        'numeric_columns': ['purchase_value', 'age'],
        'categorical_columns': ['source', 'browser', 'sex']
    },
    'credit_data': {
        'required_columns': ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)],
        'target_column': 'Class',
        'numeric_columns': ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    },
    'ip_country_data': {
        'required_columns': ['lower_bound_ip_address', 'upper_bound_ip_address', 'country']
    }
}

# Performance monitoring
PERFORMANCE_CONFIG = {
    'monitor_memory': True,
    'monitor_time': True,
    'memory_limit_gb': 8,
    'time_limit_minutes': 60
}

# Export configuration
EXPORT_CONFIG = {
    'save_models': True,
    'save_predictions': True,
    'save_feature_importance': True,
    'save_evaluation_metrics': True,
    'save_plots': True,
    'export_format': 'joblib',  # 'joblib', 'pickle'
    'compression': True
}

# Notification configuration (for long-running processes)
NOTIFICATION_CONFIG = {
    'enabled': False,
    'email': None,
    'slack_webhook': None
}

def get_config(section: str = None) -> Dict[str, Any]:
    """Get configuration for a specific section or all configurations.
    
    Args:
        section: Configuration section name
        
    Returns:
        Configuration dictionary
    """
    all_configs = {
        'data': DATA_CONFIG,
        'feature': FEATURE_CONFIG,
        'model': MODEL_CONFIG,
        'business': BUSINESS_CONFIG,
        'visualization': VIZ_CONFIG,
        'shap': SHAP_CONFIG,
        'logging': LOGGING_CONFIG,
        'hyperparameters': HYPERPARAMETER_GRIDS,
        'threshold': THRESHOLD_CONFIG,
        'cv': CV_CONFIG,
        'ensemble': ENSEMBLE_CONFIG,
        'validation': DATA_VALIDATION_RULES,
        'performance': PERFORMANCE_CONFIG,
        'export': EXPORT_CONFIG,
        'notification': NOTIFICATION_CONFIG
    }
    
    if section:
        return all_configs.get(section, {})
    return all_configs

def update_config(section: str, updates: Dict[str, Any]):
    """Update configuration for a specific section.
    
    Args:
        section: Configuration section name
        updates: Dictionary of updates to apply
    """
    config_map = {
        'data': DATA_CONFIG,
        'feature': FEATURE_CONFIG,
        'model': MODEL_CONFIG,
        'business': BUSINESS_CONFIG,
        'visualization': VIZ_CONFIG,
        'shap': SHAP_CONFIG,
        'threshold': THRESHOLD_CONFIG,
        'cv': CV_CONFIG,
        'ensemble': ENSEMBLE_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'export': EXPORT_CONFIG,
        'notification': NOTIFICATION_CONFIG
    }
    
    if section in config_map:
        config_map[section].update(updates)
    else:
        raise ValueError(f"Unknown configuration section: {section}")

def validate_config():
    """Validate configuration settings."""
    errors = []
    
    # Check if data files exist
    for file_path in [FRAUD_DATA_PATH, CREDIT_DATA_PATH, IP_COUNTRY_PATH]:
        if not file_path.exists():
            errors.append(f"Data file not found: {file_path}")
    
    # Validate numeric ranges
    if not 0 < DATA_CONFIG['test_size'] < 1:
        errors.append("test_size must be between 0 and 1")
    
    if not 0 < DATA_CONFIG['validation_size'] < 1:
        errors.append("validation_size must be between 0 and 1")
    
    if MODEL_CONFIG['cv_folds'] < 2:
        errors.append("cv_folds must be at least 2")
    
    # Validate model names
    valid_models = ['logistic_regression', 'random_forest', 'gradient_boosting', 
                   'xgboost', 'lightgbm', 'svm', 'naive_bayes', 'knn', 'mlp']
    
    for model in MODELS_TO_TRAIN:
        if model not in valid_models:
            errors.append(f"Unknown model: {model}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    print("✓ Configuration validation passed")

def print_config_summary():
    """Print a summary of current configuration."""
    print("Fraud Detection Project Configuration")
    print("=" * 50)
    
    print(f"\n📁 Paths:")
    print(f"   Project Root: {PROJECT_ROOT}")
    print(f"   Data Directory: {DATA_DIR}")
    print(f"   Results Directory: {RESULTS_DIR}")
    
    print(f"\n🎯 Models to Train: {len(MODELS_TO_TRAIN)}")
    for model in MODELS_TO_TRAIN:
        print(f"   • {model}")
    
    print(f"\n⚖️ Sampling Strategies: {len(SAMPLING_STRATEGIES)}")
    for strategy in SAMPLING_STRATEGIES:
        print(f"   • {strategy}")
    
    print(f"\n📊 Evaluation Metrics: {len(EVALUATION_METRICS)}")
    for metric in EVALUATION_METRICS:
        print(f"   • {metric}")
    
    print(f"\n🔧 Key Settings:")
    print(f"   Random State: {RANDOM_STATE}")
    print(f"   Test Size: {DATA_CONFIG['test_size']}")
    print(f"   CV Folds: {MODEL_CONFIG['cv_folds']}")
    print(f"   Hyperparameter Tuning: {MODEL_CONFIG['hyperparameter_tuning']}")
    
    print(f"\n💰 Business Impact:")
    print(f"   Average Fraud Amount: ${BUSINESS_CONFIG['average_fraud_amount']}")
    print(f"   Investigation Cost: ${BUSINESS_CONFIG['investigation_cost_per_case']}")
    print(f"   False Positive Cost: ${BUSINESS_CONFIG['false_positive_cost']}")

if __name__ == "__main__":
    validate_config()
    print_config_summary()