"""Model training utilities for fraud detection project."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

class FraudModelTrainer:
    """Comprehensive model training utility for fraud detection."""
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """Initialize model trainer.
        
        Args:
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = {}
        self.best_models = {}
        self.training_history = []
        self.sampling_strategies = {}
        
    def get_base_models(self) -> Dict[str, Any]:
        """Get dictionary of base models with default parameters.
        
        Returns:
            Dictionary of model name to model instance
        """
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=self.n_jobs
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=self.random_state
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1
            ),
            'svm': SVC(
                random_state=self.random_state,
                probability=True
            ),
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(
                n_jobs=self.n_jobs
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=self.random_state
            ),
            'mlp': MLPClassifier(
                random_state=self.random_state,
                max_iter=500
            )
        }
        
        return models
    
    def get_sampling_strategies(self) -> Dict[str, Any]:
        """Get dictionary of sampling strategies for handling class imbalance.
        
        Returns:
            Dictionary of strategy name to sampler instance
        """
        strategies = {
            'none': None,
            'smote': SMOTE(random_state=self.random_state),
            'adasyn': ADASYN(random_state=self.random_state),
            'borderline_smote': BorderlineSMOTE(random_state=self.random_state),
            'svm_smote': SVMSMOTE(random_state=self.random_state),
            'random_undersample': RandomUnderSampler(random_state=self.random_state),
            'tomek_links': TomekLinks(),
            'edited_nn': EditedNearestNeighbours(),
            'smote_tomek': SMOTETomek(random_state=self.random_state),
            'smote_enn': SMOTEENN(random_state=self.random_state)
        }
        
        return strategies
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """Get hyperparameter grids for model tuning.
        
        Returns:
            Dictionary of model name to parameter grid
        """
        param_grids = {
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'decision_tree': {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        return param_grids
    
    def train_single_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                          model_name: str, sampling_strategy: str = 'none',
                          hyperparameter_tuning: bool = False,
                          cv_folds: int = 5) -> Dict[str, Any]:
        """Train a single model with optional sampling and hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name of the model to train
            sampling_strategy: Sampling strategy to use
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results
        """
        print(f"Training {model_name} with {sampling_strategy} sampling...")
        
        # Get base model
        base_models = self.get_base_models()
        if model_name not in base_models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = base_models[model_name]
        
        # Get sampling strategy
        sampling_strategies = self.get_sampling_strategies()
        if sampling_strategy not in sampling_strategies:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        
        sampler = sampling_strategies[sampling_strategy]
        
        # Create pipeline with sampling (if specified)
        if sampler is not None:
            pipeline = ImbPipeline([
                ('sampler', sampler),
                ('classifier', model)
            ])
        else:
            pipeline = model
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            param_grids = self.get_hyperparameter_grids()
            if model_name in param_grids:
                param_grid = param_grids[model_name]
                
                # Adjust parameter names for pipeline
                if sampler is not None:
                    param_grid = {f'classifier__{k}': v for k, v in param_grid.items()}
                
                # Use RandomizedSearchCV for efficiency
                search = RandomizedSearchCV(
                    pipeline,
                    param_grid,
                    n_iter=20,
                    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                    scoring='f1',
                    n_jobs=self.n_jobs,
                    random_state=self.random_state
                )
                
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                best_params = search.best_params_
                best_score = search.best_score_
            else:
                # No hyperparameter grid available, use default model
                pipeline.fit(X_train, y_train)
                best_model = pipeline
                best_params = {}
                best_score = None
        else:
            # Train with default parameters
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = {}
            best_score = None
        
        # Cross-validation score
        cv_scores = cross_val_score(
            best_model, X_train, y_train,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring='f1',
            n_jobs=self.n_jobs
        )
        
        # Store results
        results = {
            'model': best_model,
            'model_name': model_name,
            'sampling_strategy': sampling_strategy,
            'best_params': best_params,
            'best_score': best_score,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': datetime.now()
        }
        
        # Store in models dictionary
        key = f"{model_name}_{sampling_strategy}"
        self.models[key] = results
        
        print(f"  CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def train_multiple_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                             model_names: List[str] = None,
                             sampling_strategies: List[str] = None,
                             hyperparameter_tuning: bool = False,
                             cv_folds: int = 5) -> pd.DataFrame:
        """Train multiple models with different sampling strategies.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_names: List of model names to train
            sampling_strategies: List of sampling strategies to use
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            cv_folds: Number of cross-validation folds
            
        Returns:
            DataFrame with training results
        """
        if model_names is None:
            model_names = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
        
        if sampling_strategies is None:
            sampling_strategies = ['none', 'smote', 'adasyn']
        
        results = []
        
        for model_name in model_names:
            for sampling_strategy in sampling_strategies:
                try:
                    result = self.train_single_model(
                        X_train, y_train, model_name, sampling_strategy,
                        hyperparameter_tuning, cv_folds
                    )
                    
                    results.append({
                        'model_name': model_name,
                        'sampling_strategy': sampling_strategy,
                        'cv_mean': result['cv_mean'],
                        'cv_std': result['cv_std'],
                        'best_score': result['best_score']
                    })
                    
                except Exception as e:
                    print(f"Error training {model_name} with {sampling_strategy}: {str(e)}")
                    continue
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('cv_mean', ascending=False)
        
        print(f"\nTraining completed. Best model: {results_df.iloc[0]['model_name']} with {results_df.iloc[0]['sampling_strategy']}")
        
        return results_df
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series,
                       models_to_evaluate: List[str] = None) -> pd.DataFrame:
        """Evaluate trained models on test set.
        
        Args:
            X_test: Test features
            y_test: Test target
            models_to_evaluate: List of model keys to evaluate
            
        Returns:
            DataFrame with evaluation results
        """
        if models_to_evaluate is None:
            models_to_evaluate = list(self.models.keys())
        
        results = []
        
        for model_key in models_to_evaluate:
            if model_key not in self.models:
                print(f"Model {model_key} not found in trained models.")
                continue
            
            model_info = self.models[model_key]
            model = model_info['model']
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            results.append({
                'model_key': model_key,
                'model_name': model_info['model_name'],
                'sampling_strategy': model_info['sampling_strategy'],
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_mean': model_info['cv_mean']
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        return results_df
    
    def get_feature_importance(self, model_key: str, feature_names: List[str] = None) -> pd.DataFrame:
        """Get feature importance from a trained model.
        
        Args:
            model_key: Key of the trained model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not found.")
        
        model_info = self.models[model_key]
        model = model_info['model']
        
        # Extract the actual classifier from pipeline if needed
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps['classifier']
        else:
            classifier = model
        
        # Get feature importance
        if hasattr(classifier, 'feature_importances_'):
            importance = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            importance = np.abs(classifier.coef_[0])
        else:
            print(f"Model {model_info['model_name']} does not support feature importance.")
            return pd.DataFrame()
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_models(self, save_dir: str, models_to_save: List[str] = None):
        """Save trained models to disk.
        
        Args:
            save_dir: Directory to save models
            models_to_save: List of model keys to save (None for all)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if models_to_save is None:
            models_to_save = list(self.models.keys())
        
        for model_key in models_to_save:
            if model_key in self.models:
                model_info = self.models[model_key]
                model_path = os.path.join(save_dir, f"{model_key}.joblib")
                
                # Save model
                joblib.dump(model_info['model'], model_path)
                
                # Save metadata
                metadata = {
                    'model_name': model_info['model_name'],
                    'sampling_strategy': model_info['sampling_strategy'],
                    'best_params': model_info['best_params'],
                    'cv_mean': model_info['cv_mean'],
                    'cv_std': model_info['cv_std'],
                    'training_time': model_info['training_time'].isoformat()
                }
                
                metadata_path = os.path.join(save_dir, f"{model_key}_metadata.json")
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"Saved {model_key} to {model_path}")
    
    def load_models(self, save_dir: str, model_keys: List[str] = None):
        """Load trained models from disk.
        
        Args:
            save_dir: Directory containing saved models
            model_keys: List of model keys to load (None for all)
        """
        if model_keys is None:
            # Find all model files
            model_files = [f for f in os.listdir(save_dir) if f.endswith('.joblib')]
            model_keys = [f.replace('.joblib', '') for f in model_files]
        
        for model_key in model_keys:
            model_path = os.path.join(save_dir, f"{model_key}.joblib")
            metadata_path = os.path.join(save_dir, f"{model_key}_metadata.json")
            
            if os.path.exists(model_path):
                # Load model
                model = joblib.load(model_path)
                
                # Load metadata
                metadata = {}
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                # Reconstruct model info
                model_info = {
                    'model': model,
                    'model_name': metadata.get('model_name', 'unknown'),
                    'sampling_strategy': metadata.get('sampling_strategy', 'unknown'),
                    'best_params': metadata.get('best_params', {}),
                    'cv_mean': metadata.get('cv_mean', 0),
                    'cv_std': metadata.get('cv_std', 0),
                    'training_time': datetime.fromisoformat(metadata.get('training_time', datetime.now().isoformat()))
                }
                
                self.models[model_key] = model_info
                print(f"Loaded {model_key} from {model_path}")
    
    def get_best_model(self, metric: str = 'cv_mean') -> Tuple[str, Dict[str, Any]]:
        """Get the best model based on specified metric.
        
        Args:
            metric: Metric to use for selection
            
        Returns:
            Tuple of (model_key, model_info)
        """
        if not self.models:
            raise ValueError("No models trained yet.")
        
        best_key = max(self.models.keys(), key=lambda k: self.models[k].get(metric, 0))
        return best_key, self.models[best_key]
    
    def create_ensemble_model(self, model_keys: List[str], method: str = 'voting') -> Any:
        """Create an ensemble model from multiple trained models.
        
        Args:
            model_keys: List of model keys to include in ensemble
            method: Ensemble method ('voting', 'stacking')
            
        Returns:
            Ensemble model
        """
        from sklearn.ensemble import VotingClassifier
        from sklearn.ensemble import StackingClassifier
        
        estimators = []
        for key in model_keys:
            if key in self.models:
                model_info = self.models[key]
                estimators.append((key, model_info['model']))
        
        if method == 'voting':
            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft'
            )
        elif method == 'stacking':
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(random_state=self.random_state),
                cv=5
            )
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return ensemble
    
    def get_training_summary(self) -> pd.DataFrame:
        """Get summary of all trained models.
        
        Returns:
            DataFrame with training summary
        """
        if not self.models:
            return pd.DataFrame()
        
        summary_data = []
        for key, info in self.models.items():
            summary_data.append({
                'model_key': key,
                'model_name': info['model_name'],
                'sampling_strategy': info['sampling_strategy'],
                'cv_mean': info['cv_mean'],
                'cv_std': info['cv_std'],
                'best_score': info.get('best_score'),
                'training_time': info['training_time']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('cv_mean', ascending=False)
        
        return summary_df

def quick_model_comparison(X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series,
                          models: List[str] = None) -> pd.DataFrame:
    """Quick comparison of multiple models with default settings.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: List of model names to compare
        
    Returns:
        DataFrame with comparison results
    """
    if models is None:
        models = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
    
    trainer = FraudModelTrainer()
    
    # Train models
    results = []
    for model_name in models:
        try:
            result = trainer.train_single_model(X_train, y_train, model_name)
            
            # Evaluate on test set
            model = result['model']
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            results.append({
                'model': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
                'cv_score': result['cv_mean']
            })
            
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
            continue
    
    return pd.DataFrame(results).sort_values('f1_score', ascending=False)