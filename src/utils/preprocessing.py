"""Data preprocessing utilities for fraud detection project."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from datetime import datetime, timedelta
import ipaddress
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

class FraudDataPreprocessor:
    """Comprehensive data preprocessing utility for fraud detection."""
    
    def __init__(self, random_state: int = 42):
        """Initialize preprocessor.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_stats = {}
        
    def handle_missing_values(self, df: pd.DataFrame, 
                             strategy: Dict[str, str] = None,
                             threshold: float = 0.5) -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Dict mapping column names to imputation strategies
            threshold: Drop columns with missing ratio above this threshold
            
        Returns:
            DataFrame with missing values handled
        """
        df_processed = df.copy()
        
        # Calculate missing ratios
        missing_ratios = df_processed.isnull().sum() / len(df_processed)
        
        # Drop columns with high missing ratio
        cols_to_drop = missing_ratios[missing_ratios > threshold].index
        if len(cols_to_drop) > 0:
            print(f"Dropping columns with >={threshold*100}% missing values: {list(cols_to_drop)}")
            df_processed = df_processed.drop(columns=cols_to_drop)
        
        # Default strategies
        default_strategy = {
            'numeric': 'median',
            'categorical': 'most_frequent'
        }
        
        if strategy is None:
            strategy = {}
        
        # Separate numeric and categorical columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        
        # Handle numeric columns
        for col in numeric_cols:
            if df_processed[col].isnull().any():
                impute_strategy = strategy.get(col, default_strategy['numeric'])
                
                if impute_strategy in ['mean', 'median', 'most_frequent', 'constant']:
                    imputer = SimpleImputer(strategy=impute_strategy)
                    df_processed[col] = imputer.fit_transform(df_processed[[col]]).ravel()
                    self.imputers[col] = imputer
                elif impute_strategy == 'knn':
                    imputer = KNNImputer(n_neighbors=5)
                    df_processed[col] = imputer.fit_transform(df_processed[[col]]).ravel()
                    self.imputers[col] = imputer
        
        # Handle categorical columns
        for col in categorical_cols:
            if df_processed[col].isnull().any():
                impute_strategy = strategy.get(col, default_strategy['categorical'])
                
                if impute_strategy == 'most_frequent':
                    mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
                    df_processed[col] = df_processed[col].fillna(mode_value)
                elif impute_strategy == 'constant':
                    df_processed[col] = df_processed[col].fillna('Unknown')
        
        print(f"Missing values handled. Remaining missing values: {df_processed.isnull().sum().sum()}")
        return df_processed
    
    def remove_duplicates(self, df: pd.DataFrame, 
                         subset: List[str] = None,
                         keep: str = 'first') -> pd.DataFrame:
        """Remove duplicate records.
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for identifying duplicates
            keep: Which duplicates to keep ('first', 'last', False)
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_count = len(df)
        df_processed = df.drop_duplicates(subset=subset, keep=keep)
        final_count = len(df_processed)
        
        duplicates_removed = initial_count - final_count
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate records ({duplicates_removed/initial_count*100:.2f}%)")
        
        return df_processed
    
    def detect_and_handle_outliers(self, df: pd.DataFrame, 
                                  columns: List[str] = None,
                                  method: str = 'iqr',
                                  action: str = 'cap') -> pd.DataFrame:
        """Detect and handle outliers.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers (None for all numeric)
            method: Outlier detection method ('iqr', 'zscore', 'isolation')
            action: Action to take ('cap', 'remove', 'transform')
            
        Returns:
            DataFrame with outliers handled
        """
        df_processed = df.copy()
        
        if columns is None:
            columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_stats = {}
        
        for col in columns:
            if col not in df_processed.columns:
                continue
                
            if method == 'iqr':
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df_processed[col].dropna()))
                outliers_mask = z_scores > 3
                lower_bound = df_processed[col].mean() - 3 * df_processed[col].std()
                upper_bound = df_processed[col].mean() + 3 * df_processed[col].std()
            
            outlier_count = outliers_mask.sum()
            outlier_stats[col] = {
                'count': outlier_count,
                'percentage': outlier_count / len(df_processed) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            if outlier_count > 0:
                if action == 'cap':
                    df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
                    df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
                elif action == 'remove':
                    df_processed = df_processed[~outliers_mask]
                elif action == 'transform':
                    # Log transformation for positive values
                    if df_processed[col].min() > 0:
                        df_processed[col] = np.log1p(df_processed[col])
        
        # Store outlier statistics
        self.feature_stats['outliers'] = outlier_stats
        
        print(f"Outlier detection completed using {method} method with {action} action.")
        for col, stats in outlier_stats.items():
            if stats['count'] > 0:
                print(f"  {col}: {stats['count']} outliers ({stats['percentage']:.2f}%)")
        
        return df_processed
    
    def create_time_features(self, df: pd.DataFrame, 
                           datetime_cols: List[str],
                           reference_col: str = None) -> pd.DataFrame:
        """Create time-based features from datetime columns.
        
        Args:
            df: Input DataFrame
            datetime_cols: List of datetime column names
            reference_col: Reference datetime column for calculating differences
            
        Returns:
            DataFrame with additional time features
        """
        df_processed = df.copy()
        
        for col in datetime_cols:
            if col not in df_processed.columns:
                continue
                
            # Convert to datetime if not already
            df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
            
            # Extract time components
            df_processed[f'{col}_year'] = df_processed[col].dt.year
            df_processed[f'{col}_month'] = df_processed[col].dt.month
            df_processed[f'{col}_day'] = df_processed[col].dt.day
            df_processed[f'{col}_hour'] = df_processed[col].dt.hour
            df_processed[f'{col}_dayofweek'] = df_processed[col].dt.dayofweek
            df_processed[f'{col}_is_weekend'] = (df_processed[col].dt.dayofweek >= 5).astype(int)
            
            # Create time periods
            df_processed[f'{col}_time_period'] = pd.cut(
                df_processed[col].dt.hour,
                bins=[0, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                include_lowest=True
            )
            
            # Calculate time since reference
            if reference_col and reference_col in df_processed.columns:
                df_processed[f'{col}_days_since_{reference_col}'] = (
                    df_processed[col] - df_processed[reference_col]
                ).dt.total_seconds() / (24 * 3600)
        
        print(f"Created time features for columns: {datetime_cols}")
        return df_processed
    
    def map_ip_to_country(self, df: pd.DataFrame, 
                         ip_col: str,
                         ip_country_df: pd.DataFrame) -> pd.DataFrame:
        """Map IP addresses to countries using IP range mapping.
        
        Args:
            df: Input DataFrame with IP addresses
            ip_col: Name of IP address column
            ip_country_df: DataFrame with IP range to country mapping
            
        Returns:
            DataFrame with country information added
        """
        df_processed = df.copy()
        
        def ip_to_int(ip_str):
            """Convert IP address string to integer."""
            try:
                return int(ipaddress.IPv4Address(ip_str))
            except:
                return None
        
        # Convert IP addresses to integers
        df_processed[f'{ip_col}_int'] = df_processed[ip_col].apply(ip_to_int)
        
        # Prepare IP country mapping
        ip_country_df = ip_country_df.copy()
        ip_country_df['lower_bound_int'] = ip_country_df['lower_bound_ip_address'].apply(ip_to_int)
        ip_country_df['upper_bound_int'] = ip_country_df['upper_bound_ip_address'].apply(ip_to_int)
        
        # Sort by lower bound for efficient searching
        ip_country_df = ip_country_df.sort_values('lower_bound_int')
        
        def find_country(ip_int):
            """Find country for given IP integer."""
            if pd.isna(ip_int):
                return 'Unknown'
            
            # Binary search for efficiency
            mask = (ip_country_df['lower_bound_int'] <= ip_int) & (ip_country_df['upper_bound_int'] >= ip_int)
            matches = ip_country_df[mask]
            
            if len(matches) > 0:
                return matches.iloc[0]['country']
            else:
                return 'Unknown'
        
        # Map IP addresses to countries
        df_processed['country'] = df_processed[f'{ip_col}_int'].apply(find_country)
        
        # Clean up temporary column
        df_processed = df_processed.drop(columns=[f'{ip_col}_int'])
        
        country_counts = df_processed['country'].value_counts()
        print(f"IP to country mapping completed. Found {len(country_counts)} unique countries.")
        print(f"Top 5 countries: {country_counts.head().to_dict()}")
        
        return df_processed
    
    def create_user_behavior_features(self, df: pd.DataFrame,
                                     user_col: str = 'user_id',
                                     amount_col: str = 'purchase_value',
                                     time_col: str = 'purchase_time') -> pd.DataFrame:
        """Create user behavior features.
        
        Args:
            df: Input DataFrame
            user_col: User identifier column
            amount_col: Transaction amount column
            time_col: Transaction time column
            
        Returns:
            DataFrame with user behavior features
        """
        df_processed = df.copy()
        
        # Ensure time column is datetime
        df_processed[time_col] = pd.to_datetime(df_processed[time_col])
        
        # Calculate user-level statistics
        user_stats = df_processed.groupby(user_col).agg({
            amount_col: ['count', 'sum', 'mean', 'std', 'min', 'max'],
            time_col: ['min', 'max']
        }).round(4)
        
        # Flatten column names
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
        user_stats = user_stats.reset_index()
        
        # Calculate additional features
        user_stats[f'{time_col}_span_days'] = (
            user_stats[f'{time_col}_max'] - user_stats[f'{time_col}_min']
        ).dt.total_seconds() / (24 * 3600)
        
        user_stats[f'{amount_col}_cv'] = (
            user_stats[f'{amount_col}_std'] / user_stats[f'{amount_col}_mean']
        ).fillna(0)
        
        # Merge back to original dataframe
        df_processed = df_processed.merge(user_stats, on=user_col, how='left')
        
        print(f"Created user behavior features for {len(user_stats)} unique users.")
        return df_processed
    
    def create_device_features(self, df: pd.DataFrame,
                              device_col: str = 'device_id') -> pd.DataFrame:
        """Create device-based features.
        
        Args:
            df: Input DataFrame
            device_col: Device identifier column
            
        Returns:
            DataFrame with device features
        """
        df_processed = df.copy()
        
        # Device usage statistics
        device_stats = df_processed.groupby(device_col).agg({
            'user_id': 'nunique',  # Number of unique users per device
            'purchase_value': ['count', 'sum', 'mean'],  # Transaction statistics
        }).round(4)
        
        device_stats.columns = ['device_user_count', 'device_transaction_count', 
                               'device_total_amount', 'device_avg_amount']
        device_stats = device_stats.reset_index()
        
        # Flag suspicious devices (multiple users)
        device_stats['device_is_shared'] = (device_stats['device_user_count'] > 1).astype(int)
        
        # Merge back to original dataframe
        df_processed = df_processed.merge(device_stats, on=device_col, how='left')
        
        print(f"Created device features for {len(device_stats)} unique devices.")
        return df_processed
    
    def encode_categorical_features(self, df: pd.DataFrame,
                                   categorical_cols: List[str] = None,
                                   encoding_method: str = 'label',
                                   handle_unknown: str = 'ignore') -> pd.DataFrame:
        """Encode categorical features.
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical columns (None for auto-detect)
            encoding_method: Encoding method ('label', 'onehot', 'target')
            handle_unknown: How to handle unknown categories
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_processed = df.copy()
        
        if categorical_cols is None:
            categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            if col not in df_processed.columns:
                continue
                
            if encoding_method == 'label':
                encoder = LabelEncoder()
                # Handle unknown values
                unique_values = df_processed[col].unique()
                encoder.fit(unique_values)
                
                # Transform with unknown handling
                def safe_transform(x):
                    try:
                        return encoder.transform([x])[0]
                    except ValueError:
                        return -1  # Unknown category
                
                df_processed[f'{col}_encoded'] = df_processed[col].apply(safe_transform)
                self.encoders[col] = encoder
                
            elif encoding_method == 'onehot':
                # One-hot encoding with top categories
                top_categories = df_processed[col].value_counts().head(10).index
                for category in top_categories:
                    df_processed[f'{col}_{category}'] = (df_processed[col] == category).astype(int)
        
        print(f"Encoded categorical features: {categorical_cols}")
        return df_processed
    
    def scale_features(self, df: pd.DataFrame,
                      feature_cols: List[str] = None,
                      method: str = 'standard',
                      fit_transform: bool = True) -> pd.DataFrame:
        """Scale numerical features.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature columns (None for all numeric)
            method: Scaling method ('standard', 'minmax', 'robust')
            fit_transform: Whether to fit and transform or just transform
            
        Returns:
            DataFrame with scaled features
        """
        df_processed = df.copy()
        
        if feature_cols is None:
            feature_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if present
        feature_cols = [col for col in feature_cols if col not in ['class', 'Class', 'target']]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        if fit_transform:
            df_processed[feature_cols] = scaler.fit_transform(df_processed[feature_cols])
            self.scalers[method] = scaler
        else:
            if method in self.scalers:
                df_processed[feature_cols] = self.scalers[method].transform(df_processed[feature_cols])
            else:
                raise ValueError(f"Scaler for method '{method}' not fitted yet.")
        
        print(f"Scaled {len(feature_cols)} features using {method} scaling.")
        return df_processed
    
    def create_interaction_features(self, df: pd.DataFrame,
                                   feature_pairs: List[Tuple[str, str]] = None,
                                   max_interactions: int = 10) -> pd.DataFrame:
        """Create interaction features between pairs of features.
        
        Args:
            df: Input DataFrame
            feature_pairs: List of feature pairs for interactions
            max_interactions: Maximum number of interactions to create
            
        Returns:
            DataFrame with interaction features
        """
        df_processed = df.copy()
        
        if feature_pairs is None:
            # Auto-select important numerical features
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in ['class', 'Class', 'target']]
            
            # Create pairs from top features (by variance)
            if len(numeric_cols) > 1:
                variances = df_processed[numeric_cols].var().sort_values(ascending=False)
                top_features = variances.head(min(5, len(numeric_cols))).index.tolist()
                
                feature_pairs = []
                for i in range(len(top_features)):
                    for j in range(i+1, len(top_features)):
                        feature_pairs.append((top_features[i], top_features[j]))
                        if len(feature_pairs) >= max_interactions:
                            break
                    if len(feature_pairs) >= max_interactions:
                        break
        
        # Create interaction features
        for i, (feat1, feat2) in enumerate(feature_pairs[:max_interactions]):
            if feat1 in df_processed.columns and feat2 in df_processed.columns:
                # Multiplication interaction
                df_processed[f'{feat1}_x_{feat2}'] = df_processed[feat1] * df_processed[feat2]
                
                # Ratio interaction (avoid division by zero)
                df_processed[f'{feat1}_div_{feat2}'] = df_processed[feat1] / (df_processed[feat2] + 1e-8)
        
        print(f"Created interaction features for {len(feature_pairs)} feature pairs.")
        return df_processed
    
    def prepare_train_test_split(self, df: pd.DataFrame,
                                target_col: str,
                                test_size: float = 0.2,
                                stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare train-test split.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            test_size: Proportion of test set
            stratify: Whether to stratify split by target
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=stratify_param
        )
        
        print(f"Train-test split completed:")
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        print(f"  Training fraud rate: {y_train.mean():.4f}")
        print(f"  Test fraud rate: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing steps applied.
        
        Returns:
            Dictionary with preprocessing summary
        """
        summary = {
            'scalers_fitted': list(self.scalers.keys()),
            'encoders_fitted': list(self.encoders.keys()),
            'imputers_fitted': list(self.imputers.keys()),
            'feature_statistics': self.feature_stats
        }
        
        return summary
    
    def save_preprocessing_artifacts(self, filepath: str):
        """Save preprocessing artifacts for later use.
        
        Args:
            filepath: Path to save artifacts
        """
        import joblib
        
        artifacts = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'imputers': self.imputers,
            'feature_stats': self.feature_stats,
            'random_state': self.random_state
        }
        
        joblib.dump(artifacts, filepath)
        print(f"Preprocessing artifacts saved to {filepath}")
    
    def load_preprocessing_artifacts(self, filepath: str):
        """Load preprocessing artifacts.
        
        Args:
            filepath: Path to load artifacts from
        """
        import joblib
        
        artifacts = joblib.load(filepath)
        
        self.scalers = artifacts.get('scalers', {})
        self.encoders = artifacts.get('encoders', {})
        self.imputers = artifacts.get('imputers', {})
        self.feature_stats = artifacts.get('feature_stats', {})
        self.random_state = artifacts.get('random_state', 42)
        
        print(f"Preprocessing artifacts loaded from {filepath}")

def create_feature_engineering_pipeline(df: pd.DataFrame,
                                       target_col: str,
                                       ip_country_df: pd.DataFrame = None,
                                       datetime_cols: List[str] = None,
                                       categorical_cols: List[str] = None) -> Tuple[pd.DataFrame, FraudDataPreprocessor]:
    """Create a complete feature engineering pipeline.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        ip_country_df: IP to country mapping DataFrame
        datetime_cols: List of datetime columns
        categorical_cols: List of categorical columns
        
    Returns:
        Tuple of (processed_df, preprocessor)
    """
    preprocessor = FraudDataPreprocessor()
    
    print("Starting feature engineering pipeline...")
    
    # Step 1: Handle missing values
    df_processed = preprocessor.handle_missing_values(df)
    
    # Step 2: Remove duplicates
    df_processed = preprocessor.remove_duplicates(df_processed)
    
    # Step 3: Create time features
    if datetime_cols:
        df_processed = preprocessor.create_time_features(df_processed, datetime_cols)
    
    # Step 4: Map IP to country
    if ip_country_df is not None and 'ip_address' in df_processed.columns:
        df_processed = preprocessor.map_ip_to_country(df_processed, 'ip_address', ip_country_df)
    
    # Step 5: Create user behavior features
    if 'user_id' in df_processed.columns:
        df_processed = preprocessor.create_user_behavior_features(df_processed)
    
    # Step 6: Create device features
    if 'device_id' in df_processed.columns:
        df_processed = preprocessor.create_device_features(df_processed)
    
    # Step 7: Handle outliers
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col]
    df_processed = preprocessor.detect_and_handle_outliers(df_processed, numeric_cols)
    
    # Step 8: Encode categorical features
    if categorical_cols:
        df_processed = preprocessor.encode_categorical_features(df_processed, categorical_cols)
    
    # Step 9: Create interaction features
    df_processed = preprocessor.create_interaction_features(df_processed)
    
    print("Feature engineering pipeline completed.")
    print(f"Final dataset shape: {df_processed.shape}")
    
    return df_processed, preprocessor