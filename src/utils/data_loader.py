"""Data loading utilities for fraud detection project."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDataLoader:
    """Utility class for loading and initial processing of fraud detection datasets."""
    
    def __init__(self, data_dir: str = "../Data"):
        """Initialize data loader with data directory path.
        
        Args:
            data_dir: Path to directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.fraud_file = self.data_dir / "Fraud_Data.csv"
        self.cc_file = self.data_dir / "creditcard.csv"
        self.ip_country_file = self.data_dir / "IpAddress_to_Country.csv"
        
    def load_fraud_data(self) -> pd.DataFrame:
        """Load e-commerce fraud dataset.
        
        Returns:
            DataFrame containing fraud data
        """
        try:
            logger.info(f"Loading fraud data from {self.fraud_file}")
            df = pd.read_csv(self.fraud_file)
            logger.info(f"Loaded fraud data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading fraud data: {str(e)}")
            raise
    
    def load_creditcard_data(self) -> pd.DataFrame:
        """Load credit card fraud dataset.
        
        Returns:
            DataFrame containing credit card data
        """
        try:
            logger.info(f"Loading credit card data from {self.cc_file}")
            df = pd.read_csv(self.cc_file)
            logger.info(f"Loaded credit card data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading credit card data: {str(e)}")
            raise
    
    def load_ip_country_mapping(self) -> pd.DataFrame:
        """Load IP address to country mapping data.
        
        Returns:
            DataFrame containing IP to country mappings
        """
        try:
            logger.info(f"Loading IP-country mapping from {self.ip_country_file}")
            df = pd.read_csv(self.ip_country_file)
            logger.info(f"Loaded IP-country mapping: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading IP-country mapping: {str(e)}")
            raise
    
    def load_all_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all datasets.
        
        Returns:
            Tuple of (fraud_data, creditcard_data, ip_country_mapping)
        """
        fraud_data = self.load_fraud_data()
        cc_data = self.load_creditcard_data()
        ip_mapping = self.load_ip_country_mapping()
        
        return fraud_data, cc_data, ip_mapping
    
    def get_data_info(self) -> Dict[str, Dict]:
        """Get basic information about all datasets.
        
        Returns:
            Dictionary containing dataset information
        """
        info = {}
        
        try:
            # Fraud data info
            fraud_df = self.load_fraud_data()
            info['fraud'] = {
                'shape': fraud_df.shape,
                'columns': fraud_df.columns.tolist(),
                'dtypes': fraud_df.dtypes.to_dict(),
                'missing_values': fraud_df.isnull().sum().to_dict(),
                'fraud_rate': fraud_df['class'].mean() if 'class' in fraud_df.columns else None
            }
            
            # Credit card data info
            cc_df = self.load_creditcard_data()
            info['creditcard'] = {
                'shape': cc_df.shape,
                'columns': cc_df.columns.tolist(),
                'dtypes': cc_df.dtypes.to_dict(),
                'missing_values': cc_df.isnull().sum().to_dict(),
                'fraud_rate': cc_df['Class'].mean() if 'Class' in cc_df.columns else None
            }
            
            # IP mapping info
            ip_df = self.load_ip_country_mapping()
            info['ip_mapping'] = {
                'shape': ip_df.shape,
                'columns': ip_df.columns.tolist(),
                'dtypes': ip_df.dtypes.to_dict(),
                'missing_values': ip_df.isnull().sum().to_dict(),
                'unique_countries': ip_df['country'].nunique() if 'country' in ip_df.columns else None
            }
            
        except Exception as e:
            logger.error(f"Error getting data info: {str(e)}")
            
        return info
    
    def validate_data_quality(self) -> Dict[str, Dict]:
        """Validate data quality for all datasets.
        
        Returns:
            Dictionary containing data quality metrics
        """
        quality_report = {}
        
        try:
            # Validate fraud data
            fraud_df = self.load_fraud_data()
            quality_report['fraud'] = self._validate_fraud_data(fraud_df)
            
            # Validate credit card data
            cc_df = self.load_creditcard_data()
            quality_report['creditcard'] = self._validate_creditcard_data(cc_df)
            
            # Validate IP mapping data
            ip_df = self.load_ip_country_mapping()
            quality_report['ip_mapping'] = self._validate_ip_mapping_data(ip_df)
            
        except Exception as e:
            logger.error(f"Error validating data quality: {str(e)}")
            
        return quality_report
    
    def _validate_fraud_data(self, df: pd.DataFrame) -> Dict:
        """Validate fraud dataset quality."""
        validation = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_records': df.duplicated().sum(),
            'class_distribution': df['class'].value_counts().to_dict() if 'class' in df.columns else None,
            'date_range': None,
            'issues': []
        }
        
        # Check date columns
        date_columns = ['signup_time', 'purchase_time']
        for col in date_columns:
            if col in df.columns:
                try:
                    dates = pd.to_datetime(df[col])
                    if validation['date_range'] is None:
                        validation['date_range'] = {}
                    validation['date_range'][col] = {
                        'min': dates.min().strftime('%Y-%m-%d'),
                        'max': dates.max().strftime('%Y-%m-%d')
                    }
                except:
                    validation['issues'].append(f"Invalid date format in {col}")
        
        # Check for negative values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (df[col] < 0).any():
                validation['issues'].append(f"Negative values found in {col}")
        
        return validation
    
    def _validate_creditcard_data(self, df: pd.DataFrame) -> Dict:
        """Validate credit card dataset quality."""
        validation = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_records': df.duplicated().sum(),
            'class_distribution': df['Class'].value_counts().to_dict() if 'Class' in df.columns else None,
            'feature_ranges': {},
            'issues': []
        }
        
        # Check feature ranges
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            validation['feature_ranges'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
        
        # Check for extreme outliers in Amount
        if 'Amount' in df.columns:
            q99 = df['Amount'].quantile(0.99)
            extreme_amounts = (df['Amount'] > q99 * 10).sum()
            if extreme_amounts > 0:
                validation['issues'].append(f"{extreme_amounts} transactions with extremely high amounts")
        
        return validation
    
    def _validate_ip_mapping_data(self, df: pd.DataFrame) -> Dict:
        """Validate IP mapping dataset quality."""
        validation = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_records': df.duplicated().sum(),
            'unique_countries': df['country'].nunique() if 'country' in df.columns else None,
            'ip_range_coverage': None,
            'issues': []
        }
        
        # Check IP range consistency
        if 'lower_bound_ip_address' in df.columns and 'upper_bound_ip_address' in df.columns:
            invalid_ranges = (df['lower_bound_ip_address'] > df['upper_bound_ip_address']).sum()
            if invalid_ranges > 0:
                validation['issues'].append(f"{invalid_ranges} invalid IP ranges (lower > upper)")
        
        return validation

def load_datasets(data_dir: str = "../Data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to load all datasets.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Tuple of (fraud_data, creditcard_data, ip_country_mapping)
    """
    loader = FraudDataLoader(data_dir)
    return loader.load_all_datasets()

def get_dataset_summary(data_dir: str = "../Data") -> Dict:
    """Get comprehensive summary of all datasets.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dictionary containing dataset summaries
    """
    loader = FraudDataLoader(data_dir)
    info = loader.get_data_info()
    quality = loader.validate_data_quality()
    
    summary = {
        'info': info,
        'quality': quality,
        'recommendations': _generate_recommendations(quality)
    }
    
    return summary

def _generate_recommendations(quality_report: Dict) -> Dict[str, list]:
    """Generate data quality recommendations."""
    recommendations = {}
    
    for dataset, quality in quality_report.items():
        recs = []
        
        if quality.get('missing_values', 0) > 0:
            recs.append("Handle missing values before modeling")
        
        if quality.get('duplicate_records', 0) > 0:
            recs.append("Remove or investigate duplicate records")
        
        if quality.get('issues'):
            recs.extend([f"Address issue: {issue}" for issue in quality['issues']])
        
        # Dataset-specific recommendations
        if dataset == 'fraud':
            class_dist = quality.get('class_distribution', {})
            if class_dist and len(class_dist) == 2:
                fraud_rate = min(class_dist.values()) / sum(class_dist.values())
                if fraud_rate < 0.05:
                    recs.append("Consider sampling techniques for class imbalance")
        
        elif dataset == 'creditcard':
            class_dist = quality.get('class_distribution', {})
            if class_dist and len(class_dist) == 2:
                fraud_rate = min(class_dist.values()) / sum(class_dist.values())
                if fraud_rate < 0.01:
                    recs.append("Severe class imbalance - use advanced sampling methods")
        
        recommendations[dataset] = recs
    
    return recommendations