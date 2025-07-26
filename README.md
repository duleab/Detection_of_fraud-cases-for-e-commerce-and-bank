# Fraud Detection in E-commerce and Credit Card Transactions

## Project Overview

This project aims to develop robust fraud detection models for both e-commerce and credit card transactions using advanced machine learning techniques. The project addresses the critical challenge of class imbalance while maintaining a balance between security and user experience.

## Business Context

As a data scientist at Adey Innovations Inc., this project focuses on:
- Improving fraud detection accuracy for e-commerce and banking transactions
- Implementing geolocation analysis and transaction pattern recognition
- Building explainable AI models for trust and regulatory compliance
- Balancing false positives and false negatives to optimize business outcomes

## Datasets

### 1. Fraud_Data.csv (E-commerce Transactions)
- **user_id**: Unique user identifier
- **signup_time**: User registration timestamp
- **purchase_time**: Transaction timestamp
- **purchase_value**: Transaction amount in USD
- **device_id**: Unique device identifier
- **source**: Traffic source (SEO, Ads, etc.)
- **browser**: Browser used for transaction
- **sex**: User gender (M/F)
- **age**: User age
- **ip_address**: Transaction IP address
- **class**: Target variable (1=fraud, 0=legitimate)

### 2. IpAddress_to_Country.csv (Geolocation Data)
- **lower_bound_ip_address**: IP range lower bound
- **upper_bound_ip_address**: IP range upper bound
- **country**: Corresponding country

### 3. creditcard.csv (Credit Card Transactions)
- **Time**: Seconds elapsed from first transaction
- **V1-V28**: PCA-transformed anonymized features
- **Amount**: Transaction amount in USD
- **Class**: Target variable (1=fraud, 0=legitimate)

## Project Structure

```
├── data/                          # Raw and processed datasets
├── notebooks/                     # Jupyter notebooks for analysis
├── src/                          # Source code modules
│   ├── data_preprocessing/       # Data cleaning and preprocessing
│   ├── feature_engineering/      # Feature creation and selection
│   ├── models/                   # ML model implementations
│   ├── evaluation/               # Model evaluation and metrics
│   └── utils/                    # Utility functions
├── results/                      # Model outputs and visualizations
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Key Challenges

1. **Class Imbalance**: Both datasets are highly imbalanced with very few fraud cases
2. **Feature Engineering**: Creating meaningful features from raw transaction data
3. **Model Interpretability**: Ensuring models are explainable for business stakeholders
4. **Real-time Performance**: Models must be efficient for production deployment

## Methodology

1. **Data Analysis & Preprocessing**
   - Exploratory data analysis
   - Data cleaning and quality assessment
   - Missing value handling
   - Outlier detection and treatment

2. **Feature Engineering**
   - Time-based features (time_since_signup, transaction_hour, etc.)
   - IP geolocation mapping
   - Device and browser pattern analysis
   - Transaction velocity features

3. **Model Development**
   - Baseline models (Logistic Regression, Random Forest)
   - Advanced models (XGBoost, LightGBM)
   - Deep learning approaches (Neural Networks)
   - Ensemble methods

4. **Imbalanced Data Handling**
   - SMOTE (Synthetic Minority Oversampling Technique)
   - ADASYN (Adaptive Synthetic Sampling)
   - Cost-sensitive learning
   - Threshold optimization

5. **Model Evaluation**
   - Precision, Recall, F1-Score
   - AUC-ROC and AUC-PR curves
   - Confusion matrices
   - Business impact metrics

6. **Model Interpretability**
   - SHAP (SHapley Additive exPlanations)
   - LIME (Local Interpretable Model-agnostic Explanations)
   - Feature importance analysis

## Installation & Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analysis notebooks in order:
   - `01_data_exploration.ipynb`
   - `02_feature_engineering.ipynb`


## Expected Outcomes

- High-performance fraud detection models for both datasets
- Comprehensive feature engineering pipeline
- Detailed model comparison and selection justification
- Explainable AI insights for business stakeholders
- Production-ready model deployment guidelines
