"""Project setup script for fraud detection project."""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)
    else:
        print(f"✓ Python {sys.version.split()[0]} detected")

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully")
    except subprocess.CalledProcessError:
        print("Error: Failed to install packages. Please install manually using:")
        print("pip install -r requirements.txt")
        sys.exit(1)

def verify_data_files():
    """Verify that required data files exist."""
    data_dir = Path("Data")
    required_files = [
        "Fraud_Data.csv",
        "creditcard.csv", 
        "IpAddress_to_Country.csv"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            print(f"✓ Found {file}")
        else:
            missing_files.append(file)
            print(f"✗ Missing {file}")
    
    if missing_files:
        print(f"\nError: Missing required data files: {missing_files}")
        print("Please ensure all data files are in the 'Data' directory.")
        return False
    
    return True

def verify_data_integrity():
    """Verify data file integrity and structure."""
    print("\nVerifying data file integrity...")
    
    try:
        # Check Fraud_Data.csv
        fraud_data = pd.read_csv("Data/Fraud_Data.csv")
        expected_fraud_cols = ['user_id', 'signup_time', 'purchase_time', 'purchase_value', 
                              'device_id', 'source', 'browser', 'sex', 'age', 'ip_address', 'class']
        
        if all(col in fraud_data.columns for col in expected_fraud_cols):
            print(f"✓ Fraud_Data.csv structure verified ({fraud_data.shape[0]:,} rows, {fraud_data.shape[1]} columns)")
        else:
            print("✗ Fraud_Data.csv has unexpected structure")
            return False
        
        # Check creditcard.csv
        credit_data = pd.read_csv("Data/creditcard.csv")
        if 'Class' in credit_data.columns and credit_data.shape[1] == 31:
            print(f"✓ creditcard.csv structure verified ({credit_data.shape[0]:,} rows, {credit_data.shape[1]} columns)")
        else:
            print("✗ creditcard.csv has unexpected structure")
            return False
        
        # Check IpAddress_to_Country.csv
        ip_data = pd.read_csv("Data/IpAddress_to_Country.csv")
        expected_ip_cols = ['lower_bound_ip_address', 'upper_bound_ip_address', 'country']
        
        if all(col in ip_data.columns for col in expected_ip_cols):
            print(f"✓ IpAddress_to_Country.csv structure verified ({ip_data.shape[0]:,} rows, {ip_data.shape[1]} columns)")
        else:
            print("✗ IpAddress_to_Country.csv has unexpected structure")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error reading data files: {str(e)}")
        return False

def create_jupyter_config():
    """Create Jupyter notebook configuration."""
    config_content = '''
# Jupyter Notebook Configuration for Fraud Detection Project

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)

# Plotting settings
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("Fraud Detection Project - Jupyter environment configured successfully!")
'''
    
    with open("jupyter_config.py", "w") as f:
        f.write(config_content)
    
    print("✓ Jupyter configuration created")

def create_quick_start_notebook():
    """Create a quick start notebook."""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Fraud Detection Project - Quick Start\n",
                    "\n",
                    "This notebook provides a quick start guide for the fraud detection project.\n",
                    "\n",
                    "## Project Structure\n",
                    "\n",
                    "- `notebooks/`: Jupyter notebooks for analysis\n",
                    "- `src/`: Source code modules\n",
                    "- `Data/`: Dataset files\n",
                    "- `results/`: Output files and models\n",
                    "\n",
                    "## Getting Started\n",
                    "\n",
                    "1. Run the data exploration notebook: `01_data_exploration.ipynb`\n",
                    "2. Perform feature engineering: `02_feature_engineering.ipynb`\n",
                    "3. Train models: `03_model_training.ipynb`\n",
                    "4. Evaluate models: `04_model_evaluation.ipynb`\n",
                    "5. Interpret results: `05_model_interpretation.ipynb`"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Import required libraries\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Import project utilities\n",
                    "import sys\n",
                    "sys.path.append('src')\n",
                    "\n",
                    "from utils.data_loader import FraudDataLoader\n",
                    "from utils.preprocessing import FraudDataPreprocessor\n",
                    "from utils.model_training import FraudModelTrainer\n",
                    "from utils.evaluation import ModelEvaluator\n",
                    "from utils.visualization import FraudVisualization\n",
                    "\n",
                    "print('All libraries imported successfully!')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Quick data overview\n",
                    "loader = FraudDataLoader()\n",
                    "\n",
                    "# Load datasets\n",
                    "fraud_data, credit_data, ip_data = loader.load_all_datasets()\n",
                    "\n",
                    "# Get basic information\n",
                    "fraud_info = loader.get_basic_info(fraud_data, 'Fraud_Data')\n",
                    "credit_info = loader.get_basic_info(credit_data, 'Credit_Data')\n",
                    "\n",
                    "print('Data loaded successfully!')\n",
                    "print(f'Fraud Data: {fraud_data.shape}')\n",
                    "print(f'Credit Data: {credit_data.shape}')\n",
                    "print(f'IP Mapping Data: {ip_data.shape}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Next Steps\n",
                    "\n",
                    "Now you can proceed with the detailed analysis notebooks:\n",
                    "\n",
                    "1. **Data Exploration**: Open `01_data_exploration.ipynb` to explore the datasets\n",
                    "2. **Feature Engineering**: Use `02_feature_engineering.ipynb` to create features\n",
                    "3. **Model Training**: Train models using `03_model_training.ipynb`\n",
                    "4. **Model Evaluation**: Evaluate performance with `04_model_evaluation.ipynb`\n",
                    "5. **Model Interpretation**: Understand models with `05_model_interpretation.ipynb`"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    import json
    with open("notebooks/00_quick_start.ipynb", "w") as f:
        json.dump(notebook_content, f, indent=2)
    
    print("✓ Quick start notebook created")

def print_project_summary():
    """Print project summary and next steps."""
    print("\n" + "="*60)
    print("FRAUD DETECTION PROJECT SETUP COMPLETE")
    print("="*60)
    
    print("\n📁 Project Structure:")
    print("   ├── Data/                    # Dataset files")
    print("   ├── notebooks/               # Jupyter notebooks")
    print("   │   ├── 00_quick_start.ipynb")
    print("   │   ├── 01_data_exploration.ipynb")
    print("   │   ├── 02_feature_engineering.ipynb")
    print("   │   ├── 03_model_training.ipynb")
    print("   │   ├── 04_model_evaluation.ipynb")
    print("   │   └── 05_model_interpretation.ipynb")
    print("   ├── src/                     # Source code")
    print("   │   └── utils/               # Utility modules")
    print("   ├── results/                 # Output files")
    print("   ├── requirements.txt         # Dependencies")
    print("   └── README.md               # Project documentation")
    
    print("\n🚀 Next Steps:")
    print("   1. Start Jupyter: jupyter notebook")
    print("   2. Open: notebooks/00_quick_start.ipynb")
    print("   3. Follow the analysis workflow")
    
    print("\n📊 Available Datasets:")
    print("   • Fraud_Data.csv - E-commerce fraud detection")
    print("   • creditcard.csv - Credit card fraud detection")
    print("   • IpAddress_to_Country.csv - IP geolocation mapping")
    
    print("\n🛠️ Key Features:")
    print("   • Comprehensive data preprocessing")
    print("   • Multiple ML algorithms with hyperparameter tuning")
    print("   • Advanced sampling techniques for imbalanced data")
    print("   • Model interpretation with SHAP")
    print("   • Business impact analysis")
    print("   • Interactive visualizations")
    
    print("\n💡 Tips:")
    print("   • Check README.md for detailed documentation")
    print("   • Use the utility modules in src/utils/ for reusable functions")
    print("   • Save your results in the results/ directory")
    
    print("\n" + "="*60)
    print("Happy fraud detecting! 🕵️‍♂️")
    print("="*60)

def main():
    """Main setup function."""
    print("Setting up Fraud Detection Project...")
    print("="*50)
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    if os.path.exists("requirements.txt"):
        install_requirements()
    else:
        print("Warning: requirements.txt not found. Please install packages manually.")
    
    # Verify data files
    if not verify_data_files():
        print("\nSetup incomplete due to missing data files.")
        return
    
    # Verify data integrity
    if not verify_data_integrity():
        print("\nSetup incomplete due to data integrity issues.")
        return
    
    # Create configuration files
    create_jupyter_config()
    
    # Create quick start notebook
    create_quick_start_notebook()
    
    # Print summary
    print_project_summary()

if __name__ == "__main__":
    main()