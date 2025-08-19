# Data Import & Preparation

## 📂 Dataset Structure

This guide uses a time series dataset containing hydrological and meteorological variables. Let's understand the data structure and learn how to import it properly.

### Expected Data Format

Your CSV file should have the following structure:

| Date | Rainfall | Tmax | Tmin | Discharge |
|------|----------|------|------|-----------|
| 1981-01-01 | 0.0 | 20.7 | 8.4 | 0.528 |
| 1981-01-02 | 12.2 | 17.9 | 11.2 | 0.528 |
| 1981-01-03 | 0.0 | 18.8 | 7.8 | 0.441 |
| ... | ... | ... | ... | ... |

!!! info "Variable Descriptions"
    - **Date**: Daily timestamps (YYYY-MM-DD format)
    - **Rainfall**: Daily precipitation in millimeters (mm)
    - **Tmax**: Maximum daily temperature in Celsius (°C)
    - **Tmin**: Minimum daily temperature in Celsius (°C)
    - **Discharge**: Stream discharge in cubic meters per second (m³/s)

## 🔄 Import Required Libraries

First, let's import all the necessary libraries for our analysis:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_ccf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.utils import plot_model
import warnings
import random

# Set random seeds for reproducibility
warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
```

!!! tip "Reproducibility"
    Setting random seeds ensures that your results are reproducible across different runs.

## 📥 Loading the Data

### Basic Data Import

```python
# Load the CSV file
df = pd.read_csv('Discharge_30years.csv', 
                 parse_dates=['Date'], 
                 index_col='Date')

# Sort by date (important for time series)
df = df.sort_values('Date')

# Display first 5 rows
df.head()
```

### Alternative File Paths

If your file is in a different location:

=== "Windows"
    ```python
    df = pd.read_csv(r'C:\Users\YourName\Documents\Discharge_30years.csv', 
                     parse_dates=['Date'], 
                     index_col='Date')
    ```

=== "Mac/Linux"
    ```python
    df = pd.read_csv('/home/username/documents/Discharge_30years.csv', 
                     parse_dates=['Date'], 
                     index_col='Date')
    ```

=== "Google Colab"
    ```python
    from google.colab import files
    uploaded = files.upload()
    df = pd.read_csv('Discharge_30years.csv', 
                     parse_dates=['Date'], 
                     index_col='Date')
    ```

## 🔍 Data Exploration

### Basic Information

```python
# Check data shape
print(f"Dataset shape: {df.shape}")
print(f"Number of records: {df.shape[0]}")
print(f"Number of variables: {df.shape[1]}")

# Data types
print("\nData types:")
print(df.dtypes)

# Statistical summary
print("\nStatistical Summary:")
df.describe()
```

### Check for Missing Values

```python
# Check missing values
missing = df.isnull().sum()
print("Missing values per column:")
print(missing)

# Visualize missing data
import matplotlib.pyplot as plt

if missing.sum() > 0:
    plt.figure(figsize=(10, 4))
    missing[missing > 0].plot(kind='bar')
    plt.title('Missing Values by Column')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("No missing values found!")
```

## 📊 Data Visualization

### Time Series Plot

```python
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

# Rainfall
axes[0].plot(df.index, df['Rainfall'], color='blue', alpha=0.7)
axes[0].set_ylabel('Rainfall (mm)')
axes[0].set_title('30 Years of Hydrological Data')
axes[0].grid(True, alpha=0.3)

# Temperature
axes[1].plot(df.index, df['Tmax'], color='red', alpha=0.7, label='Tmax')
axes[1].plot(df.index, df['Tmin'], color='orange', alpha=0.7, label='Tmin')
axes[1].set_ylabel('Temperature (°C)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Discharge
axes[2].plot(df.index, df['Discharge'], color='green', alpha=0.7)
axes[2].set_ylabel('Discharge (m³/s)')
axes[2].grid(True, alpha=0.3)

# Rainfall vs Discharge (scatter)
axes[3].scatter(df['Rainfall'], df['Discharge'], alpha=0.5, s=1)
axes[3].set_xlabel('Rainfall (mm)')
axes[3].set_ylabel('Discharge (m³/s)')
axes[3].set_title('Rainfall vs Discharge Relationship')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Distribution Analysis

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Rainfall distribution
axes[0, 0].hist(df['Rainfall'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Rainfall Distribution')
axes[0, 0].set_xlabel('Rainfall (mm)')
axes[0, 0].set_ylabel('Frequency')

# Temperature distribution
axes[0, 1].hist(df['Tmax'], bins=30, alpha=0.5, label='Tmax', color='red')
axes[0, 1].hist(df['Tmin'], bins=30, alpha=0.5, label='Tmin', color='blue')
axes[0, 1].set_title('Temperature Distribution')
axes[0, 1].set_xlabel('Temperature (°C)')
axes[0, 1].legend()

# Discharge distribution
axes[1, 0].hist(df['Discharge'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1, 0].set_title('Discharge Distribution')
axes[1, 0].set_xlabel('Discharge (m³/s)')

# Box plots
axes[1, 1].boxplot([df['Rainfall'], df['Tmax'], df['Tmin'], df['Discharge']], 
                    labels=['Rainfall', 'Tmax', 'Tmin', 'Discharge'])
axes[1, 1].set_title('Variable Distributions (Box Plot)')
axes[1, 1].set_ylabel('Values (normalized for comparison)')

plt.tight_layout()
plt.show()
```

## 🎯 Identify Inputs and Outputs

### For Simple Linear Regression (SLR)

```python
# SLR uses only one input variable
X_slr = df[['Rainfall']]  # Input: Rainfall only (DataFrame with shape (n, 1))
y_slr = df['Discharge']   # Output: Discharge (Series with shape (n,))

print(f"SLR Input shape: {X_slr.shape}")
print(f"SLR Output shape: {y_slr.shape}")
```

### For Multiple Linear Regression (MLR)

```python
# MLR uses multiple input variables
# Method 1: Explicit selection
X_mlr = df[['Rainfall', 'Tmax', 'Tmin']]  # All inputs
y_mlr = df['Discharge']                    # Output

# Method 2: Using iloc (by position)
X_mlr = df.iloc[:, :-1]  # All columns except last
y_mlr = df.iloc[:, -1]   # Last column

print(f"MLR Input shape: {X_mlr.shape}")
print(f"MLR Output shape: {y_mlr.shape}")
```

### Flexible Input Selection

```python
def select_features(df, target_columns, input_selection="all"):
    """
    Flexible function to select input and output features
    
    Parameters:
    -----------
    df : DataFrame
        The complete dataset
    target_columns : list
        List of column indices for output variables
    input_selection : str or list or int
        - "all": Use all columns except target
        - list: Use specific column indices
        - int: Use first N columns
    
    Returns:
    --------
    X : DataFrame of input features
    y : Series or DataFrame of output features
    """
    y = df.iloc[:, target_columns]
    
    if input_selection == "all":
        input_columns = [i for i in range(df.shape[1]) if i not in target_columns]
    elif isinstance(input_selection, list):
        input_columns = input_selection
    elif isinstance(input_selection, int):
        input_columns = list(range(input_selection))
    else:
        raise ValueError("input_selection must be 'all', a list, or an integer")
    
    X = df.iloc[:, input_columns]
    
    return X, y

# Example usage
X, y = select_features(df, target_columns=[4], input_selection="all")
```

## 🔧 Data Preprocessing

### Handle Missing Values

```python
# Option 1: Remove rows with missing values
df_clean = df.dropna()

# Option 2: Forward fill (use previous value)
df_filled = df.fillna(method='ffill')

# Option 3: Interpolation
df_interpolated = df.interpolate(method='linear')

# Option 4: Fill with mean (not recommended for time series)
df_mean_filled = df.fillna(df.mean())
```

### Remove Outliers (Optional)

```python
def remove_outliers(df, column, n_std=3):
    """Remove outliers beyond n standard deviations"""
    mean = df[column].mean()
    std = df[column].std()
    
    lower_bound = mean - n_std * std
    upper_bound = mean + n_std * std
    
    mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    return df[mask]

# Example: Remove discharge outliers
df_no_outliers = remove_outliers(df, 'Discharge', n_std=3)
print(f"Removed {len(df) - len(df_no_outliers)} outliers")
```

## 💾 Save Preprocessed Data

```python
# Save cleaned data
df.to_csv('Discharge_30years_cleaned.csv')

# Save without index
df.to_csv('Discharge_30years_cleaned.csv', index=False)

# Save as Excel
df.to_excel('Discharge_30years_cleaned.xlsx')

# Save as pickle (preserves data types)
df.to_pickle('Discharge_30years_cleaned.pkl')
```

## ✅ Data Quality Checklist

Before proceeding to modeling, ensure:

- [ ] Data is loaded correctly with proper date parsing
- [ ] Data is sorted chronologically
- [ ] Missing values are handled appropriately
- [ ] Outliers are identified and addressed if necessary
- [ ] Data types are correct (dates as datetime, numerics as float/int)
- [ ] Input and output variables are properly separated
- [ ] Data visualization confirms expected patterns

## 🚀 Next Steps

Now that your data is properly imported and prepared, you can proceed to:

1. [Performance Metrics](../fundamentals/performance-metrics.md) - Understanding model evaluation
2. [Feature Engineering](../fundamentals/feature-engineering.md) - Creating lag features
3. [Simple Linear Regression](../models/simple-linear-regression.md) - Building your first model

!!! success "Ready to Model?"
    Your data is now ready for analysis! The next sections will guide you through building predictive models.

---

<div class="grid" markdown>

:material-arrow-left: [Installation](installation.md){ .md-button }

:material-arrow-right: [Performance Metrics](../fundamentals/performance-metrics.md){ .md-button .md-button--primary }

</div>