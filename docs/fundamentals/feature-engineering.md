# Feature Engineering for Hydrological Models

## 🎯 Overview

Feature engineering is the process of creating new input variables (features) from existing data to improve model performance. In hydrology, this often involves capturing temporal dependencies and seasonal patterns.

## 🌊 Why Feature Engineering Matters in Hydrology

Hydrological systems have **memory**:
- Rainfall today doesn't immediately become runoff
- Soil moisture affects future discharge
- Groundwater responds slowly to precipitation
- Temperature influences evapotranspiration with delays

By engineering features that capture these relationships, we can significantly improve model predictions.

## 📊 Cross-Correlation Analysis

### Understanding Cross-Correlation Function (CCF)

The CCF measures how past values of one variable relate to current values of another.

#### Mathematical Formula

$$
\text{CCF}(k) = \frac{\sum_{t} (x_{t+k} - \bar{x})(y_t - \bar{y})}{\sqrt{\sum_{t} (x_{t+k} - \bar{x})^2 \sum_{t} (y_t - \bar{y})^2}}
$$

Where:
- $x_t$ = Independent variable (e.g., rainfall) at time $t$
- $y_t$ = Dependent variable (e.g., discharge) at time $t$
- $k$ = Lag value
- $\bar{x}$, $\bar{y}$ = Means of respective variables

#### Significance Threshold

$$
\text{Significance Limit} = \pm \frac{1.96}{\sqrt{N}}
$$

Where $N$ is the number of observations.

### Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_ccf

def analyze_cross_correlation(df, max_lags=12):
    """
    Compute and plot cross-correlation between inputs and discharge
    
    Parameters:
    -----------
    df : DataFrame
        Data with columns for inputs and 'Discharge'
    max_lags : int
        Maximum number of lags to analyze
    
    Returns:
    --------
    dict : Significant lags for each variable
    """
    # Separate inputs and output
    inputs = df.drop('Discharge', axis=1)
    output = df['Discharge']
    
    # Store significant lags
    significant_lags = {}
    
    # Create subplots
    fig, axes = plt.subplots(len(inputs.columns), 1, 
                             figsize=(10, 3*len(inputs.columns)))
    
    if len(inputs.columns) == 1:
        axes = [axes]
    
    # Analyze each input variable
    for i, col in enumerate(inputs.columns):
        # Remove mean (required for CCF)
        var1 = inputs[col] - inputs[col].mean()
        var2 = output - output.mean()
        
        # Plot CCF
        plot_ccf(var1, var2, lags=max_lags, ax=axes[i])
        axes[i].set_title(f'CCF: {col} vs Discharge')
        axes[i].set_xlabel('Lag (days)')
        axes[i].set_ylabel('Correlation')
        
        # Find significant lags (simplified approach)
        n = len(var1)
        significance_level = 1.96 / np.sqrt(n)
        
        # You can extract significant lags programmatically here
        # For now, we'll note them visually from the plot
        
    plt.tight_layout()
    plt.show()
    
    return significant_lags

# Example usage
df = pd.read_csv('Discharge_30years.csv', parse_dates=['Date'], index_col='Date')
analyze_cross_correlation(df)
```

## 🔄 Lag Features

### What are Lag Features?

Lag features are past values of variables used as inputs. They help models "remember" previous conditions.

### Example: Creating Lag Features

| Date | Rainfall | Rainfall_lag1 | Rainfall_lag2 | Rainfall_lag3 |
|------|----------|---------------|---------------|---------------|
| Day 1 | 10.5 | NaN | NaN | NaN |
| Day 2 | 5.2 | 10.5 | NaN | NaN |
| Day 3 | 0.0 | 5.2 | 10.5 | NaN |
| Day 4 | 15.3 | 0.0 | 5.2 | 10.5 |
| Day 5 | 8.7 | 15.3 | 0.0 | 5.2 |

### Implementation

```python
def create_lag_features(df, variable_lags):
    """
    Create lagged features for specified variables
    
    Parameters:
    -----------
    df : DataFrame
        Original data
    variable_lags : dict
        Dictionary mapping variable names to list of lag values
        Example: {'Rainfall': [1, 2, 3], 'Tmax': [1, 2]}
    
    Returns:
    --------
    DataFrame : Data with added lag features
    """
    df_features = df.copy()
    
    for variable, lags in variable_lags.items():
        for lag in lags:
            df_features[f'{variable}_lag{lag}'] = df[variable].shift(lag)
    
    return df_features

# Example usage
df = pd.read_csv('Discharge_30years.csv', parse_dates=['Date'], index_col='Date')

# Define lags based on CCF analysis
lag_config = {
    'Rainfall': [1, 2, 3],
    'Tmax': [1, 2, 3],
    'Tmin': [1, 2, 3]
}

# Create features
df_with_lags = create_lag_features(df, lag_config)

# Remove rows with NaN (from lagging)
df_clean = df_with_lags.dropna()

print(f"Original shape: {df.shape}")
print(f"After adding lags: {df_with_lags.shape}")
print(f"After removing NaN: {df_clean.shape}")
print("\nNew features created:")
print([col for col in df_with_lags.columns if 'lag' in col])
```

## 📈 Rolling Statistics

### Moving Averages and Sums

Rolling statistics smooth out short-term fluctuations and highlight longer-term trends.

```python
def add_rolling_features(df, windows=[3, 7, 14, 30]):
    """
    Add rolling statistics as features
    
    Parameters:
    -----------
    df : DataFrame
        Original data
    windows : list
        Window sizes for rolling calculations
    
    Returns:
    --------
    DataFrame : Data with rolling features
    """
    df_features = df.copy()
    
    for window in windows:
        # Rolling mean
        df_features[f'Rainfall_mean_{window}d'] = (
            df['Rainfall'].rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling sum (cumulative rainfall)
        df_features[f'Rainfall_sum_{window}d'] = (
            df['Rainfall'].rolling(window=window, min_periods=1).sum()
        )
        
        # Rolling max temperature
        df_features[f'Tmax_max_{window}d'] = (
            df['Tmax'].rolling(window=window, min_periods=1).max()
        )
        
        # Rolling standard deviation (variability)
        df_features[f'Rainfall_std_{window}d'] = (
            df['Rainfall'].rolling(window=window, min_periods=1).std()
        )
    
    return df_features

# Example usage
df_rolling = add_rolling_features(df)
```

## 🗓️ Temporal Features

### Extracting Time-Based Information

```python
def add_temporal_features(df):
    """
    Add temporal features from date index
    
    Parameters:
    -----------
    df : DataFrame
        Data with DatetimeIndex
    
    Returns:
    --------
    DataFrame : Data with temporal features
    """
    df_features = df.copy()
    
    # Basic temporal features
    df_features['day_of_year'] = df.index.dayofyear
    df_features['month'] = df.index.month
    df_features['quarter'] = df.index.quarter
    df_features['week_of_year'] = df.index.isocalendar().week
    
    # Cyclical encoding for month (captures seasonality)
    df_features['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # Cyclical encoding for day of year
    df_features['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df_features['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    
    # Season indicator
    seasons = {1: 'Winter', 2: 'Winter', 3: 'Spring', 
               4: 'Spring', 5: 'Spring', 6: 'Summer',
               7: 'Summer', 8: 'Summer', 9: 'Fall', 
               10: 'Fall', 11: 'Fall', 12: 'Winter'}
    df_features['season'] = df.index.month.map(seasons)
    
    # One-hot encode season
    season_dummies = pd.get_dummies(df_features['season'], prefix='season')
    df_features = pd.concat([df_features, season_dummies], axis=1)
    
    return df_features

# Example usage
df_temporal = add_temporal_features(df)
```

## 🧮 Interaction Features

### Creating Feature Combinations

```python
def add_interaction_features(df):
    """
    Create interaction features between variables
    """
    df_features = df.copy()
    
    # Rainfall-Temperature interaction
    df_features['Rain_Temp_interaction'] = df['Rainfall'] * df['Tmax']
    
    # Temperature range
    df_features['Temp_range'] = df['Tmax'] - df['Tmin']
    
    # Antecedent Precipitation Index (API)
    # Weighted sum of past rainfall
    weights = np.array([0.5, 0.3, 0.2])  # Decreasing weights
    for i, w in enumerate(weights, 1):
        if f'Rainfall_lag{i}' in df_features.columns:
            if i == 1:
                df_features['API'] = w * df_features[f'Rainfall_lag{i}']
            else:
                df_features['API'] += w * df_features[f'Rainfall_lag{i}']
    
    return df_features
```

## 🔬 Domain-Specific Features

### Hydrological Indices

```python
def add_hydrological_features(df):
    """
    Add hydrology-specific features
    """
    df_features = df.copy()
    
    # Baseflow index (using simple filter)
    # This is a simplified approach
    alpha = 0.925  # Filter parameter
    baseflow = [df['Discharge'].iloc[0]]
    
    for i in range(1, len(df)):
        bf = alpha * baseflow[-1] + (1 - alpha) * df['Discharge'].iloc[i]
        baseflow.append(min(bf, df['Discharge'].iloc[i]))
    
    df_features['Baseflow'] = baseflow
    df_features['Quickflow'] = df['Discharge'] - df_features['Baseflow']
    
    # Antecedent discharge (previous day's discharge)
    df_features['Discharge_lag1'] = df['Discharge'].shift(1)
    
    # Rate of change in discharge
    df_features['Discharge_change'] = df['Discharge'].diff()
    
    # Cumulative rainfall over season
    df_features['Cumulative_rainfall'] = df.groupby(df.index.year)['Rainfall'].cumsum()
    
    return df_features
```

## 📊 Complete Feature Engineering Pipeline

```python
def complete_feature_engineering(df):
    """
    Complete feature engineering pipeline for hydrological data
    """
    print("Starting feature engineering...")
    print(f"Original shape: {df.shape}")
    
    # Step 1: Create lag features
    lag_config = {
        'Rainfall': [1, 2, 3],
        'Tmax': [1, 2],
        'Tmin': [1, 2],
        'Discharge': [1, 2]  # Auto-regressive component
    }
    df_features = create_lag_features(df, lag_config)
    print(f"After lag features: {df_features.shape}")
    
    # Step 2: Add rolling statistics
    df_features = add_rolling_features(df_features, windows=[3, 7, 14])
    print(f"After rolling features: {df_features.shape}")
    
    # Step 3: Add temporal features
    df_features = add_temporal_features(df_features)
    print(f"After temporal features: {df_features.shape}")
    
    # Step 4: Add interaction features
    df_features = add_interaction_features(df_features)
    print(f"After interaction features: {df_features.shape}")
    
    # Step 5: Add hydrological features
    df_features = add_hydrological_features(df_features)
    print(f"After hydrological features: {df_features.shape}")
    
    # Step 6: Remove rows with NaN
    df_clean = df_features.dropna()
    print(f"After removing NaN: {df_clean.shape}")
    
    # Step 7: Separate features and target
    target = df_clean['Discharge']
    features = df_clean.drop(['Discharge', 'season'], axis=1)  # Remove non-numeric
    
    print(f"\nFinal features shape: {features.shape}")
    print(f"Final target shape: {target.shape}")
    print(f"\nTotal features created: {len(features.columns)}")
    
    return features, target

# Run the complete pipeline
features, target = complete_feature_engineering(df)
```

## ✅ Feature Selection

### Importance-Based Selection

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def select_important_features(features, target, n_features=20):
    """
    Select most important features using Random Forest
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': features.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot top features
    plt.figure(figsize=(10, 6))
    plt.barh(importances['feature'][:n_features], 
             importances['importance'][:n_features])
    plt.xlabel('Importance')
    plt.title(f'Top {n_features} Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    # Return top features
    top_features = importances['feature'][:n_features].tolist()
    return top_features, importances

# Select top features
top_features, importance_df = select_important_features(features, target)
```

## 🎯 Best Practices

1. **Start Simple**: Begin with basic lag features before complex transformations
2. **Domain Knowledge**: Use hydrological understanding to guide feature creation
3. **Avoid Leakage**: Don't use future information in features
4. **Handle Missing Data**: Decide strategy before creating features
5. **Scale Features**: Normalize/standardize for neural networks
6. **Validate Impact**: Test if new features actually improve performance
7. **Document Features**: Keep track of what each feature represents

## ⚠️ Common Pitfalls

!!! warning "Avoid These Mistakes"
    - **Data Leakage**: Using future values to predict past (shuffle=False for time series!)
    - **Too Many Features**: Can lead to overfitting (curse of dimensionality)
    - **Highly Correlated Features**: Remove redundant features
    - **Not Checking Feature Distributions**: Outliers can dominate models
    - **Ignoring Physical Constraints**: Features should make hydrological sense

## 📊 Feature Visualization

```python
def visualize_features(df_features, target_col='Discharge', n_features=6):
    """
    Visualize relationships between features and target
    """
    import seaborn as sns
    
    # Select features to plot
    feature_cols = [col for col in df_features.columns if col != target_col][:n_features]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(feature_cols):
        axes[idx].scatter(df_features[col], df_features[target_col], 
                         alpha=0.5, s=10)
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel(target_col)
        axes[idx].set_title(f'{col} vs {target_col}')
        
        # Add trend line
        z = np.polyfit(df_features[col].dropna(), 
                      df_features.loc[df_features[col].notna(), target_col], 1)
        p = np.poly1d(z)
        axes[idx].plot(df_features[col].sort_values(), 
                      p(df_features[col].sort_values()), 
                      "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.show()

# Visualize feature relationships
visualize_features(df_with_lags)
```

## 🚀 Next Steps

Now that you've mastered feature engineering:

1. Apply these features to [Multiple Linear Regression](../models/multiple-linear-regression.md)
2. Use them in [Artificial Neural Networks](../models/artificial-neural-network.md)
3. Experiment with different lag configurations
4. Try feature selection techniques

## 📚 Additional Resources

- [Time Series Feature Engineering](https://www.kaggle.com/learn/time-series)
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Hydrological Feature Engineering Papers](https://scholar.google.com/scholar?q=feature+engineering+hydrology)

---

<div class="grid" markdown>

:material-arrow-left: [Performance Metrics](performance-metrics.md){ .md-button }

:material-arrow-right: [Simple Linear Regression](../models/simple-linear-regression.md){ .md-button .md-button--primary }

</div>