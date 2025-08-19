# Model Performance Indicators

## 📊 Overview

This guide covers the essential metrics for evaluating hydrological prediction models. Understanding these metrics is crucial for assessing model reliability and comparing different approaches.

## 🎯 Why Performance Metrics Matter

- **Model Selection**: Choose the best model for your specific application
- **Validation**: Ensure your model generalizes well to unseen data
- **Improvement**: Identify areas where models need refinement
- **Communication**: Report model performance to stakeholders

## 📈 Core Metrics

### 1. Coefficient of Determination (R²)

#### Mathematical Formula

$$
R^2 = \left( \frac{ \sum_{i=1}^{n} (Q_{i}^{\mathrm{obs}} - \overline{Q^{\mathrm{obs}}})(Q_{i}^{\mathrm{sim}} - \overline{Q^{\mathrm{sim}}}) }{ \sqrt{ \sum_{i=1}^{n} (Q_{i}^{\mathrm{obs}} - \overline{Q^{\mathrm{obs}}})^2 } \cdot \sqrt{ \sum_{i=1}^{n} (Q_{i}^{\mathrm{sim}} - \overline{Q^{\mathrm{sim}}})^2 } } \right)^2
$$

Where:
- $Q_{i}^{\mathrm{obs}}$ = observed value at time step $i$
- $Q_{i}^{\mathrm{sim}}$ = simulated value at time step $i$
- $\overline{Q^{\mathrm{obs}}}$, $\overline{Q^{\mathrm{sim}}}$ = means of observed and simulated values
- $n$ = number of observations

#### Interpretation

| R² Value | Interpretation |
|----------|---------------|
| 1.0 | Perfect correlation |
| 0.8-1.0 | Very strong relationship |
| 0.6-0.8 | Strong relationship |
| 0.4-0.6 | Moderate relationship |
| 0.2-0.4 | Weak relationship |
| < 0.2 | Very weak or no relationship |

!!! info "Key Points"
    - R² measures the proportion of variance explained by the model
    - Values range from 0 to 1
    - Higher values indicate better linear correlation
    - Does not indicate whether predictions are biased

#### Python Implementation

```python
import numpy as np

def calculate_r2(observed, simulated):
    """
    Calculate R² between observed and simulated values
    """
    # Calculate correlation coefficient
    r = np.corrcoef(observed, simulated)[0, 1]
    
    # Square it to get R²
    r2 = r ** 2
    
    return r2
```

---

### 2. Nash-Sutcliffe Efficiency (NSE)

#### Mathematical Formula

$$
\mathrm{NSE} = 1 - \frac{\sum_{i=1}^{n} \left(Q_{i}^{\mathrm{obs}} - Q_{i}^{\mathrm{sim}}\right)^2}{\sum_{i=1}^{n} \left(Q_{i}^{\mathrm{obs}} - \overline{Q^{\mathrm{obs}}}\right)^2}
$$

#### Interpretation

| NSE Value | Model Performance |
|-----------|------------------|
| 1.0 | Perfect model |
| 0.75-1.0 | Very good performance |
| 0.65-0.75 | Good performance |
| 0.50-0.65 | Satisfactory performance |
| 0.0-0.50 | Unsatisfactory (but better than mean) |
| < 0 | Unacceptable (worse than using mean) |

!!! warning "Important"
    NSE < 0 means the observed mean is a better predictor than the model!

#### Python Implementation

```python
def calculate_nse(observed, simulated):
    """
    Calculate Nash-Sutcliffe Efficiency
    """
    # Calculate numerator (sum of squared errors)
    numerator = np.sum((observed - simulated) ** 2)
    
    # Calculate denominator (variance of observed)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    
    # Calculate NSE
    nse = 1 - (numerator / denominator)
    
    return nse
```

---

### 3. Percent Bias (PBIAS)

#### Mathematical Formula

$$
\mathrm{PBIAS} = 100 \times \frac{ \sum_{i=1}^{n} (Q_{i}^{\mathrm{obs}} - Q_{i}^{\mathrm{sim}}) }{ \sum_{i=1}^{n} Q_{i}^{\mathrm{obs}} }
$$

#### Interpretation

| PBIAS Value | Interpretation |
|-------------|---------------|
| 0% | No bias (perfect) |
| > 0% | Model underestimates |
| < 0% | Model overestimates |
| ±5% | Very good |
| ±10% | Good |
| ±15% | Satisfactory |
| > ±25% | Unsatisfactory |

!!! tip "Rule of Thumb"
    - For streamflow: PBIAS < ±10% is very good
    - For sediment: PBIAS < ±15% is very good
    - For nutrients: PBIAS < ±25% is acceptable

#### Python Implementation

```python
def calculate_pbias(observed, simulated):
    """
    Calculate Percent Bias
    """
    pbias = 100 * np.sum(observed - simulated) / np.sum(observed)
    return pbias
```

---

## 🔧 Complete Evaluation Function

Here's the comprehensive function used throughout this guide:

```python
import numpy as np

def evaluate_model(obs, sim):
    """
    Calculate R², NSE, and PBIAS between observed and simulated values.
    
    Parameters:
    -----------
    obs : array-like
        Observed values
    sim : array-like
        Simulated/predicted values
    
    Returns:
    --------
    dict : Dictionary containing R², NSE, and PBIAS
    
    Example:
    --------
    >>> obs = np.array([1.2, 2.3, 3.1, 4.5, 5.2])
    >>> sim = np.array([1.3, 2.1, 3.3, 4.2, 5.5])
    >>> results = evaluate_model(obs, sim)
    >>> print(f"R² = {results['R²']:.3f}")
    >>> print(f"NSE = {results['NSE']:.3f}")
    >>> print(f"PBIAS = {results['PBIAS']:.2f}%")
    """
    # Convert to numpy arrays
    obs = np.array(obs)
    sim = np.array(sim)
    
    # R² (Coefficient of Determination)
    r = np.corrcoef(obs, sim)[0, 1]
    r2 = r ** 2
    
    # NSE (Nash-Sutcliffe Efficiency)
    nse = 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
    
    # PBIAS (Percent Bias)
    pbias = 100 * np.sum(obs - sim) / np.sum(obs)
    
    return {'R²': r2, 'NSE': nse, 'PBIAS': pbias}
```

## 📊 Additional Metrics

### Mean Absolute Error (MAE)

$$
\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^{n} |Q_{i}^{\mathrm{obs}} - Q_{i}^{\mathrm{sim}}|
$$

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(observed, simulated)
```

### Root Mean Square Error (RMSE)

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (Q_{i}^{\mathrm{obs}} - Q_{i}^{\mathrm{sim}})^2}
$$

```python
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(observed, simulated))
```

## 📈 Visualization of Performance

### Scatter Plot with Metrics

```python
import matplotlib.pyplot as plt

def plot_performance(obs, sim, metrics):
    """
    Create a scatter plot with performance metrics
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(obs, sim, alpha=0.5, s=20)
    
    # 1:1 line
    min_val = min(obs.min(), sim.min())
    max_val = max(obs.max(), sim.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    
    # Add metrics as text
    textstr = f"R² = {metrics['R²']:.3f}\n"
    textstr += f"NSE = {metrics['NSE']:.3f}\n"
    textstr += f"PBIAS = {metrics['PBIAS']:.2f}%"
    
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Observed')
    ax.set_ylabel('Simulated')
    ax.set_title('Model Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

## 🎯 Performance Standards for Hydrological Models

### Monthly Time Step

| Performance Rating | NSE | R² | PBIAS |
|-------------------|-----|-----|-------|
| Very Good | > 0.75 | > 0.75 | < ±10% |
| Good | 0.65-0.75 | 0.65-0.75 | ±10-15% |
| Satisfactory | 0.50-0.65 | 0.50-0.65 | ±15-25% |
| Unsatisfactory | < 0.50 | < 0.50 | > ±25% |

### Daily Time Step

| Performance Rating | NSE | R² | PBIAS |
|-------------------|-----|-----|-------|
| Very Good | > 0.65 | > 0.70 | < ±15% |
| Good | 0.50-0.65 | 0.60-0.70 | ±15-20% |
| Satisfactory | 0.40-0.50 | 0.50-0.60 | ±20-30% |
| Unsatisfactory | < 0.40 | < 0.50 | > ±30% |

!!! note "Time Scale Matters"
    Daily models typically have lower performance metrics than monthly models due to higher temporal variability.

## ✅ Best Practices

1. **Use Multiple Metrics**: No single metric tells the complete story
2. **Consider Time Scale**: Adjust expectations based on temporal resolution
3. **Check Residuals**: Look for patterns in prediction errors
4. **Validate on Independent Data**: Always test on unseen data
5. **Report Uncertainty**: Include confidence intervals when possible

## 🚀 Next Steps

Now that you understand performance metrics:
- Apply them to evaluate your [Simple Linear Regression](../models/simple-linear-regression.md) model
- Use them to compare different models
- Learn about [Feature Engineering](feature-engineering.md) to improve performance

---

<div class="grid" markdown>

:material-arrow-left: [Data Import](../setup/data-import.md){ .md-button }

:material-arrow-right: [Feature Engineering](feature-engineering.md){ .md-button .md-button--primary }

</div>