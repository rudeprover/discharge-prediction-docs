# Installation Guide

## 📦 Required Libraries

This project uses several powerful Python libraries for data analysis, machine learning, and deep learning.

### Libraries Overview

<div class="grid cards" markdown>

- :simple-pandas:{ .lg .middle } **Pandas**
    
    ---
    For handling and manipulating tabular data using DataFrames

- :simple-numpy:{ .lg .middle } **NumPy**
    
    ---
    Provides fast numerical operations and multi-dimensional arrays

- :material-chart-line:{ .lg .middle } **Matplotlib**
    
    ---
    Used to create static, animated, and interactive plots

- :simple-scikitlearn:{ .lg .middle } **Scikit-learn**
    
    ---
    A machine learning library with tools for modeling and evaluation

- :material-chart-box:{ .lg .middle } **Statsmodels**
    
    ---
    Enables statistical analysis and time series exploration

- :simple-tensorflow:{ .lg .middle } **TensorFlow**
    
    ---
    A powerful library for building and training deep learning models

- :simple-keras:{ .lg .middle } **Keras**
    
    ---
    High-level API within TensorFlow for fast neural network development

- :material-graph:{ .lg .middle } **Pydot & Graphviz**
    
    ---
    Used to visualize neural network architectures (optional)

</div>

## 🔧 Installation Methods

### Method 1: Using Conda (Recommended)

If you have Anaconda or Miniconda installed, this is the most straightforward method:

```bash
conda install -c conda-forge pandas numpy matplotlib scikit-learn statsmodels tensorflow pydot graphviz -y
```

!!! success "Why Conda?"
    Conda handles complex dependencies better, especially for TensorFlow and its GPU support.

### Method 2: Using pip

For those using standard Python installations:

```bash
# Upgrade pip first
pip install --upgrade pip

# Install core packages
pip install pandas numpy matplotlib scikit-learn statsmodels

# Install TensorFlow (CPU version)
pip install tensorflow

# Optional: For GPU support
pip install tensorflow-gpu

# Optional: For neural network visualization
pip install pydot graphviz
```

### Method 3: Using Requirements File

Create a `requirements.txt` file:

```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
tensorflow>=2.10.0
pydot>=1.4.0
graphviz>=0.20.0
```

Then install all at once:

```bash
pip install -r requirements.txt
```

## 🐍 Setting Up Virtual Environment

!!! tip "Best Practice"
    Always use a virtual environment to avoid package conflicts!

### Windows

```powershell
# Create virtual environment
python -m venv hydro_env

# Activate it
hydro_env\Scripts\activate

# Install packages
pip install -r requirements.txt

# To deactivate when done
deactivate
```

### macOS/Linux

```bash
# Create virtual environment
python3 -m venv hydro_env

# Activate it
source hydro_env/bin/activate

# Install packages
pip install -r requirements.txt

# To deactivate when done
deactivate
```

## ✅ Verify Installation

After installation, verify everything is working:

```python
import sys
print(f"Python version: {sys.version}")

# Test imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import statsmodels
import tensorflow as tf
from tensorflow import keras

# Print versions
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"Keras: {keras.__version__}")

# Test TensorFlow
print(f"TensorFlow GPU Available: {tf.config.list_physical_devices('GPU')}")
```

Expected output:
```
Python version: 3.9.x (or higher)
Pandas: 1.3.x
NumPy: 1.21.x
Scikit-learn: 1.0.x
TensorFlow: 2.10.x
Keras: 2.10.x
TensorFlow GPU Available: [] (or list of GPUs if available)
```

## 🚨 Troubleshooting

### Common Issues and Solutions

??? bug "ImportError: No module named 'tensorflow'"
    **Solution**: Ensure you've activated your virtual environment and installed TensorFlow:
    ```bash
    pip install tensorflow
    ```

??? bug "Graphviz not found"
    **Solution**: Graphviz requires system installation:
    
    **Windows**: Download from [Graphviz website](https://graphviz.org/download/)
    
    **macOS**: 
    ```bash
    brew install graphviz
    ```
    
    **Linux**:
    ```bash
    sudo apt-get install graphviz  # Ubuntu/Debian
    sudo yum install graphviz       # RHEL/CentOS
    ```

??? bug "Memory errors with large datasets"
    **Solution**: Consider using:
    - Smaller batch sizes in neural networks
    - Data chunking with pandas
    - Google Colab for free GPU access

## 💻 Alternative: Google Colab

If you prefer not to install locally, use Google Colab:

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Most libraries are pre-installed
4. For additional packages:
   ```python
   !pip install statsmodels pydot
   ```

## 🎯 Next Steps

Now that you have all the required libraries installed, proceed to:

- [Data Import](data-import.md) - Learn how to load and prepare your discharge data
- [Performance Metrics](../fundamentals/performance-metrics.md) - Understand model evaluation metrics

!!! note "GPU Support"
    For faster neural network training, consider setting up GPU support:
    - NVIDIA GPU with CUDA support
    - Install CUDA toolkit and cuDNN
    - Install `tensorflow-gpu` instead of `tensorflow`

---

<div class="grid" markdown>

:material-arrow-left: [Home](../index.md){ .md-button }

:material-arrow-right: [Data Import](data-import.md){ .md-button .md-button--primary }

</div>