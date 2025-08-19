# Discharge Prediction with Python

![Python Logo](assets/images/Python-Logo.jpg)
*The foundation of scientific computing in this project.*

---

## Welcome to the Hydrological Prediction Guide

This interactive documentation demonstrates step-by-step methodologies for making hydrological predictions using Python. Whether you're a student, researcher, or practitioner in hydrology, this guide will help you understand and implement various prediction models.

## 📚 What You'll Learn

This comprehensive guide covers:

- **Simple Linear Regression (SLR)** - Understanding basic relationships between rainfall and discharge
- **Multiple Linear Regression (MLR)** - Incorporating multiple variables for improved predictions
- **Artificial Neural Networks (ANN)** - Using deep learning for complex hydrological patterns

## 🎯 Key Features

<div class="grid cards" markdown>

- :material-chart-line:{ .lg .middle } **Performance Metrics**

    ---

    Learn about R², NSE, and PBIAS for model evaluation

    [:octicons-arrow-right-24: Learn more](fundamentals/performance-metrics.md)

- :material-cog:{ .lg .middle } **Feature Engineering**

    ---

    Master lag features and cross-correlation analysis

    [:octicons-arrow-right-24: Explore](fundamentals/feature-engineering.md)

- :material-brain:{ .lg .middle } **Neural Networks**

    ---

    Build and train ANNs for discharge prediction

    [:octicons-arrow-right-24: Get started](models/artificial-neural-network.md)

- :material-library:{ .lg .middle } **Resources**

    ---

    Discover powerful time series forecasting libraries

    [:octicons-arrow-right-24: Browse](resources/libraries.md)

</div>

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Basic understanding of hydrology
- Familiarity with Python programming

### Installation

```bash
# Create a virtual environment
python -m venv hydro_env

# Activate it
# On Windows:
hydro_env\Scripts\activate
# On Mac/Linux:
source hydro_env/bin/activate

# Install required packages
pip install pandas numpy matplotlib scikit-learn statsmodels tensorflow
```

### Sample Dataset

This guide uses a 30-year discharge dataset with the following variables:
- **Date**: Daily timestamps
- **Rainfall**: Daily precipitation (mm)
- **Tmax**: Maximum temperature (°C)
- **Tmin**: Minimum temperature (°C)
- **Discharge**: Stream discharge (m³/s)

!!! tip "Download Sample Data"
    You can download the sample dataset `Discharge_30years.csv` from our [GitHub repository](https://github.com/YOUR_USERNAME/discharge-prediction-docs/tree/main/sample_data).

## 📊 Interactive Code Examples

All code examples in this documentation are interactive and can be copied with a single click. Try it out:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your discharge data
df = pd.read_csv('Discharge_30years.csv', 
                 parse_dates=['Date'], 
                 index_col='Date')

# Quick visualization
df['Discharge'].plot(figsize=(12, 4))
plt.title('30 Years of Discharge Data')
plt.ylabel('Discharge (m³/s)')
plt.show()
```

## 🎓 Learning Path

We recommend following this learning path:

1. **Start with Setup**: Install necessary libraries and import your data
2. **Understand Fundamentals**: Learn about performance metrics and feature engineering
3. **Build Models**: Progress from simple to complex models
4. **Explore Resources**: Discover advanced libraries for your projects

## 🤝 Contributing

Found an issue or want to contribute? Visit our [GitHub repository](https://github.com/YOUR_USERNAME/discharge-prediction-docs) to:
- Report issues
- Suggest improvements
- Submit pull requests

## 📝 Citation

If you use this guide in your research, please cite:

```bibtex
@online{discharge_prediction_2024,
  author = {Your Name},
  title = {Discharge Prediction with Python: An Interactive Guide},
  year = {2024},
  url = {https://YOUR_USERNAME.github.io/discharge-prediction-docs/}
}
```

---

!!! success "Ready to Start?"
    Head over to the [Installation Guide](setup/installation.md) to begin your journey in hydrological prediction!

## 📧 Contact

For questions or feedback, please reach out through:
- GitHub Issues: [Create an issue](https://github.com/YOUR_USERNAME/discharge-prediction-docs/issues)
- Email: your.email@example.com