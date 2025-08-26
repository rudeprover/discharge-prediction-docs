# 💧 Discharge Prediction with Python - Interactive Documentation

[![MkDocs](https://img.shields.io/badge/MkDocs-Material-blue)](https://squidfunk.github.io/mkdocs-material/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An interactive MkDocs-based documentation site for learning hydrological discharge prediction using Python, featuring Simple Linear Regression, Multiple Linear Regression, and Artificial Neural Networks.

🌐 **Live Site**: [https://rudeprover.github.io/discharge-prediction-docs/](https://rudeprover.github.io/discharge-prediction-docs/)

## 📚 Features

- **Interactive Code Examples** - Copy and run code snippets with one click
- **Step-by-Step Tutorials** - From data import to advanced neural networks
- **Mathematical Foundations** - Understand the theory behind each model
- **Performance Metrics** - Learn R², NSE, and PBIAS for model evaluation
- **Feature Engineering** - Master lag features and cross-correlation
- **Resource Library** - Discover powerful time series forecasting libraries
- **Dark/Light Theme** - Toggle between themes for comfortable reading
- **Mobile Responsive** - Access tutorials on any device

## 🚀 Quick Start

### Option 1: View Online
Visit the live documentation at: https://rudeprover.github.io/discharge-prediction-docs/

### Option 2: Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/rudeprover/discharge-prediction-docs.git
   cd discharge-prediction-docs
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Serve locally**
   ```bash
   mkdocs serve
   ```
   Visit http://localhost:8000 in your browser

### Option 3: Deploy Your Own

1. Fork this repository
2. Update `mkdocs.yml` with your GitHub username
3. Enable GitHub Pages in repository settings
4. Push changes to trigger automatic deployment

## 📂 Project Structure

```
discharge-prediction-docs/
├── mkdocs.yml                 # MkDocs configuration
├── docs/                       # Documentation content
│   ├── index.md               # Home page
│   ├── setup/                 # Installation & data import guides
│   │   ├── installation.md
│   │   └── data-import.md
│   ├── fundamentals/          # Core concepts
│   │   ├── performance-metrics.md
│   │   └── feature-engineering.md
│   ├── models/                # Model implementations
│   │   ├── simple-linear-regression.md
│   │   ├── multiple-linear-regression.md
│   │   └── artificial-neural-network.md
│   ├── resources/             # Additional resources
│   │   └── libraries.md
│   └── assets/                # CSS, JS, and images
│       ├── css/
│       ├── js/
│       └── images/
├── requirements.txt           # Python dependencies
└── .github/
    └── workflows/
        └── ci.yml            # GitHub Actions for auto-deployment
```

## 🛠️ Technologies Used

- **MkDocs** - Static site generator
- **Material for MkDocs** - Beautiful theme with many features
- **Python** - For hydrological modeling examples
- **GitHub Pages** - Free hosting
- **GitHub Actions** - Automated deployment

## 📖 Documentation Sections

### 1. **Setup**
- Installing required libraries (pandas, numpy, scikit-learn, TensorFlow)
- Loading and preparing discharge data
- Data visualization techniques

### 2. **Fundamentals**
- Performance metrics (R², NSE, PBIAS)
- Feature engineering with lag variables
- Cross-correlation analysis

### 3. **Models**
- **Simple Linear Regression**: Basic rainfall-discharge relationships
- **Multiple Linear Regression**: Incorporating multiple variables
- **Artificial Neural Networks**: Deep learning for complex patterns

### 4. **Resources**
- Time series forecasting libraries (Prophet, Darts, statsmodels)
- Hydrology-specific tools
- Learning resources and tutorials

## 🎯 Learning Objectives

By following this documentation, you will:
- ✅ Understand hydrological prediction fundamentals
- ✅ Implement regression models from scratch
- ✅ Build and train neural networks for time series
- ✅ Evaluate model performance using appropriate metrics
- ✅ Apply feature engineering techniques
- ✅ Discover advanced forecasting libraries

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original Jupyter notebook content adapted for interactive documentation
- Inspired by hydrological modeling best practices
- Built with the amazing MkDocs Material theme

## 📧 Contact

[@Zuhail Abdullah](https://linkedin.com/in/zuhail)
[@Dr.Harsh Upadhyay]([https://linkedin.com/in/zuhail](https://www.linkedin.com/in/dr-harsh-upadhyay-893726189/))

Project Link: [https://github.com/rudeprover/discharge-prediction-docs](https://github.com/rudeprover/discharge-prediction-docs)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=rudeprover/discharge-prediction-docs&type=Date)](https://star-history.com/#rudeprover/discharge-prediction-docs&Date)

---

<p align="center">Made with ❤️ for the Hydrology Community</p>
