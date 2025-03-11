# Commodity Market Risk Analysis

## Advanced Machine Learning Framework for Price Prediction, Anomaly Detection, and Risk Assessment

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)
![Pandas](https://img.shields.io/badge/pandas-1.3%2B-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

This repository contains an advanced analytical framework for commodity markets that integrates deep learning price prediction, hybrid anomaly detection, and multidimensional risk assessment techniques. The project focuses on three key commodities: Crude Oil (CL=F), Gold (GC=F), and Natural Gas (NG=F).

## Project Overview

Commodity markets are inherently volatile due to macroeconomic forces, geopolitical events, and seasonal factors. This project develops a comprehensive framework to:

- Predict commodity price movements using state-of-the-art deep learning models
- Detect market anomalies through a hybrid statistical and machine learning approach
- Assess supply chain risks with commodity-specific evaluation frameworks
- Analyze cross-commodity relationships through dynamic correlation analysis
- Evaluate portfolio performance under different market conditions

## Key Features

### 1. Deep Learning Price Prediction

The project implements and compares two sophisticated deep learning architectures:

- **LSTM Model**: Specialized for capturing temporal dependencies and sequential patterns
- **Transformer Model**: Leverages self-attention mechanism for complex non-sequential patterns

Our results show that different models excel for different commodities:
- Transformer performs better for Crude Oil
- LSTM performs better for Gold and Natural Gas

### 2. Hybrid Anomaly Detection

A two-component system combining:

- **Statistical Rule-Based Detection**:
  - Price shock detection
  - Volatility spike identification
  - Trend break analysis

- **Machine Learning Enhancement**:
  - Autoencoder for unsupervised pattern learning
  - Random Forest classifier for future anomaly probability forecasting
  - Multi-horizon predictions (5, 10, and 20 days)

### 3. Supply Chain Risk Assessment

Comprehensive risk evaluation framework incorporating:
- Supply chain length
- Production flexibility
- Storage capacity
- Transportation dependencies
- Geopolitical sensitivity

### 4. Cross-Commodity Correlation Analysis

- Static correlation measurement across the entire period
- Dynamic rolling correlation to identify regime shifts
- Market state identification based on correlation patterns

## Results and Performance

### Price Prediction Accuracy (5-day Horizon)

| Commodity | Model | MAE | RMSE | MAPE |
|-----------|-------|-----|------|------|
| Crude Oil | LSTM | 4.53 | 5.59 | 5.59% |
| Crude Oil | Transformer | 3.45 | 4.41 | 4.23% |
| Gold | LSTM | 52.32 | 68.08 | 2.71% |
| Gold | Transformer | 69.03 | 78.65 | 3.50% |
| Natural Gas | LSTM | 0.21 | 0.25 | 7.39% |
| Natural Gas | Transformer | 0.34 | 0.42 | 11.86% |

### Anomaly Detection

- Crude Oil: 11 anomalies detected (1.15% of trading days)
- Gold: 5 anomalies detected (0.52% of trading days)
- Natural Gas: 14 anomalies detected (1.46% of trading days)

### Supply Chain Risk

| Commodity | Risk Level | Risk Score | Risk Trend |
|-----------|------------|------------|------------|
| Crude Oil | Medium | 58.24/100 | Decreasing |
| Gold | Low | 32.27/100 | Stable |
| Natural Gas | Medium | 50.65/100 | Decreasing |

## Installation and Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/commodity-market-risk-analysis.git
cd commodity-market-risk-analysis

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt

# Open the Jupyter notebook
jupyter notebook Commodity_Market_Risk_Analysis.ipynb
```

### Required Dependencies

```
pandas
numpy
matplotlib
seaborn
torch
scikit-learn
plotly
yfinance
jupyter
```

## Key Findings

1. **Model Performance Varies by Commodity**:
   - LSTM performs better for Gold (MAPE: 2.71%) and Natural Gas (MAPE: 7.39%)
   - Transformer performs better for Crude Oil (MAPE: 4.23%)
   - This suggests that different commodities exhibit different types of price patterns

2. **Anomaly Patterns**:
   - Gold shows fewest anomalies (0.52%) but high market impact
   - Natural Gas shows highest anomaly frequency (1.46%)
   - Crude Oil shows medium frequency (1.15%) but highest risk score (58.24/100)

3. **Dynamic Correlations**:
   - Commodity relationships are highly regime-dependent
   - Correlations fluctuate dramatically between -0.75 and +0.8
   - Correlation breakdowns often precede major market events

4. **Risk Management Implications**:
   - Equal-weighted portfolio strategy led to extreme drawdowns (-99.44%)
   - Highlights critical importance of anomaly-triggered risk reduction
   - Suggests need for dynamic allocation based on volatility and correlation regimes

## Future Work

1. **Adaptive Ensemble Learning**: Dynamically adjust model weights based on recent performance and detected market regimes

2. **Alternative Data Integration**: Incorporate satellite imagery for oil storage, social media sentiment, and futures curve structures

3. **Enhanced Risk Management**: Implement a continuous capital allocation optimizer with Conditional Value at Risk (CVaR) constraints

4. **Real-time Monitoring System**: Develop a dashboard for live anomaly detection and risk assessment

## License

This project is licensed under the MIT License - see the LICENSE file for details.