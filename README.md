# Stock Market Volatility Forecasting Using Linear Regression

**Author:** Katlego Mathebula
**Tech Stack:** Python · pandas · NumPy · matplotlib · scikit-learn · yfinance
**Project Type:** Time-Series Risk Modeling
**Asset Modeled:** Apple Inc. (AAPL)

## Executive Summary

Financial markets are inherently uncertain. While price direction is difficult to predict, volatility — the magnitude of price fluctuations — can be modeled to estimate risk exposure.

This project builds a time-series regression model to forecast short-term volatility for Apple Inc. (AAPL) using historical price data.

Volatility is defined as the **21-day rolling standard deviation of daily returns**, serving as a proxy for monthly market risk.

The central research question:

> Can historical return behavior help predict future short-term volatility?


##  Business Problem

Investors, hedge funds, and financial institutions rely on volatility estimates for:

- Risk management
- Portfolio allocation
- Options pricing
- Capital requirement estimation
- Position sizing strategies

Accurate volatility forecasting helps institutions:

- Reduce exposure during high-risk regimes
- Adjust leverage dynamically
- Improve downside protection

This project simulates a quantitative risk modeling workflow.

##  Methodology Overview

The project follows a structured financial modeling pipeline:

1. Download historical price data
2. Compute daily returns
3. Estimate rolling volatility
4. Preserve chronological order (no shuffling)
5. Train a regression model
6. Evaluate out-of-sample performance
7. Compare actual vs predicted volatility

## Data Collection

Historical daily closing prices were retrieved using:

```python
import yfinance as yf
```

Data source: Yahoo Finance
Period: 2010–2023
Asset: AAPL

Chronological order was strictly preserved to maintain time-series integrity.

##  Volatility Construction

Volatility was calculated as:

```python
data['Daily Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily Return'].rolling(window=21).std()
```

Why 21 days?

- Represents approximately one trading month
- Common standard in financial risk modeling
- Captures short-term risk dynamics

Missing values introduced by rolling windows were removed to ensure clean modeling.


##  Feature & Target Design

- Feature (X): Daily Return
- Target (y): 21-Day Rolling Volatility

This approach tests whether immediate return behavior contains predictive information about short-term volatility.

## Train-Test Strategy

The dataset was split chronologically:

```python
train_test_split(..., shuffle=False)
```

80% Training
20% Testing

Why no shuffling?

Shuffling time-series data introduces look-ahead bias and data leakage.

This preserves real-world forecasting conditions.

##  Model Selection

Algorithm Used: **Linear Regression**

Why Linear Regression?

- Baseline interpretable model
- Tests linear dependency between returns and volatility
- Provides transparent coefficient interpretation
- Serves as a benchmark for more advanced models


##  Model Evaluation

Performance was evaluated using:

-  Mean Squared Error (MSE)
-  R-squared (R²)
-  Visual comparison of actual vs predicted volatility

Example Output:

-  MSE: 5.41e-05
-  R²: -0.38


## Interpretation of Results

The negative R² suggests:

-  Linear regression struggles to capture volatility dynamics
-  Volatility clustering is likely nonlinear
-  Financial volatility exhibits heteroskedastic behavior

This is consistent with financial theory, where volatility often follows nonlinear processes.

This demonstrates awareness that:

> A simple linear model may not be sufficient for financial volatility forecasting.

##  Visualization

The model compares predicted volatility against actual realized volatility to evaluate trend tracking ability and lag behavior.

This visual inspection helps assess:

- Volatility regime detection
- Responsiveness to spikes
- Forecast smoothing behavior

##  Financial Significance

Even with modest predictive performance, this project demonstrates:
- Time-series discipline
- Risk modeling fundamentals
- Financial return transformation
- Rolling window techniques
- Out-of-sample evaluation

These are foundational skills in:

- Quantitative finance
- Risk analytics
- Portfolio management
- Financial data science

## Libraries Used

```bash
pip install yfinance pandas numpy matplotlib scikit-learn
```

Core Libraries:

-  yfinance → Data acquisition
-  pandas → Data manipulation
-  NumPy → Numerical computation
-  matplotlib → Visualization
-  scikit-learn → Modeling & evaluation

##  Potential Improvements

To extend this project toward institutional-grade modeling:

-  Lagged return features
-  Rolling variance features
-  GARCH modeling
- LSTM neural networks
-  Feature engineering using technical indicators
-  Volatility regime classification
-  Multi-asset comparison

## Conclusion

This project demonstrates how raw financial market data can be transformed into a structured volatility forecasting framework.

It highlights:

- Time-series awareness
-  Avoidance of data leakage
-  Baseline model benchmarking
-  Risk-focused interpretation

Rather than overfitting for performance, the focus was on methodological correctness and financial reasoning.

