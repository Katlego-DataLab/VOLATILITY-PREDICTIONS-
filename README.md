# VOLATILITY-PREDICTIONS-
This project aims to forecast stock market volatility using historical price data in order to estimate future levels of market uncertainty. Volatility, measured as the rolling standard deviation of returns.

tock Market Volatility Forecasting Using Linear Regression

Author: Katlego Mathebula

This project forecasts short-term stock market volatility using historical price data for AAPL (Apple Inc.). Volatility, measured as the rolling standard deviation of daily returns, represents the magnitude of price fluctuations and financial risk rather than price direction. The central question: Can historical price behavior help predict short-term volatility?

1. Project Overview

Historical daily closing prices for AAPL were collected using Yahoo Finance.

Daily returns were calculated, and 21-day rolling standard deviation was used to estimate monthly volatility.

A Linear Regression model was trained to predict volatility based on past returns.

Model evaluation included Mean Squared Error (MSE), R², and visual comparison of actual vs predicted volatility.

This project demonstrates how time-series analysis and regression modeling can transform raw market data into actionable insights for risk management.

2.Libraries Required
pip install yfinance pandas numpy matplotlib scikit-learn


Python libraries used in the project:

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

3. Workflow

Download historical data:

data = yf.download('AAPL', start='2010-01-01', end='2023-01-01')


Visualize closing prices to understand trends.

Calculate daily returns and 21-day rolling volatility:

data['Daily Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily Return'].rolling(window=21).std()
data = data.dropna()


Prepare features and target:

X = Daily Return

y = Volatility

Train-test split (chronological, 80/20):

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


Train Linear Regression model:

model = LinearRegression()
model.fit(X_train, y_train)


Predict and evaluate:

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


Visualize predictions vs actual volatility.

Model Evaluation

Mean Squared Error (MSE): Measures prediction error

R-squared (R²): Measures model fit
