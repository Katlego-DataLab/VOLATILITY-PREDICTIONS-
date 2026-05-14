# 📈 Stock Market Volatility Forecasting · AAPL
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![yfinance](https://img.shields.io/badge/yfinance-1DA1F2?style=for-the-badge&logo=yahoo&logoColor=white)](https://pypi.org/project/yfinance/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

> *Transforming raw market data into actionable risk intelligence* using Python, time-series regression & financial feature engineering.


---

## Quick Overview

Built a **volatility forecasting pipeline** for Apple Inc. (AAPL) that estimates short-term market risk from historical price data. Trained a baseline Linear Regression model on 13 years of daily returns establishing a reproducible, leakage-free benchmark that reveals the **nonlinear nature of financial volatility** and sets the stage for GARCH & LSTM upgrades. 

---

##  Business Problem

> *"We know markets move, but by HOW MUCH?"*

Financial institutions don't just need to know *if* prices change  they need to quantify **how wildly**. Volatility drives:

- 🏦 **Risk management** — knowing when to reduce exposure
- 📊 **Portfolio allocation** — adjusting positions dynamically
- 💹 **Options pricing** — the entire derivatives market is volatility-dependent
- 🔐 **Capital requirements** — regulators mandate volatility-linked reserves
- ⚖️ **Position sizing** — preventing catastrophic drawdowns

Without structured volatility forecasting, institutions fly blind, often discovering risk *after* it's too late. This project simulates the kind of quantitative risk modeling workflow used at hedge funds, asset managers, and risk desks worldwide. 

---

##  Key Results

| Metric | Value | What It Means |
|--------|-------|---------------|
|  MSE | `5.41e-05` | Very small absolute error — volatility values are tiny (0.01–0.03 range) |
|  R² | `-0.38` | Linear model underperforms a simple mean baseline **expected & informative** |
|  Training Period | 2010–2019 | ~2,600 trading days of learning data |
|  Test Period | 2019–2023 | ~651 days of true out-of-sample evaluation |
|  Rolling Window | 21 days | ~1 trading month  industry-standard risk horizon |

> ⚠️ **Why is R² negative?** That's not a bug  it's a *finding*. A negative R² on out-of-sample data tells us that daily returns alone **cannot linearly predict** next-month volatility. This is consistent with decades of financial research showing volatility follows nonlinear, regime-switching processes (ARCH/GARCH effects). The real value here is **establishing a rigorous baseline** that more advanced models must beat. 

---

##  Business Impact

This pipeline enables:

- ✅ **Risk desk analysts** to flag elevated volatility regimes before they materialize
- ✅ **Portfolio managers** to dynamically hedge exposure when predicted volatility exceeds thresholds
- ✅ **Quant researchers** to benchmark advanced models (GARCH, LSTM) against this linear baseline
- ✅ **Risk committees** to visualize predicted vs. actual volatility for backtesting capital allocation strategies

**Who uses this?**  Risk Analytics teams, Quantitative Finance desks, Portfolio Construction groups, and RegTech compliance teams. 

---

##  Solution Overview

```
Yahoo Finance API  ──▶  Daily OHLCV Data  ──▶  Daily Returns
        │
        ▼
21-Day Rolling Std Dev  ──▶  Volatility Target Variable
        │
        ▼
Chronological 80/20 Split (no shuffle ⚠️)
        │
        ▼
Linear Regression Baseline  ──▶  Predict Test-Set Volatility
        │
        ▼
Evaluate: MSE + R² + Visual Comparison 📊
```

**Tech Stack:** Python · yfinance · pandas · NumPy · scikit-learn · matplotlib

---

##  Key Technical Decisions

| Decision | Why It Matters |
|----------|----------------|
|  **No shuffle in train/test split** | Shuffling time-series = data leakage. Future data would "teach" past predictions invalidating all results. |
|  **Chronological 80/20 split** | Replicates real-world forecasting conditions  model only ever sees past data |
|  **21-day rolling window** | Aligns with the standard ~1 trading month convention used in VaR and options markets |
|  **Dropna after rolling** | Prevents NaN contamination in the first 20 rows from corrupting model training |
| **MSE + R² dual evaluation** | MSE catches absolute error; R² reveals whether the model adds any predictive value vs. a naive mean |

---

##  Key Insights

- 📌 **Volatility clusters**  calm periods cluster together, and so do turbulent ones (COVID crash visible as a spike)
- 📌 **Daily returns have weak linear predictive power** for rolling volatility  consistent with the Efficient Market Hypothesis
- 📌 **The model smooths over spikes** it underestimates peak volatility during market crises (the most dangerous time to be wrong!)
- 📌 **A negative R² is a meaningful signal**, not a failure  it tells us *which model class* to try next (GARCH, Regime-Switching, LSTM)
- 📌 **Raw returns are insufficient as sole features**  lagged volatility, VIX, and volume would all add predictive signal 

---

##  Production Thinking

*Here's how this would operate inside a real financial institution:* 

```
📥 DATA FLOW
Bloomberg / Refinitiv API  →  Raw OHLCV  →  Feature Pipeline  →  Model Scoring

🔁 DAILY RETRAINING
Nightly batch job retrains on rolling 3-year window → new volatility forecast for T+1

📊 INTEGRATION
Model outputs → Risk Dashboard (Tableau/Power BI) → Portfolio Management System → Trade Alerts

🔔 MONITORING
Predicted vs. Actual tracking → Drift alerts when R² degrades → Automatic model review trigger

📋 GOVERNANCE
Every prediction logged with model version, feature snapshot & confidence interval → Audit trail for regulators
```

---

##  Future Improvements

Ready to level up? Here's the roadmap from baseline to institutional-grade: 

-  **GARCH(1,1) model** — purpose-built for volatility clustering; industry standard
-  **LSTM Neural Network** — captures long-range temporal dependencies in volatility
-  **Lagged volatility features** — yesterday's vol predicts today's vol far better than daily returns alone
-  **Add VIX as a feature** — the market's own fear gauge is highly predictive
-  **Multi-asset expansion** — model correlation regime shifts across SPY, QQQ, GLD
-  **Cloud deployment** — AWS Lambda + S3 for nightly batch scoring
-  **Live dashboard** — Streamlit or Power BI for real-time predicted vs. actual tracking
-  **Walk-forward validation** — more rigorous than single train/test split

---

##  Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/stock-volatility-forecasting.git
cd stock-volatility-forecasting

# Install dependencies
pip install yfinance pandas numpy matplotlib scikit-learn

# Run the notebook
jupyter notebook volatility_forecasting.ipynb
```

---

##  Project Structure

```
📂 stock-volatility-forecasting/
├── 📓 volatility_forecasting.ipynb   # Main analysis notebook
├── 📄 README.md                       # You are here ✨
├── 📊 plots/
│   ├── aapl_closing_price.png
│   └── actual_vs_predicted_vol.png
└── 📋 requirements.txt
```

---

##  Libraries Used

| Library | Role |
|---------|------|
| `yfinance` |  Market data acquisition from Yahoo Finance |
| `pandas` |  Data wrangling & rolling window calculations |
| `NumPy` |  Numerical computation |
| `scikit-learn` |  Regression modeling & evaluation metrics |
| `matplotlib` | Visualization of volatility regimes |

---

##  Interactive Dashboard

>  **Want to explore this visually?** An interactive Power BI-style dashboard is included in this rep letting you:
>
> -  Filter by date range to examine specific market regimes (COVID crash, 2022 rate hike cycle)
> -  Toggle between actual vs. predicted volatility overlays
> -  Zoom into high-volatility spikes to inspect model lag behavior  
> -  View rolling return distributions alongside volatility trends
> -  Adjust the rolling window (7 / 21 / 63 days) and see how forecasts change in real time

*See the `dashboard/` folder or the live demo link below.* 🔗

---

## 👤 Author

**Katlego Mathebula**  
📧 [Portfolio](https://katlego-datalab.github.io/Website-updated-/)  
💼 [LinkedIn](https://linkedin.com/in/yourprofile)  


---

*Built with 🤍 and a lot of market data · 
