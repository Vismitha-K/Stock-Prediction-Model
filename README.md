# 📊 Hybrid LSTM + Linear Regression Stock Price Prediction

A machine learning project that forecasts future stock prices using a **hybrid model combining LSTM and Linear Regression**.  
The model learns both **sequential trends** (via LSTM) and **linear relationships** (via Linear Regression) from historical stock data and technical indicators like **SMA, EMA, and RSI**.

---

## 🚀 Project Overview

This project aims to predict the **next 7 days of stock prices** based on past historical data.  
It uses a **hybrid deep learning and statistical approach**, combining the **pattern-recognition capability of LSTM** with the **trend-fitting ability of Linear Regression**.

The system automatically:
- Takes a **stock ticker symbol** (e.g., `AAPL`, `TSLA`, `RELIANCE.NS`)
- Fetches historical data using **Yahoo Finance API (`yfinance`)**
- Computes technical indicators (**SMA, EMA, RSI**)
- Trains a **hybrid forecasting model**
- Displays both **predicted and actual trends** via interactive visualization

---

## 🎯 Objectives

- Build a predictive model for short-term stock price forecasting  
- Use **SMA**, **EMA**, and **RSI** indicators for feature enhancement  
- Combine deep learning (LSTM) and statistical regression (Linear Regression)  
- Ensure the system dynamically adapts to **any stock ticker**  
- Visualize trends and future predictions interactively

---

## 🧠 Motivation / Inspiration

Stock markets are highly dynamic and influenced by complex patterns.  
This project was inspired by the idea of combining **statistical learning** and **deep learning** to capture both **short-term volatility** and **long-term trends**.

> “I wanted to understand how preprocessing and feature engineering can significantly influence model accuracy and interpretability — especially in real-world financial data.”

---

## ⚙️ Tech Stack

| Category | Technologies Used |
|-----------|-------------------|
| **Programming Language** | Python |
| **Data Fetching** | yfinance |
| **Data Processing & Analysis** | Pandas, NumPy |
| **Feature Engineering** | SMA, EMA, RSI (using Pandas rolling windows) |
| **Modeling** | TensorFlow / Keras (LSTM), scikit-learn (Linear Regression) |
| **Scaling** | MinMaxScaler |
| **Visualization** | Matplotlib, Streamlit |
| **Interface** | Streamlit Web App |
| **Deployment-ready Format** | `.h5` model and dynamic retraining |

---

## 🧩 Model Architecture

### 1️⃣ **LSTM Component**
- Input: 60-day rolling window of features (`Close`, `SMA`, `EMA`, `RSI`)  
- Learns sequential patterns and temporal dependencies  
- Output: Future price predictions

### 2️⃣ **Linear Regression Component**
- Input: Latest engineered features (`SMA`, `EMA`, `RSI`)  
- Captures linear trend and direction  
- Output: Complementary trend signal

### 3️⃣ **Hybrid Prediction**
- Final output is the **weighted combination** of LSTM and Linear Regression predictions  
- Improves both **stability** and **accuracy**

---

## 📈 Feature Engineering Details

| Feature | Description |
|----------|-------------|
| **SMA (Simple Moving Average)** | Shows average trend over a defined period; used to identify price direction |
| **EMA (Exponential Moving Average)** | Gives more weight to recent prices for responsiveness |
| **RSI (Relative Strength Index)** | Measures price momentum — helps detect overbought or oversold conditions |

Additional preprocessing steps:
- Handled missing data using forward-fill
- Scaled all features to 0–1 range using **MinMaxScaler**
- Aligned feature data with correct prediction targets
- Used rolling 60-day sequences to avoid time leakage

---

## 🔍 Data Source

- **Yahoo Finance API** via `yfinance`
- Example:  
  ```python
  import yfinance as yf
  data = yf.download("AAPL", start="2015-01-01", end="2024-12-31")
  ```
  
---
  
## 🧰 Installation & Setup

# Clone the repository
```bash
git clone https://github.com/Vismitha-K/Stock-Prediction-Model.git
cd stock-price-hybrid
```

# Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # For Windows
source venv/bin/activate  # For macOS/Linux
```

# Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

# Run Streamlit App
```bash
streamlit run web_stock_predictor_hybrid.py
```

# In the app:
Enter a stock ticker symbol (e.g., AAPL, TSLA, RELIANCE.NS)
Choose number of epochs for training (if retraining is required)

# View:
Historical price trend
Predicted future price trend
Table of recent and predicted values

---

## 📊 Results & Visualization

Training window: 60 past days
Prediction horizon: 7 future days
Performance Metrics:
- R² ≈ 0.88
- RMSE and MAE for model evaluation
The visualization shows:
- Historical prices (blue)
- Predicted trend (red)
Smooth transition across actual and forecasted data

---

## 🎯 Learning Outcomes

- Understood the importance of data preprocessing in ML pipelines
- Learned to handle real-world time-series challenges
- Gained hands-on experience in hybrid model development
- Improved understanding of feature scaling, windowing, and evaluation metrics

---

## 📚 Future Improvements

- Include additional indicators like MACD, Bollinger Bands, Volatility Index (VIX)
- Implement GRU or Transformer-based architectures for better sequential learning
- Add real-time stock data streaming
- Deploy app using Streamlit Cloud or AWS
