# 📊 Stock Price Prediction using LSTM

> *"Forecasting Stock Prices using Long Short-Term Memory Networks for Financial Decision Support"*  
> Demonstrates how deep learning can model complex temporal dependencies in financial time series data to enhance prediction accuracy.

---

## 🧠 Overview

This project implements a **stock price forecasting system** using **LSTM (Long Short-Term Memory) neural networks**, a deep learning architecture particularly effective for **time-series prediction**.  
It is an end-to-end pipeline built from research to deployment — fetching live stock data, preprocessing, training or loading models, and visualizing actual vs. predicted prices with future forecasts.

---

## 🚀 Key Features

- **Live Stock Data Fetching** — powered by [Yahoo Finance API](https://pypi.org/project/yfinance/).  
- **Automatic Model Caching** — previously trained models are reused to avoid retraining.  
- **LSTM-based Deep Learning** — captures long-term dependencies in price movements.  
- **Dynamic Forecasting** — predicts user-defined future days (e.g., 30, 60, 90).  
- **Interactive Streamlit Dashboard** — visualize historical trends and forecasts.  
- **Fully Configurable** — change tickers, forecast horizon, or retraining options.

---

## 🧩 Folder Structure

```

Stock-Price-Prediction-YFinance-LSTM-RNN/
│
├── ml_pipeline/
│   ├── train.py          # Data loading, model training, evaluation
│   ├── utils.py          # Helper functions for scaling and RMSE
│
├── models/               # Auto-generated folder for saved models
│
├── app.py                # Streamlit web app
├── engine.py             # CLI-based runner
├── requirements.txt      # Dependencies
└── README.md

````

---

## 🛠️ Setup & Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Vismitha-K/Stock-Price-Predictiongit
cd Stock-Price-Prediction-YFinance-LSTM-RNN
````

### 2️⃣ (Optional) Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # on Windows
source venv/bin/activate   # on Mac/Linux
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧮 Run Options

### ▶️ Command-Line Version

```bash
python engine.py
```

You’ll be prompted for a stock ticker and the script will:

* Fetch Yahoo Finance data dynamically
* Load or train the LSTM model
* Plot actual vs. predicted prices
* Forecast future prices (e.g., next 30 days)

---

### 💻 Streamlit Web App

```bash
streamlit run app.py
```

Then open your browser at:

```
http://localhost:8501
```

**Features:**

* Enter any stock ticker (`GOOG`, `AAPL`, `TSLA`, etc.)
* Adjust forecast days using a slider
* Choose whether to retrain or reuse existing models
* Interactive visualizations for both past and predicted prices

---

## 🧩 Tech Stack

* **Python 3.10+**
* **TensorFlow / Keras**
* **Scikit-learn**
* **Matplotlib**
* **Streamlit**
* **yFinance**

---

## 🧠 Future Work

* Integration of hybrid deep learning models (LSTM + CNN or GRU).
* Inclusion of technical indicators (RSI, MACD) as input features.
* Comparative evaluation with ARIMA and Transformer architectures.
* Streamlit cloud deployment for public demonstration.


