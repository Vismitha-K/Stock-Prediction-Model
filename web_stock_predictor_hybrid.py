import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from train_hybrid_model import train_hybrid_model
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression
from datetime import timedelta

st.set_page_config(page_title="Hybrid LR + LSTM Stock Predictor", layout="centered", page_icon="📊")

st.title("📊 Hybrid Linear Regression + LSTM Stock Price Predictor")

ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, RELIANCE.NS):", "AAPL").upper()
epochs = st.slider("Training Epochs (only if retraining):", 3, 20, 8)

if st.button("Run Prediction"):
    try:
        model_path = f"{ticker}_hybrid_lstm.keras"
        lstm_model = load_model(model_path)
        st.info("✅ Loaded saved model successfully.")
    except:
        st.warning(f"No saved model found for {ticker}. Training a new one...")
        _, lstm_model, _, _ = train_hybrid_model(ticker, epochs)

    # Fetch 6 months of data
    data = yf.download(ticker, period="6mo", interval="1d")
    if "Adj Close" not in data.columns:
        data["Adj Close"] = data["Close"]

    close_prices = data["Adj Close"].values.reshape(-1, 1)
    X_lr = np.arange(len(close_prices)).reshape(-1, 1)

    # Linear Regression on close prices
    lr_model = LinearRegression()
    lr_model.fit(X_lr, close_prices)
    lr_pred = lr_model.predict(X_lr)

    residuals = close_prices - lr_pred

    # LSTM input preparation
    residuals_scaled = (residuals - residuals.min()) / (residuals.max() - residuals.min())
    X_test = []
    for i in range(60, len(residuals_scaled)):
        X_test.append(residuals_scaled[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    res_pred = lstm_model.predict(X_test)
    res_pred_inv = res_pred * (residuals.max() - residuals.min()) + residuals.min()

    final_pred = lr_pred[60:] + res_pred_inv
    pred_dates = data.index[-len(final_pred):]

    # Plot Actual vs Predicted
    st.subheader(f"📈 Predicted Stock Prices for {ticker}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pred_dates, data["Adj Close"].values[-len(final_pred):], label="Actual")
    ax.plot(pred_dates, final_pred, label="Predicted", linestyle="--")
    ax.legend()
    st.pyplot(fig)

    # Future 7-day forecast (based on trend)
    future_days = 7
    last_index = X_lr[-1][0]
    future_indices = np.arange(last_index + 1, last_index + future_days + 1).reshape(-1, 1)
    future_lr_pred = lr_model.predict(future_indices)
    future_res_pred = res_pred_inv[-future_days:]
    future_final_pred = future_lr_pred + future_res_pred

    future_dates = [data.index[-1] + timedelta(days=i+1) for i in range(future_days)]
    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": future_final_pred.flatten()
    })

    # Recent historical table
    st.subheader("📘 Recent Historical Data (Last 7 Days)")
    st.dataframe(data[["Close", "Open", "High", "Low", "Volume"]].tail(7).style.format("{:.2f}"))

    # Future forecast table
    st.subheader("🔮 Predicted Future Prices (Next 7 Days)")
    st.dataframe(future_df.style.format({"Predicted Price": "{:.2f}"}))

    st.success("✅ Prediction Complete!")