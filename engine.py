# engine.py — Finalized for IJRPR Stock Price Prediction implementation

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import datetime
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from ml_pipeline.train import (
    load_data,
    preprocess_data,
    build_lstm_model,
    train_and_evaluate,
    plot_results
)

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_TICKER = "GOOG"
FUTURE_DAYS = 30
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print("=======================================")
    print("   STOCK PRICE PREDICTION USING LSTM   ")
    print("=======================================")

    ticker = input("Enter Stock Ticker (default: GOOG): ") or DEFAULT_TICKER
    print(f"\nFetching {ticker} stock data (2004–current) from Yahoo Finance...\n")

    # Step 1: Load Data
    df = load_data(ticker)
    print(f"Data Loaded: {len(df)} records from {df.index.min().date()} to {df.index.max().date()}\n")

    # Step 2: Preprocess Data
    x_train, y_train, scaler, scaled_data, train_size, test_data = preprocess_data(df)
    print("Data preprocessing completed (scaling + 70/30 split).")

    n_steps = x_train.shape[1]
    model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")

    # Step 3: Load or Train Model
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"\n✅ Found saved model and scaler for {ticker}. Loading them...\n")
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        retrain = False
    else:
        print(f"\n⚙️ No saved model found for {ticker}. Training new LSTM model...\n")
        model = build_lstm_model((n_steps, 1))
        retrain = True

    predictions, y_test, rmse, model = train_and_evaluate(
        model, x_train, y_train, test_data, scaler, train_size, ticker, retrain=retrain
    )

    if retrain:
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        print(f"\n✅ Model and scaler saved to {MODEL_DIR}\n")

    # Step 4: Visualize Results
    print("\nPlotting results...\n")
    plot_results(y_test, predictions, ticker)

    print("=======================================")
    print(f"Training Complete for {ticker}")
    print(f"RMSE: {rmse:.2f}")
    print("=======================================")

    # -----------------------------
    # ✅ FUTURE FORECAST SECTION
    # -----------------------------
    print(f"\nGenerating {ticker} future price forecast...")

    last_sequence = scaled_data[-n_steps:]
    future_predictions = []
    current_input = last_sequence.reshape(1, n_steps, 1)

    for _ in range(FUTURE_DAYS):
        next_pred = model.predict(current_input, verbose=0)
        future_predictions.append(next_pred[0, 0])
        next_pred_reshaped = next_pred.reshape(1, 1, 1)
        current_input = np.concatenate((current_input[:, 1:, :], next_pred_reshaped), axis=1)

    # ✅ Inverse scale predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # ✅ Generate correct date range (from today, not dataset end)
    today = datetime.date.today()
    future_dates = [today + datetime.timedelta(days=i) for i in range(1, FUTURE_DAYS + 1)]

    # ✅ Plot forecast
    plt.figure(figsize=(10, 5))
    plt.plot(df.index[-200:], df['Adj Close'][-200:], label='Past 200 Days (Actual)')
    plt.plot(future_dates, future_predictions, color='orange', linestyle='--', label=f'{FUTURE_DAYS}-Day Forecast')
    plt.title(f'{ticker} Stock Price Forecast ({FUTURE_DAYS} Days Ahead)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

    print(f"\n✅ Future prediction completed: {FUTURE_DAYS} days ahead for {ticker}.")