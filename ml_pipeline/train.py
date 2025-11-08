import warnings
warnings.filterwarnings("ignore")

import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_TICKER = "GOOG"
START_DATE = "2004-01-01"
LOOKBACK = 100  # window size for sequences
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Step 1: Load data
# -----------------------------
def load_data(ticker):
    """Download and clean stock data."""
    print(f"Fetching data for {ticker} from Yahoo Finance...")
    df = yf.download(ticker, start=START_DATE, end=datetime.now().strftime("%Y-%m-%d"))

    # Handle MultiIndex issue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    df.dropna(inplace=True)
    return df


# -----------------------------
# Step 2: Preprocess data
# -----------------------------
def preprocess_data(df):
    """Scale, split, and sequence the data."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[["Adj Close"]])

    train_size = int(len(scaled_data) * 0.7)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - LOOKBACK:]

    x_train, y_train = [], []
    for i in range(LOOKBACK, len(train_data)):
        x_train.append(train_data[i - LOOKBACK:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler, scaled_data, train_size, test_data


# -----------------------------
# Step 3: Build LSTM model
# -----------------------------
def build_lstm_model(input_shape):
    """Two-layer LSTM model."""
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# -----------------------------
# Step 4: Train and evaluate (persistent)
# -----------------------------
def train_and_evaluate(model, x_train, y_train, test_data, scaler, train_size, ticker, retrain=True):
    """Train model if needed and evaluate test set."""
    model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.keras")

    # Load model if available and retrain not requested
    if os.path.exists(model_path) and not retrain:
        print(f"📁 Loaded existing model from {model_path}")
        model = load_model(model_path)
    else:
        print("\nTraining LSTM model (2 epochs, batch_size=1)...\n")
        model.fit(x_train, y_train, batch_size=1, epochs=2, verbose=1)
        model.save(model_path)
        print(f"✅ Model saved to {model_path}")

    # Prepare test data
    x_test, y_test = [], []
    for i in range(LOOKBACK, len(test_data)):
        x_test.append(test_data[i - LOOKBACK:i, 0])
        y_test.append(test_data[i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = np.array(y_test)

    # Predict and inverse transform
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
    print(f"\nRMSE: {rmse:.2f}")

    return predictions, y_test_scaled, rmse, model


# -----------------------------
# Step 5: Plot results
# -----------------------------
def plot_results(y_test, predictions, ticker):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"{ticker} Stock Price Prediction (LSTM)")
    ax.plot(y_test, label="Actual Price", color="blue")
    ax.plot(predictions, label="Predicted Price", color="red")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    return fig


# -----------------------------
# CLI entrypoint
# -----------------------------
if __name__ == "__main__":
    ticker = input("Enter Stock Ticker (default: GOOG): ") or DEFAULT_TICKER
    df = load_data(ticker)
    x_train, y_train, scaler, scaled_data, train_size, test_data = preprocess_data(df)
    model = build_lstm_model((x_train.shape[1], 1))
    predictions, y_test, rmse, model = train_and_evaluate(model, x_train, y_train, test_data, scaler, train_size, ticker)
    plt = plot_results(y_test, predictions, ticker)
    plt.show()