import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------- Helper: Compute Technical Indicators ----------
def add_indicators(df):
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df["RSI"] = 100 - (100 / (1 + RS))
    df.fillna(method="bfill", inplace=True)
    return df

# ---------- Main Function ----------
def train_hybrid_model(ticker, epochs=8):
    print(f"\n📈 Downloading data for {ticker} ...")
    data = yf.download(ticker, period="5y", interval="1d")

    if "Adj Close" not in data.columns:
        data["Adj Close"] = data["Close"]

    data = add_indicators(data)
    features = ["Adj Close", "SMA_10", "EMA_10", "RSI"]
    df = data[features].dropna()

    close_values = df["Adj Close"].values.reshape(-1, 1)

    # ---------- Linear Regression ----------
    X_lr = np.arange(len(close_values)).reshape(-1, 1)
    y_lr = close_values
    lr_model = LinearRegression()
    lr_model.fit(X_lr, y_lr)
    lr_pred = lr_model.predict(X_lr)

    # ---------- Residuals ----------
    residuals = y_lr - lr_pred

    # ---------- Scale residuals ----------
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_res = scaler.fit_transform(residuals)

    # ---------- LSTM sequence preparation ----------
    X_train, y_train = [], []
    for i in range(60, len(scaled_res)):
        X_train.append(scaled_res[i-60:i, 0])
        y_train.append(scaled_res[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # ---------- Build LSTM ----------
    lstm_model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer="adam", loss="mean_squared_error")
    lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)

    # ---------- Predict on training data ----------
    res_pred = lstm_model.predict(X_train)
    res_pred_inv = scaler.inverse_transform(res_pred)

    # ---------- Combine predictions ----------
    final_pred = lr_pred[60:] + res_pred_inv
    true_values = y_lr[60:]

    rmse = np.sqrt(mean_squared_error(true_values, final_pred))
    mae = mean_absolute_error(true_values, final_pred)
    r2 = r2_score(true_values, final_pred)

    print(f"\n✅ Hybrid LR+LSTM Model trained successfully for {ticker}")
    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    # ---------- Save models ----------
    lstm_model.save(f"{ticker}_hybrid_lstm.keras")
    np.save(f"{ticker}_lr_coef.npy", lr_model.coef_)
    np.save(f"{ticker}_lr_intercept.npy", lr_model.intercept_)
    np.save(f"{ticker}_scaler.npy", scaler.data_max_)

    return lr_model, lstm_model, scaler, data