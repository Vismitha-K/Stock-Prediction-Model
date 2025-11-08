import streamlit as st
import numpy as np
import os
import datetime
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from ml_pipeline.train import load_data, preprocess_data, build_lstm_model, train_and_evaluate

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📊 Stock Price Prediction using LSTM")

# -----------------------------
# Sidebar Configuration
# -----------------------------
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="", placeholder="e.g. GOOG, AAPL, TSLA")
FUTURE_DAYS = st.sidebar.slider("Days to Forecast", 10, 90, 30)
force_retrain = st.sidebar.checkbox("Force Retrain Model")
run_forecast = st.sidebar.button("🚀 Start Prediction")

# -----------------------------
# Model Directory Setup
# -----------------------------
MODEL_DIR = os.path.join(os.getcwd(), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Helper: Load or Train Model
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_or_train_model(ticker: str, force_retrain=False):
    """Load model and scaler if available; else train and save."""
    model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")

    # ✅ Load if available and retrain not requested
    if os.path.exists(model_path) and os.path.exists(scaler_path) and not force_retrain:
        st.info(f"📁 Loaded existing model for **{ticker}**")
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler, False

    # ✅ Train new model
    st.warning(f"⚙️ Training new model for **{ticker}** (no cache found)...")
    df = load_data(ticker)
    x_train, y_train, scaler, scaled_data, train_size, test_data = preprocess_data(df)
    model = build_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, epochs=2, batch_size=1, verbose=1)

    # Save model and scaler
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    st.success(f"✅ Model & scaler saved at: `{MODEL_DIR}`")

    return model, scaler, True

# -----------------------------
# Main App Logic
# -----------------------------
if run_forecast:
    if not ticker:
        st.error("⚠️ Please enter a valid stock ticker (e.g., GOOG).")
    else:
        st.write(f"Fetching **{ticker}** data from Yahoo Finance...")
        df = load_data(ticker)
        st.success(f"✅ Data Loaded: {len(df)} records ({df.index.min().date()} → {df.index.max().date()})")

        # Load or train
        model, scaler, retrained = get_or_train_model(ticker, force_retrain)

        # Preprocess again for predictions
        x_train, y_train, _, scaled_data, train_size, test_data = preprocess_data(df)
        predictions, y_test, rmse, model = train_and_evaluate(
            model, x_train, y_train, test_data, scaler, train_size, ticker, retrain=False
        )

        # -----------------------------
        # Actual vs Predicted Plot
        # -----------------------------
        st.subheader("📈 Actual vs Predicted Prices")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test, color="blue", label="Actual Prices")
        ax.plot(predictions, color="red", label="Predicted Prices")
        ax.set_title(f"{ticker} Stock Price Prediction")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)

        # -----------------------------
        # Future Forecast Section
        # -----------------------------
        st.subheader("📅 Future Forecast")
        n_steps = x_train.shape[1]
        last_sequence = scaled_data[-n_steps:]
        future_preds = []
        current_input = last_sequence.reshape(1, n_steps, 1)

        for _ in range(FUTURE_DAYS):
            next_pred = model.predict(current_input, verbose=0)
            future_preds.append(next_pred[0, 0])
            next_pred_reshaped = next_pred.reshape(1, 1, 1)
            current_input = np.concatenate((current_input[:, 1:, :], next_pred_reshaped), axis=1)

        future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

        # ✅ Forecast from today instead of dataset end
        start_date = datetime.date.today()
        future_dates = [start_date + datetime.timedelta(days=i) for i in range(1, FUTURE_DAYS + 1)]

        # Plot forecast
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(df.index[-200:], df["Adj Close"][-200:], label="Past 200 Days (Actual)")
        ax2.plot(future_dates, future_preds, "--", color="orange", label=f"{FUTURE_DAYS}-Day Forecast")
        ax2.set_title(f"{ticker} {FUTURE_DAYS}-Day Stock Price Forecast")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price (USD)")
        ax2.legend()
        st.pyplot(fig2)

        st.success(f"✅ Forecast ready — {FUTURE_DAYS} days ahead | RMSE: {rmse:.2f}")
else:
    st.info("👈 Enter a stock ticker and click **Start Prediction** to begin.")