import warnings
warnings.filterwarnings("ignore")

# utils.py — streamlined for IJRPR Stock Price Prediction paper

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# ------------------------------------------------------------
# Function: split_sequence
# Purpose: Create sequences of given length (lookback window)
# ------------------------------------------------------------
def split_sequence(sequence, n_steps, reshape_for_lstm=False):
    """
    Split a sequence into input/output pairs for time-series modeling.

    Parameters
    ----------
    sequence : array-like
        The full time series data (1D or 2D array).
    n_steps : int
        The number of time steps (lookback window).
    reshape_for_lstm : bool, default=False
        Whether to reshape output X into (samples, n_steps, 1) for LSTM models.

    Returns
    -------
    X : np.ndarray
        Input sequences
    y : np.ndarray
        Corresponding target values
    """
    sequence = np.array(sequence).flatten()
    X, y = [], []
    for i in range(len(sequence) - n_steps):
        X.append(sequence[i:i + n_steps])
        y.append(sequence[i + n_steps])
    X, y = np.array(X), np.array(y)

    if reshape_for_lstm:
        X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


# ------------------------------------------------------------
# Function: scale_data
# Purpose: Apply MinMax scaling to Adjusted Close prices
# ------------------------------------------------------------
def scale_data(data):
    """
    Scale Adjusted Close prices between 0 and 1 for model training.

    Parameters
    ----------
    data : array-like
        1D or 2D array of prices

    Returns
    -------
    scaled : np.ndarray
        Scaled data between 0 and 1
    scaler : MinMaxScaler
        Fitted scaler instance
    """
    arr = np.array(data).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(arr)
    return scaled, scaler


# ------------------------------------------------------------
# Function: calculate_rmse
# Purpose: Compute Root Mean Squared Error between predictions and true values
# ------------------------------------------------------------
def calculate_rmse(y_true, y_pred, print_result=True):
    """
    Compute RMSE (Root Mean Squared Error) for model evaluation.

    Parameters
    ----------
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
    print_result : bool
        Whether to print RMSE

    Returns
    -------
    rmse : float
        The computed RMSE
    """
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    if print_result:
        print(f"The root mean squared error is {rmse:.2f}.")
    return rmse