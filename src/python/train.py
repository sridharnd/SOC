import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import joblib # To save/load scaler
import sys
import tensorflow as tf

# Dynamically add the project root to sys.path
# train.py is in src/python, so its parent is src, and its parent's parent is the project root.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now, import modules relative to the project root or within the src.python "package"
from src.python.model import build_trading_model
from src.python.data_processor import load_and_process_data

# --- CRITICAL FIX FOR "type already registered" ---
# Access the already loaded trading_strategies module from sys.modules
# This ensures it's not loaded/registered a second time.
try:
    # If main.py has already loaded trading_strategies, it will be in sys.modules
    from trading_strategies import (
        run_rsi_strategy, run_macd_strategy, run_supertrend_strategy, Candle,
        calculate_rsi_series, calculate_macd_series, calculate_supertrend_series
    )
except ImportError:
    # This block should ideally not be hit if main.py is the entry point
    print("Warning: trading_strategies not found in sys.modules during train.py import. Attempting direct import.")
    from src.python.trading_strategies import (
        run_rsi_strategy, run_macd_strategy, run_supertrend_strategy, Candle,
        calculate_rsi_series, calculate_macd_series, calculate_supertrend_series
    )
# --- END CRITICAL FIX ---


# Constants for NN configuration
SEQUENCE_LENGTH = 60 # Number of past time steps to look at for each prediction
FUTURE_PREDICT_PERIOD = 1 # Predict 1 candle into the future for label generation
N_FEATURES = 4 # Number of features per timestep (e.g., Normalized_Close, RSI_Value, MACD_Line, Supertrend_Value)
OUTPUT_CLASSES = 3 # Buy (Long - [1,0,0]), Sell (Short - [0,1,0]), Hold (None - [0,0,1])

def calculate_indicators_as_features(candles):
    """
    Calculates raw indicator values (RSI, MACD, Supertrend lines) from Candle data
    to be used as continuous numerical features for the Neural Network.
    """
    # Using actual candle close for the first feature, will be normalized later
    close_prices_array = np.array([c.close for c in candles])

    # Calculate raw indicator series using C++ functions
    # RSI Series (default period is 14)
    rsi_series = np.array(calculate_rsi_series(candles, 14)) # EXPLICITLY PASS PERIOD

    # MACD Series (default fast_period=12, slow_period=26, signal_period=9)
    macd_line_series_cpp, macd_signal_series_cpp = calculate_macd_series(candles, 12, 26, 9) # EXPLICITLY PASS PERIODS
    macd_line_series = np.array(macd_line_series_cpp)
    macd_signal_series = np.array(macd_signal_series_cpp) # Can be used as an additional feature if N_FEATURES increases

    # Supertrend Series (default period=10, multiplier=3.0)
    supertrend_line_series_cpp, supertrend_trend_series_cpp = calculate_supertrend_series(candles, 10, 3.0) # EXPLICITLY PASS PERIODS
    supertrend_line_series = np.array(supertrend_line_series_cpp)
    # supertrend_trend_series can be converted to 0/1 if used as a categorical feature

    # Ensure all series have the same length as candles (handled by C++ functions filling warm-up)
    # Combine into a single feature array: [Close_Price, RSI_Value, MACD_Line, Supertrend_Value]
    features = []
    for i in range(len(candles)):
        features.append([
            close_prices_array[i],
            rsi_series[i],
            macd_line_series[i],
            supertrend_line_series[i]
        ])

    return np.array(features)

# Add a parameter for look-ahead window for labels
LOOK_AHEAD_WINDOW = 5 # How many candles into the future to look for profit/loss

def create_sequences_and_labels(features, candles, sequence_length, future_predict_period, profit_threshold):
    X, y = [], []
    # Ensure enough data for sequences AND look-ahead window for labels
    if len(features) < sequence_length + future_predict_period + LOOK_AHEAD_WINDOW:
        print(f"Warning: Not enough data ({len(features)}) for sequences and look-ahead window ({sequence_length} + {future_predict_period} + {LOOK_AHEAD_WINDOW}). Skipping sequence creation.")
        return np.array([]), np.array([])

    for i in range(len(features) - sequence_length - future_predict_period - LOOK_AHEAD_WINDOW + 1):
        sequence = features[i : i + sequence_length]
        # The label is based on future price movement *after* the prediction point
        # Prediction is for candle `i + sequence_length + future_predict_period - 1`
        # We look `LOOK_AHEAD_WINDOW` candles *after* this prediction point
        label_start_candle_index = i + sequence_length + future_predict_period - 1

        # Calculate future max/min close prices within the look-ahead window
        future_window_closes = [
            candles[j].close for j in range(label_start_candle_index + 1,
                                             min(label_start_candle_index + 1 + LOOK_AHEAD_WINDOW, len(candles)))
        ]

        if not future_window_closes: # Not enough future data
            continue

        current_close_of_sequence_end = candles[label_start_candle_index].close

        # Define label based on sustained future price change
        max_future_gain = max(0.0, max(c - current_close_of_sequence_end for c in future_window_closes)) / current_close_of_sequence_end
        max_future_loss = max(0.0, max(current_close_of_sequence_end - c for c in future_window_closes)) / current_close_of_sequence_end

        label = [0, 0, 1] # Default: Hold ([Buy, Sell, Hold])

        if max_future_gain > profit_threshold and max_future_gain > max_future_loss:
            label = [1, 0, 0] # Strong Buy signal
        elif max_future_loss > profit_threshold and max_future_loss > max_future_gain:
            label = [0, 1, 0] # Strong Sell signal
        # Else, it's a Hold (if neither gain nor loss met threshold, or if they're balanced)

        X.append(sequence)
        y.append(label)

    return np.array(X), np.array(y)


def train_model(training_candles, testing_candles, model_save_path):
    """
    Prepares data, trains the neural network model, and saves weights.
    """
    print("--- Generating features for training data ---")
    all_features_train = calculate_indicators_as_features(training_candles)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features_train = scaler.fit_transform(all_features_train)

    scaler_save_path = os.path.join(model_save_path, 'scaler.pkl')
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler saved to: {scaler_save_path}")

    X_train, y_train = create_sequences_and_labels(
        scaled_features_train, training_candles,
        SEQUENCE_LENGTH, FUTURE_PREDICT_PERIOD, profit_threshold=0.01
    )
    if X_train.size == 0:
        print("Not enough training data to create sequences. Training skipped.")
        return

    print(f"Generated {len(X_train)} training sequences.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    model = build_trading_model(input_shape=(SEQUENCE_LENGTH, N_FEATURES))

    checkpoint_filepath = os.path.join(model_save_path, 'nn_model_weights.weights.h5')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
        )

    print("--- Training Neural Network Model ---")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[model_checkpoint_callback, early_stopping_callback],
        verbose=1
    )
    print("Model training finished.")

    print(f"Best model weights saved/restored from: {checkpoint_filepath}")


if __name__ == "__main__":
    print("--- Starting train.py for NN model ---")
    try:
        training_candles = load_and_process_data(os.path.join(project_root, 'data', 'AAPL_training.csv'))
        testing_candles = load_and_process_data(os.path.join(project_root, 'data', 'AAPL_testing.csv'))
        print(f"Loaded {len(training_candles)} training candles for NN training.")
        print(f"Loaded {len(testing_candles)} testing candles for NN evaluation.")

        model_save_dir = os.path.join(project_root, 'models')
        os.makedirs(model_save_dir, exist_ok=True)

        train_model(training_candles, testing_candles, model_save_dir)

    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure data files are present.")
        print(f"You can create dummy data by running 'python {os.path.join(project_root, 'src', 'python', 'data_processor.py')}' from the project root.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

    print("--- train.py finished ---")