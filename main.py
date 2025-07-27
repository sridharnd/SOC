import os
import sys

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import joblib # To load scaler
import numpy as np # For array manipulation
import pandas as pd # For saving results to CSV

# --- DEBUGGING OUTPUT ---
print("--- DEBUG: Script Start ---")
print(f"--- DEBUG: Current Working Directory: {os.getcwd()} ---")

# Correct calculation of project_root (main.py is in the project root)
project_root = os.path.dirname(os.path.abspath(__file__))

# The directory containing trading_strategies.so and data_processor.py
src_python_path = os.path.join(project_root, 'src', 'python')

# Add src/python to sys.path
if src_python_path not in sys.path:
    sys.path.insert(0, src_python_path)

print(f"--- DEBUG: Calculated Project Root: {project_root} ---")
print(f"--- DEBUG: src/python path added to sys.path: {src_python_path} ---")
print(f"--- DEBUG: Current sys.path (first entry): {sys.path[0]} ---")
sys.stdout.flush()

try:
    print("--- DEBUG: Attempting to import data_processor and trading_strategies ---")
    sys.stdout.flush()
    # Import data_processor as a top-level module from src/python
    from data_processor import load_and_process_data

    # trading_strategies is also directly importable from src/python path
    from trading_strategies import TradeResult, run_rsi_strategy, run_macd_strategy, run_supertrend_strategy

    # Import NN model builder and feature generation from train.py
    from train import build_trading_model, calculate_indicators_as_features, create_sequences_and_labels

    print("--- DEBUG: Successfully imported data_processor and trading_strategies ---")
    sys.stdout.flush()
except ImportError as e:
    print(f"--- CRITICAL ERROR: ImportError during initial imports: {e} ---")
    print(f"  Check if 'trading_strategies.cpython-3x-darwin.so' is in '{src_python_path}'.")
    print(f"  Current sys.path (full): {sys.path}")
    sys.stdout.flush()
    sys.exit(1)
except Exception as e:
    print(f"--- CRITICAL ERROR: General Exception during initial imports: {e} ---")
    sys.stdout.flush()
    sys.exit(1)

print("--- DEBUG: All initial imports successful. Starting main logic... ---")
sys.stdout.flush()

# Constants for NN strategy (should match train.py)
NN_SEQUENCE_LENGTH = 60
NN_N_FEATURES = 4
NN_OUTPUT_CLASSES = 3 # Buy, Sell, Hold
FUTURE_PREDICT_PERIOD = 1


def run_nn_strategy(candles, profit_threshold, model_path, scaler_path,
                    rsi_func, macd_func, supertrend_func):
    """
    Runs the Neural Network trading strategy.
    1. Loads the trained model and scaler.
    2. Generates features from historical candles.
    3. Scales features using the loaded scaler.
    4. Creates sequences for prediction.
    5. Predicts signals using the NN model.
    6. Executes trades based on predicted signals.
    """
    print("\n--- Running Neural Network Strategy ---")

    try:
        # Load the scaler
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from: {scaler_path}")

        # Load the model weights
        model = build_trading_model(input_shape=(NN_SEQUENCE_LENGTH, NN_N_FEATURES))
        model.load_weights(model_path)
        print(f"Model weights loaded from: {model_path}")

        # Generate features for prediction (using the same logic as training)
        all_features = calculate_indicators_as_features(candles)
        scaled_features = scaler.transform(all_features) # Use transform, not fit_transform!

        # Create sequences for prediction (labels are not needed here, but function expects them)
        X_test, _ = create_sequences_and_labels(
            scaled_features, candles,
            NN_SEQUENCE_LENGTH, future_predict_period=FUTURE_PREDICT_PERIOD, profit_threshold=0.0
        )
        if X_test.size == 0:
            print("Not enough testing data to create sequences for NN prediction.")
            return {"success_rate": 0.0, "per_trade_return": 0.0, "total_trades": 0, "positions": [0]*len(candles)}


        print(f"Generated {len(X_test)} testing sequences for NN prediction.")
        print(f"X_test shape: {X_test.shape}")

        # Make predictions
        predictions = model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1) # Get the index of the highest probability class

        # Map predicted class index to signal (0=Buy, 1=Sell, 2=Hold for one-hot [1,0,0], [0,1,0], [0,0,1])
        signal_map = {0: 1, 1: -1, 2: 0} # 1: Buy (Long), -1: Sell (Short), 0: Hold

        nn_positions = [0] * len(candles) # Store positions for the entire candle dataset (initialized to 0)
        profitable_trades = 0
        total_return = 0.0
        total_trades = 0

        current_position_state = 0 # 0: None, 1: Long, -1: Short
        entry_price = 0.0
        entry_index = -1

        # Iterate through predictions and simulate trading
        # Predictions align with the *end* of sequences.
        # The prediction for sequence `i` is for candle `i + NN_SEQUENCE_LENGTH + FUTURE_PREDICT_PERIOD - 1`
        # We need to map prediction index back to candle index to align with original candles
        for i in range(len(predicted_classes)):
            candle_index_for_prediction = i + NN_SEQUENCE_LENGTH + FUTURE_PREDICT_PERIOD - 1

            if candle_index_for_prediction >= len(candles):
                continue # Should not happen if X_test is built correctly

            predicted_signal = signal_map[predicted_classes[i]] # 1:Buy, -1:Sell, 0:Hold
            current_candle_close = candles[candle_index_for_prediction].close
            nn_positions[candle_index_for_prediction] = predicted_signal # Store the raw signal for plotting

            # Trading logic based on predicted signal
            if current_position_state == 0: # No current position
                if predicted_signal == 1: # Predicted Buy
                    current_position_state = 1 # Go Long
                    entry_price = current_candle_close
                    entry_index = candle_index_for_prediction
                elif predicted_signal == -1: # Predicted Sell
                    current_position_state = -1 # Go Short
                    entry_price = current_candle_close
                    entry_index = candle_index_for_prediction
            elif current_position_state == 1: # Currently Long
                if predicted_signal == -1 or predicted_signal == 0: # Predicted Sell or Hold (Exit Long)
                    exit_price = current_candle_close
                    ret = (exit_price - entry_price) / entry_price
                    total_return += ret
                    if ret > profit_threshold:
                        profitable_trades += 1
                    total_trades += 1
                    current_position_state = 0
                    entry_price = 0.0
                    entry_index = -1
            elif current_position_state == -1: # Currently Short
                if predicted_signal == 1 or predicted_signal == 0: # Predicted Buy or Hold (Exit Short)
                    exit_price = current_candle_close
                    ret = (entry_price - exit_price) / entry_price # For short, profit if price drops
                    total_return += ret
                    if ret > profit_threshold:
                        profitable_trades += 1
                    total_trades += 1
                    current_position_state = 0
                    entry_price = 0.0
                    entry_index = -1

        # Force-close any open position at the end of the data
        if current_position_state != 0:
            final_price = candles[-1].close
            ret = 0.0
            if current_position_state == 1: # Long
                ret = (final_price - entry_price) / entry_price
            elif current_position_state == -1: # Short
                ret = (entry_price - final_price) / entry_price

            total_return += ret
            if ret > profit_threshold:
                profitable_trades += 1
            total_trades += 1

        success_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        avg_return = (total_return / total_trades * 100) if total_trades > 0 else 0

        print(f"NN Strategy simulation finished. Results will be printed below.")
        return {"success_rate": success_rate, "per_trade_return": avg_return, "total_trades": total_trades, "positions": nn_positions}

    except FileNotFoundError as e:
        print(f"ERROR: NN Model or Scaler file not found: {e}")
        print("Please ensure you have run 'python src/python/train.py' successfully.")
        return {"success_rate": 0.0, "per_trade_return": 0.0, "total_trades": 0, "positions": [0]*len(candles)}
    except Exception as e:
        print(f"An unexpected error occurred during NN strategy execution: {e}")
        import traceback
        traceback.print_exc()
        return {"success_rate": 0.0, "per_trade_return": 0.0, "total_trades": 0, "positions": [0]*len(candles)}


def main():
    print("--- DEBUG: Inside main() function. ---")
    sys.stdout.flush()

    # Initialize strategy results with default values (no trades)
    # This ensures variables always exist for reporting, even if strategy execution fails
    default_trade_result = TradeResult() # Create a default TradeResult object
    default_trade_result.success_rate = 0.0
    default_trade_result.per_trade_return = 0.0
    default_trade_result.total_trades = 0
    default_trade_result.positions = [] # Empty list of positions

    rsi_result = default_trade_result
    macd_result = default_trade_result
    supertrend_result = default_trade_result
    nn_result = { # NN result is a dictionary, not TradeResult struct
        "success_rate": 0.0,
        "per_trade_return": 0.0,
        "total_trades": 0,
        "positions": []
    }
    
    try:
        # Paths are relative to the project root where main.py is run
        training_candles = load_and_process_data('data/AAPL_training.csv')
        testing_candles = load_and_process_data('data/AAPL_testing.csv')
        print(f"Loaded {len(training_candles)} training candles from data/AAPL_training.csv")
        print(f"Loaded {len(testing_candles)} testing candles from data/AAPL_testing.csv")
    except Exception as e:
        print(f"--- ERROR: Data loading failed: {e} ---")
        sys.stdout.flush()
        sys.exit(1)

    profit_threshold = 0.01 # 1% profit threshold for a successful trade

    print("\n--- Running RSI Strategy (C++) ---")
    try:
        rsi_result = run_rsi_strategy(testing_candles, profit_threshold)
        print(f"RSI Strategy Results:")
        print(f"  Success Rate: {rsi_result.success_rate:.2f}%, Trades: {rsi_result.total_trades}")
        print(f"  Per-Trade Return: {rsi_result.per_trade_return:.4f}%")
    except Exception as e:
        print(f"--- ERROR: RSI strategy execution failed: {e} ---")
        sys.stdout.flush()

    print("\n--- Running MACD Strategy (C++) ---")
    try:
        macd_result = run_macd_strategy(testing_candles, profit_threshold)
        print(f"MACD Strategy Results:")
        print(f"  Success Rate: {macd_result.success_rate:.2f}%, Trades: {macd_result.total_trades}")
        print(f"  Per-Trade Return: {macd_result.per_trade_return:.4f}%")
    except Exception as e:
        print(f"--- ERROR: MACD strategy execution failed: {e} ---")
        sys.stdout.flush()

    print("\n--- Running Supertrend Strategy (C++) ---")
    try:
        supertrend_result = run_supertrend_strategy(testing_candles, profit_threshold)
        print(f"Supertrend Strategy Results:")
        print(f"  Success Rate: {supertrend_result.success_rate:.2f}%, Trades: {supertrend_result.total_trades}")
        print(f"  Per-Trade Return: {supertrend_result.per_trade_return:.4f}%")
    except Exception as e:
        print(f"--- ERROR: Supertrend strategy execution failed: {e} ---")
        sys.stdout.flush()


    # --- Running Neural Network Strategy ---
    # Define paths to model weights and scaler
    nn_model_weights_path = os.path.join(project_root, 'models', 'nn_model_weights.weights.h5')
    nn_scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')

    # Check if model and scaler exist before running NN strategy
    if not os.path.exists(nn_model_weights_path) or not os.path.exists(nn_scaler_path):
        print("\nWARNING: Neural Network model weights or scaler not found.")
        print("Please run 'python src/python/train.py' first to train the NN model.")
    else:
        nn_result = run_nn_strategy(
            testing_candles,
            profit_threshold,
            nn_model_weights_path,
            nn_scaler_path,
            run_rsi_strategy, # Pass C++ functions needed by calculate_indicators_as_features
            run_macd_strategy,
            run_supertrend_strategy
        )
        print(f"Neural Network Strategy Results:")
        print(f"  Success Rate: {nn_result['success_rate']:.2f}%")
        print(f"  Per-Trade Return: {nn_result['per_trade_return']:.4f}%")
        print(f"  Total Trades: {nn_result['total_trades']}")

    print("\n--- Main Program Finished ---")
    sys.stdout.flush()

    # --- Saving Performance Results ---
    performance_data = {
        'Strategy': [],
        'Success Rate (%)': [],
        'Per-Trade Return (%)': [],
        'Total Trades': []
    }

    # Collect RSI results
    performance_data['Strategy'].append('RSI Strategy')
    performance_data['Success Rate (%)'].append(rsi_result.success_rate)
    performance_data['Per-Trade Return (%)'].append(rsi_result.per_trade_return)
    performance_data['Total Trades'].append(rsi_result.total_trades)

    # Collect MACD results
    performance_data['Strategy'].append('MACD Strategy')
    performance_data['Success Rate (%)'].append(macd_result.success_rate)
    performance_data['Per-Trade Return (%)'].append(macd_result.per_trade_return)
    performance_data['Total Trades'].append(macd_result.total_trades)

    # Collect Supertrend results
    performance_data['Strategy'].append('Supertrend Strategy')
    performance_data['Success Rate (%)'].append(supertrend_result.success_rate)
    performance_data['Per-Trade Return (%)'].append(supertrend_result.per_trade_return)
    performance_data['Total Trades'].append(supertrend_result.total_trades)

    # Collect Neural Network results
    performance_data['Strategy'].append('Neural Network Strategy')
    performance_data['Success Rate (%)'].append(nn_result['success_rate'])
    performance_data['Per-Trade Return (%)'].append(nn_result['per_trade_return'])
    performance_data['Total Trades'].append(nn_result['total_trades'])

    results_df = pd.DataFrame(performance_data)
    results_file_path = os.path.join(project_root, 'results', 'strategy_performance.csv')
    results_df.to_csv(results_file_path, index=False)
    print(f"Performance results saved to: {results_file_path}")
    print("\n" + results_df.to_markdown(index=False)) # Print results in markdown table format

if __name__ == "__main__":
    main()