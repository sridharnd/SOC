import pandas as pd
import os
import sys

# Explicitly add the current directory (src/python) to sys.path if not already there
# This ensures that 'trading_strategies' can be found directly.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from trading_strategies import Candle

def load_and_process_data(filepath):
    """
    Loads candlestick data from a CSV file and converts it into a list of Candle objects.
    Assumes CSV columns: Date, Open, High, Low, Close, Volume
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Ensure required columns are present
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV missing one or more required columns: {required_columns}")

    candles = []
    for index, row in df.iterrows():
        candle = Candle()
        # Convert 'Date' column to a Unix timestamp (integer)
        candle.timestamp = int(pd.to_datetime(row['Date']).timestamp())
        candle.open = float(row['Open'])
        candle.high = float(row['High'])
        candle.low = float(row['Low'])
        candle.close = float(row['Close'])
        # Convert Volume to int, first converting to string and removing commas
        candle.volume = int(str(row['Volume']).replace(',', ''))
        candles.append(candle)
    return candles # This return statement is inside the function


if __name__ == "__main__":
    # This block runs only when data_processor.py is executed directly
    # You stated you are using historical data, so this block won't
    # create dummy data if your files already exist.
    dummy_data_path_train = '../../data/AAPL_training.csv'
    dummy_data_path_test = '../../data/AAPL_testing.csv'

    # This part will ONLY create files if they don't exist.
    # If your actual files are there, this block is skipped.
    if not os.path.exists(dummy_data_path_train):
        print(f"Creating dummy training data at {dummy_data_path_train}")
        dummy_df_train = pd.DataFrame({
            'Date': pd.to_datetime(pd.date_range(start='2010-01-01', periods=2000, freq='D')).strftime('%Y-%m-%d'),
            'Open': [100 + i*0.1 for i in range(2000)],
            'High': [101 + i*0.1 for i in range(2000)],
            'Low': [99 + i*0.1 for i in range(2000)],
            'Close': [100.5 + i*0.1 for i in range(2000)],
            'Volume': [100000 + i*100 for i in range(2000)]
        })
        dummy_df_train.to_csv(dummy_data_path_train, index=False)

    if not os.path.exists(dummy_data_path_test):
        print(f"Creating dummy testing data at {dummy_data_path_test}")
        dummy_df_test = pd.DataFrame({
            'Date': pd.to_datetime(pd.date_range(start='2015-06-01', periods=1000, freq='D')).strftime('%Y-%m-%d'),
            'Open': [200 + i*0.2 for i in range(1000)],
            'High': [201 + i*0.2 for i in range(1000)],
            'Low': [199 + i*0.2 for i in range(1000)],
            'Close': [200.5 + i*0.2 for i in range(1000)],
            'Volume': [200000 + i*200 for i in range(1000)]
        })
        dummy_df_test.to_csv(dummy_data_path_test, index=False)


    print("\n--- Testing Data Processor ---")
    try:
        training_data = load_and_process_data('../../data/AAPL_training.csv')
        testing_data = load_and_process_data('../../data/AAPL_testing.csv')
        print(f"Loaded {len(training_data)} training candles.")
        print(f"Loaded {len(testing_data)} testing candles.")
        if training_data:
            print(f"First training candle close: {training_data[0].close}")
        if testing_data:
            print(f"First testing candle close: {testing_data[0].close}")

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(f"Data processing error: {e}")
    except ImportError:
        print("Error: Could not import 'trading_strategies'. Make sure C++ module is built and in src/python.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")