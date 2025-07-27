import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def build_trading_model(input_shape):
    """
    Builds a Sequential Keras model for trading signal prediction.
    A common architecture for time series data might include LSTM layers.
    """
    model = Sequential([
        # Input shape: (timesteps, features) - e.g., (SEQUENCE_LENGTH, N_FEATURES)
        LSTM(128, return_sequences=True, input_shape=input_shape, activation='tanh'),
        Dropout(0.3),
        LSTM(64, activation='tanh'),
        Dropout(0.3),
        # Output layer: 3 classes for Buy, Sell, Hold (one-hot encoded)
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', # For 3-class classification
                  metrics=['accuracy'])

    return model

if __name__ == "__main__":
    # Example usage (input shape based on train.py's SEQUENCE_LENGTH and N_FEATURES)
    example_input_shape = (60, 4)
    model = build_trading_model(example_input_shape)
    model.summary()
    print("Model created successfully.")
    