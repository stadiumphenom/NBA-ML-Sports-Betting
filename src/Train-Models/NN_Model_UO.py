import sqlite3
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

DB_PATH = "../../Data/dataset.sqlite"
TABLE_NAME = "features_all"

# Callbacks
current_time = str(time.time())
tensorboard = TensorBoard(log_dir=f'../../Logs/{current_time}')
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
mcp_save = ModelCheckpoint(f'../../Models/NN_Models/Trained-Model-NFL-OU-{current_time}.h5',
                           save_best_only=True, monitor='val_loss', mode='min')

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()

    # Target = OU cover (1 if over, 0 if under)
    y = df["ou_cover"].astype(int)

    # Features: drop identifiers + labels
    X = df.drop(columns=["home_win", "ou_cover", "gameday", "home_team", "away_team"])

    return X.values.astype(float), y.values


if __name__ == "__main__":
    X, y = load_data()

    # Normalize features
    X = tf.keras.utils.normalize(X, axis=1)

    # Build model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")  # binary output
    ])

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    # Train
    model.fit(X, y,
              epochs=50,
              validation_split=0.1,
              batch_size=32,
              callbacks=[tensorboard, earlyStopping, mcp_save])

    print("Training complete. Best model saved.")
