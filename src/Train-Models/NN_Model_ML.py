import sqlite3, numpy as np, pandas as pd, tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import time
from src.Utils.config_loader import load_config

config = load_config()
db_path, model_path = config["data"]["db_path"], config["models"]["nn_ml"]

con = sqlite3.connect(db_path)
data = pd.read_sql_query("SELECT * FROM features_all", con)
con.close()

y = data["home_win"]
X = data.drop(columns=["home_win", "ou_cover", "gameday", "home_team", "away_team"]).values.astype(float)
X = tf.keras.utils.normalize(X, axis=1)

callbacks = [
    TensorBoard(log_dir=f"Logs/{time.time()}"),
    EarlyStopping(monitor="val_loss", patience=10, mode="min"),
    ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss", mode="min"),
]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=50, validation_split=0.1, batch_size=32, callbacks=callbacks)
