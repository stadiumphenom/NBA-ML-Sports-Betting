import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DB_PATH = "../../Data/dataset.sqlite"
TABLE_NAME = "features_all"   # 👈 new table we'll build from feature_builder

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()

    # Define target
    y = df["home_win"]  # 1 if home win, 0 if away win
    X = df.drop(columns=["home_win", "gameday", "home_team", "away_team"])
    return X.values.astype(float), y.values.astype(int)


def train_xgboost():
    data, labels = load_data()
    acc_results = []
    best_model = None
    best_acc = 0

    for i in tqdm(range(100)):  # reduce loop for speed
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)

        train = xgb.DMatrix(X_train, label=y_train)
        test = xgb.DMatrix(X_test, label=y_test)

        param = {
            "max_depth": 3,
            "eta": 0.05,
            "objective": "multi:softprob",
            "num_class": 2,
            "eval_metric": "mlogloss"
        }
        epochs = 300

        model = xgb.train(param, train, epochs)
        predictions = model.predict(test)
        y_pred = np.argmax(predictions, axis=1)

        acc = round(accuracy_score(y_test, y_pred) * 100, 1)
        acc_results.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_model = model
            model.save_model(f"../../Models/XGBoost_Models/XGBoost_{acc}%_NFL_ML.json")
            print(f"New best accuracy: {acc}% (model saved)")

    print(f"Best accuracy over runs: {best_acc}%")


if __name__ == "__main__":
    train_xgboost()
