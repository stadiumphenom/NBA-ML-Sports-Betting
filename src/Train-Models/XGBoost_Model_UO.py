import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DB_PATH = "../../Data/dataset.sqlite"
TABLE_NAME = "features_all"

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()

    # Target = OU cover (1 if went over, 0 if under)
    y = df["ou_cover"]
    
    # Keep numeric features only
    X = df.drop(columns=["home_win", "ou_cover", "gameday", "home_team", "away_team"])
    return X.values.astype(float), y.values.astype(int)


def train_xgboost():
    data, labels = load_data()
    acc_results = []
    best_model = None
    best_acc = 0

    for i in tqdm(range(100)):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)

        train = xgb.DMatrix(X_train, label=y_train)
        test = xgb.DMatrix(X_test, label=y_test)

        param = {
            "max_depth": 5,
            "eta": 0.05,
            "objective": "binary:logistic",  # 👈 Over/Under binary classification
            "eval_metric": "logloss"
        }
        epochs = 300

        model = xgb.train(param, train, epochs)
        preds = model.predict(test)
        y_pred = (preds > 0.5).astype(int)  # threshold at 0.5

        acc = round(accuracy_score(y_test, y_pred) * 100, 1)
        acc_results.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_model = model
            model.save_model(f"../../Models/XGBoost_Models/XGBoost_{acc}%_NFL_UO.json")
            print(f"New best accuracy: {acc}% (model saved)")

    print(f"Best accuracy over runs: {best_acc}%")


if __name__ == "__main__":
    train_xgboost()
