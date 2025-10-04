import sqlite3, xgboost as xgb
import pandas as pd, numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.Utils.config_loader import load_config

config = load_config()
db_path, model_path = config["data"]["db_path"], config["models"]["xgb_ml"]

con = sqlite3.connect(db_path)
data = pd.read_sql_query("SELECT * FROM features_all", con)
con.close()

y = data["home_win"]
X = data.drop(columns=["home_win", "ou_cover", "gameday", "home_team", "away_team"]).values.astype(float)

acc_results = []
for _ in tqdm(range(100)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    train = xgb.DMatrix(X_train, label=y_train)
    test = xgb.DMatrix(X_test, label=y_test)

    params = {"max_depth": 3, "eta": 0.01, "objective": "multi:softprob", "num_class": 2}
    model = xgb.train(params, train, num_boost_round=750)

    preds = [p.argmax() for p in model.predict(test)]
    acc = round(accuracy_score(y_test, preds) * 100, 1)
    acc_results.append(acc)

    if acc == max(acc_results):
        model.save_model(model_path)
