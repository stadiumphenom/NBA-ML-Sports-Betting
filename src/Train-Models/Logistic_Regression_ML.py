import sqlite3, joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from src.Utils.config_loader import load_config

config = load_config()
db_path, model_path = config["data"]["db_path"], config["models"]["log_ml"]

con = sqlite3.connect(db_path)
data = pd.read_sql_query("SELECT * FROM features_all", con)
con.close()

y = data["home_win"]
X = data.drop(columns=["home_win", "ou_cover", "gameday", "home_team", "away_team"]).values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test))}")
print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, model_path)
