import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

DB_PATH = "../../Data/dataset.sqlite"
TABLE_NAME = "features_all"

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()

    # Target = Over/Under cover (1 if over, 0 if under)
    y = df["ou_cover"].astype(int)

    # Features = drop identifiers + labels
    X = df.drop(columns=["home_win", "ou_cover", "gameday", "home_team", "away_team"])
    return X.values.astype(float), y.values


if __name__ == "__main__":
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = LogisticRegression(max_iter=500)

    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.3f}")
    print("Classification Report:")
    print(report)

    # Save the trained model
    joblib.dump(model, "../../Models/Logistic_Models/LogReg_NFL_OU.pkl")
    print("Model saved to ../../Models/Logistic_Models/LogReg_NFL_OU.pkl")
