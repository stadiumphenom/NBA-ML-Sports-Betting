import argparse
import sqlite3
import pandas as pd
import tensorflow as tf

from src.Predict import NN_Runner, XGBoost_Runner

DB_PATH = "Data/dataset.sqlite"

def load_todays_games():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM todays_games", conn)
    conn.close()
    return df

def load_features():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM features_all", conn)
    conn.close()
    return df

def main():
    games = load_todays_games()
    features = load_features()

    if games.empty:
        print("No NFL games scheduled for today.")
        return

    # Filter features for today's games
    today_features = features[
        features["gameday"].isin(games["gameday"])
    ]

    # Drop label/ID cols for prediction
    X = today_features.drop(columns=["home_win", "ou_cover", "gameday", "home_team", "away_team"]).values.astype(float)

    if args.nn:
        print("------------ Neural Network Predictions -----------")
        X_norm = tf.keras.utils.normalize(X, axis=1)
        NN_Runner.nn_runner(X_norm, games)
        print("---------------------------------------------------")

    if args.xgb:
        print("------------ XGBoost Predictions ------------------")
        XGBoost_Runner.xgb_runner(X, games)
        print("---------------------------------------------------")

    if args.A:
        print("------------ XGBoost Predictions ------------------")
        XGBoost_Runner.xgb_runner(X, games)
        print("---------------------------------------------------")
        print("------------ Neural Network Predictions -----------")
        X_norm = tf.keras.utils.normalize(X, axis=1)
        NN_Runner.nn_runner(X_norm, games)
        print("---------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NFL ML Predictions")
    parser.add_argument("-xgb", action="store_true", help="Run with XGBoost Model")
    parser.add_argument("-nn", action="store_true", help="Run with Neural Network Model")
    parser.add_argument("-A", action="store_true", help="Run all Models")
    args = parser.parse_args()
    main()
