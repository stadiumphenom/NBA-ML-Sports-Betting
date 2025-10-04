import argparse
import tensorflow as tf
import sqlite3
import pandas as pd

from src.DataProviders.NFLDataProvider import get_todays_nfl_games, build_historical_features
from src.Predict import NN_Runner, XGBoost_Runner

DB_PATH = "Data/dataset.sqlite"

def load_todays_features():
    """Load features for today's games from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM todays_games", conn)
    conn.close()

    if df.empty:
        print("No NFL games scheduled for today.")
        return pd.DataFrame()

    # Drop non-feature columns
    X = df.drop(columns=["gameday", "home_team", "away_team"])
    return df, X.values.astype(float)


def main():
    # Step 1: Get today's games from nflverse
    todays_games = get_todays_nfl_games()
    if todays_games.empty:
        return

    # Step 2: Load features from DB
    games, X = load_todays_features()

    # Step 3: Run selected models
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
    parser.add_argument(
        "-hist", action="store_true",
        help="Build historical dataset (for training). Run once before training models."
    )
    args = parser.parse_args()

    if args.hist:
        build_historical_features()

    main()
