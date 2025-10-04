import argparse
import tensorflow as tf
import xgboost as xgb
import pandas as pd

from src.DataProviders.NFLDataProvider import get_todays_nfl_games
from src.Predict import NN_Runner, XGBoost_Runner
from src.Utils.config_loader import load_config
from src.Utils.tools import load_table

config = load_config()

def main():
    # Get today's games
    games = load_table("todays_games")
    if games.empty:
        print("No NFL games found today. Run Create_Games first.")
        return

    # Features for models
    X = games.drop(columns=["gameday", "home_team", "away_team"]).values.astype(float)

    if args.nn:
        print("------------ Neural Network Model Predictions -----------")
        X_norm = tf.keras.utils.normalize(X, axis=1)
        NN_Runner.nn_runner(X_norm, games)

    if args.xgb:
        print("--------------- XGBoost Model Predictions ---------------")
        XGBoost_Runner.xgb_runner(X, games)

    if args.A:
        print("--------------- Running All Models ---------------")
        XGBoost_Runner.xgb_runner(X, games)
        X_norm = tf.keras.utils.normalize(X, axis=1)
        NN_Runner.nn_runner(X_norm, games)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFL ML Prediction Runner")
    parser.add_argument("-xgb", action="store_true", help="Run with XGBoost Model")
    parser.add_argument("-nn", action="store_true", help="Run with Neural Network Model")
    parser.add_argument("-A", action="store_true", help="Run all Models")
    args = parser.parse_args()
    main()
