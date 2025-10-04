import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import xgboost as xgb
import joblib

from src.Utils.tools import load_table
from src.Utils.config_loader import load_config

config = load_config()

def load_models():
    """Load models on demand."""
    models = {}

    # Logistic Regression
    models["LOG_ML"] = joblib.load(config["models"]["log_ml"])
    models["LOG_OU"] = joblib.load(config["models"]["log_ou"])

    # XGBoost
    models["XGB_ML"] = xgb.Booster()
    models["XGB_ML"].load_model(config["models"]["xgb_ml"])

    models["XGB_OU"] = xgb.Booster()
    models["XGB_OU"].load_model(config["models"]["xgb_ou"])

    # Neural Networks
    models["NN_ML"] = tf.keras.models.load_model(config["models"]["nn_ml"])
    models["NN_OU"] = tf.keras.models.load_model(config["models"]["nn_ou"])

    return models

def run_predictions(games, X, models):
    """Run predictions with all models and return updated DataFrame."""
    results = games.copy()

    # Logistic Regression
    log_ml_probs = models["LOG_ML"].predict_proba(X)[:, 1]
    log_ou_probs = models["LOG_OU"].predict_proba(X)[:, 1]
    results["Home Win (LogReg)"] = log_ml_probs
    results["Over (LogReg)"] = log_ou_probs

    # XGBoost
    dtest = xgb.DMatrix(X)
    xgb_ml_probs = [p[1] for p in models["XGB_ML"].predict(dtest)]
    xgb_ou_probs = [p[1] for p in models["XGB_OU"].predict(dtest)]
    results["Home Win (XGB)"] = xgb_ml_probs
    results["Over (XGB)"] = xgb_ou_probs

    # Neural Net
    nn_ml_probs = [p[1] for p in models["NN_ML"].predict(X, verbose=0)]
    nn_ou_probs = [p[1] for p in models["NN_OU"].predict(X, verbose=0)]
    results["Home Win (NN)"] = nn_ml_probs
    results["Over (NN)"] = nn_ou_probs

    return results

def main():
    st.title("🏈 NFL Betting Model Dashboard")
    st.write("Compare predictions from Logistic Regression, XGBoost, and Neural Networks.")

    # Load today's games
    games = load_table("todays_games")

    if games.empty:
        st.warning("No NFL games found for today. Run Create_Games first.")
        return

    st.subheader("Today's Games")
    st.dataframe(games)

    # Feature matrix
    X = games.drop(columns=["gameday", "home_team", "away_team"]).values.astype(float)

    # Button to run predictions
    if st.button("Run Predictions"):
        with st.spinner("Loading models and running predictions..."):
            models = load_models()
            results = run_predictions(games, X, models)

        st.subheader("Predictions")
        st.dataframe(results)

if __name__ == "__main__":
    main()
