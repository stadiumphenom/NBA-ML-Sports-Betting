"""🏈 NFL Betting Model Streamlit Dashboard"""

import os
import sys
import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import xgboost as xgb
import joblib
import numpy as np

# ---------------------------------------------------------------------
# 🔧 Add src to import path (so Streamlit runs from repo root)
# ---------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from Utils.tools import load_table
from Utils.config_loader import load_config

# ---------------------------------------------------------------------
# ⚙️ Load configuration
# ---------------------------------------------------------------------
config = load_config()

# ---------------------------------------------------------------------
# 🧠 Model Loading
# ---------------------------------------------------------------------
def load_models():
    """Load models on demand."""
    models = {}

    try:
        # Logistic Regression
        models["LOG_ML"] = joblib.load(config["models"]["log_ml"])
        models["LOG_OU"] = joblib.load(config["models"]["log_ou"])

        # XGBoost
        models["XGB_ML"] = xgb.Booster()
        models["XGB_ML"].load_model(config["models"]["xgb_ml"])

        models["XGB_OU"] = xgb.Booster()
        models["XGB_OU"].load_model(config["models"]["xgb_ou"])

        # Neural Networks
        models["NN_ML"] = load_model(config["models"]["nn_ml"])
        models["NN_OU"] = load_model(config["models"]["nn_ou"])

        st.success("✅ All models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models


# ---------------------------------------------------------------------
# 🏁 Prediction Runner
# ---------------------------------------------------------------------
def run_predictions(games: pd.DataFrame, X: np.ndarray, models: dict) -> pd.DataFrame:
    """Run predictions with all models and return updated DataFrame."""
    results = games.copy()

    # Logistic Regression
    results["Home Win (LogReg)"] = models["LOG_ML"].predict_proba(X)[:, 1]
    results["Over (LogReg)"] = models["LOG_OU"].predict_proba(X)[:, 1]

    # XGBoost
    dtest = xgb.DMatrix(X)
    results["Home Win (XGB)"] = models["XGB_ML"].predict(dtest)
    results["Over (XGB)"] = models["XGB_OU"].predict(dtest)

    # Neural Nets
    results["Home Win (NN)"] = models["NN_ML"].predict(X, verbose=0).flatten()
    results["Over (NN)"] = models["NN_OU"].predict(X, verbose=0).flatten()

    return results


# ---------------------------------------------------------------------
# 🚀 Streamlit App
# ---------------------------------------------------------------------
def main():
    st.set_page_config(page_title="NFL ML Betting Dashboard", page_icon="🏈", layout="wide")

    st.title("🏈 NFL Betting Model Dashboard")
    st.markdown("### Compare model predictions: Logistic Regression, XGBoost, and Neural Network")

    # Load today's games
    try:
        games = load_table("todays_games")
    except Exception as e:
        st.error(f"Error loading games table: {e}")
        return

    if games.empty:
        st.warning("⚠️ No NFL games found for today. Run `NFLDataProvider` to fetch them.")
        return

    st.subheader("📅 Today's Games")
    st.dataframe(games, use_container_width=True)

    # Drop non-numeric columns for prediction
    X = games.drop(columns=["gameday", "home_team", "away_team"], errors="ignore").values.astype(float)

    if st.button("🏁 Run Predictions"):
        with st.spinner("Running all models..."):
            models = load_models()
            results = run_predictions(games, X, models)
        st.subheader("📊 Predictions")
        st.dataframe(results, use_container_width=True)


if __name__ == "__main__":
    main()
