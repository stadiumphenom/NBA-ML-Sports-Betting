import streamlit as st
import sqlite3
import pandas as pd
import tensorflow as tf
import xgboost as xgb

from src.Utils.tools import load_table

def load_models():
    models = {
        "NN_ML": tf.keras.models.load_model("Models/NN_Models/Trained-Model-NFL-ML.h5"),
        "NN_OU": tf.keras.models.load_model("Models/NN_Models/Trained-Model-NFL-OU.h5"),
        "XGB_ML": xgb.Booster(),
        "XGB_OU": xgb.Booster()
    }
    models["XGB_ML"].load_model("Models/XGBoost_Models/XGBoost_NFL_ML.json")
    models["XGB_OU"].load_model("Models/XGBoost_Models/XGBoost_NFL_UO.json")
    return models

def main():
    st.title("🏈 NFL Betting Model Dashboard")
    st.write("Predictions from Machine Learning models on today's NFL games.")

    # Load today's games
    games = load_table("todays_games")
    if games.empty:
        st.warning("No NFL games found for today. Run main.py -hist and refresh.")
        return

    st.subheader("Today's Games")
    st.dataframe(games)

    # Features
    X = games.drop(columns=["gameday", "home_team", "away_team"]).values.astype(float)

    # Load models
    models = load_models()

    # Predictions
    st.subheader("Predictions")

    # XGBoost
    dtest = xgb.DMatrix(X)
    ml_preds_xgb = models["XGB_ML"].predict(dtest)
    ou_preds_xgb = models["XGB_OU"].predict(dtest)

    games["Home Win (XGB)"] = [p[1] for p in ml_preds_xgb]
    games["Over (XGB)"] = ou_preds_xgb

    # NN
    ml_preds_nn = models["NN_ML"].predict(X, verbose=0)
    ou_preds_nn = models["NN_OU"].predict(X, verbose=0)

    games["Home Win (NN)"] = [p[1] for p in ml_preds_nn]
    games["Over (NN)"] = [p[1] for p in ou_preds_nn]

    # Final table
    st.dataframe(games)

if __name__ == "__main__":
    main()
