"""
🏈 NFL Machine Learning Betting Flask App
Serves model predictions for NFL games using Logistic Regression, XGBoost, and Neural Networks.
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify

# Fix path issues (so src modules import cleanly)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from Utils.tools import load_table
from Utils.config_loader import load_config

import joblib
import xgboost as xgb
import tensorflow as tf


# ---------------------------------------------------------------------
# ⚙️ Flask App Setup
# ---------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")
config = load_config()


# ---------------------------------------------------------------------
# 🧠 Load Models
# ---------------------------------------------------------------------
def load_models():
    """Load all models into memory (logistic, XGBoost, and NN)."""
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
        models["NN_ML"] = tf.keras.models.load_model(config["models"]["nn_ml"])
        models["NN_OU"] = tf.keras.models.load_model(config["models"]["nn_ou"])

        print("✅ All models loaded successfully.")
    except Exception as e:
        print(f"[Model Loader] ❌ Error loading models: {e}")

    return models


# ---------------------------------------------------------------------
# 🏈 Prediction Functions
# ---------------------------------------------------------------------
def run_predictions(games: pd.DataFrame, X: np.ndarray, models: dict) -> pd.DataFrame:
    """Run predictions for all models and return augmented dataframe."""
    results = games.copy()

    try:
        # Logistic Regression
        results["Home Win (LogReg)"] = models["LOG_ML"].predict_proba(X)[:, 1]
        results["Over (LogReg)"] = models["LOG_OU"].predict_proba(X)[:, 1]

        # XGBoost
        dtest = xgb.DMatrix(X)
        results["Home Win (XGB)"] = models["XGB_ML"].predict(dtest)
        results["Over (XGB)"] = models["XGB_OU"].predict(dtest)

        # Neural Network
        results["Home Win (NN)"] = models["NN_ML"].predict(X, verbose=0).flatten()
        results["Over (NN)"] = models["NN_OU"].predict(X, verbose=0).flatten()

    except Exception as e:
        print(f"[Predict] Error generating predictions: {e}")

    return results


# ---------------------------------------------------------------------
# 🧩 Flask Routes
# ---------------------------------------------------------------------
@app.route("/")
def home():
    """Landing page showing today’s games and model predictions."""
    try:
        games = load_table("todays_games")

        if games.empty:
            return render_template(
                "index.html",
                message="⚠️ No NFL games found for today. Run the DataProvider first.",
                games=None,
            )

        X = games.drop(columns=["gameday", "home_team", "away_team"], errors="ignore").values.astype(float)
        models = load_models()
        results = run_predictions(games, X, models)

        # Round probabilities for UI
        for col in results.columns:
            if "Win" in col or "Over" in col:
                results[col] = np.round(results[col] * 100, 2)

        return render_template("index.html", games=results.to_dict(orient="records"))

    except Exception as e:
        return render_template("index.html", message=f"❌ Error: {e}", games=None)


@app.route("/api/games")
def api_games():
    """Return today’s games and model predictions as JSON."""
    try:
        games = load_table("todays_games")
        X = games.drop(columns=["gameday", "home_team", "away_team"], errors="ignore").values.astype(float)
        models = load_models()
        results = run_predictions(games, X, models)
        return jsonify(results.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------
# 🚀 Run Flask
# ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
