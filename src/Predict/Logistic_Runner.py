import joblib
from src.Utils.tools import print_game_predictions
from src.Utils.config_loader import load_config

config = load_config()

def logistic_runner(X, games):
    """
    Run NFL predictions with trained Logistic Regression models.
    Expects:
      X     = features as numpy array
      games = dataframe of today's games
    """
    # Load models
    log_ml = joblib.load(config["models"]["log_ml"])
    log_ou = joblib.load(config["models"]["log_ou"])

    # Predictions
    ml_probs = log_ml.predict_proba(X)[:, 1]   # P(home win)
    ou_probs = log_ou.predict_proba(X)[:, 1]   # P(over)

    print_game_predictions(games, ml_probs=ml_probs, ou_probs=ou_probs)
