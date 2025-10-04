import xgboost as xgb
from src.Utils.tools import print_game_predictions

def xgb_runner(X, games):
    """
    Run NFL predictions with trained XGBoost models.
    Expects:
      X     = features as numpy array
      games = dataframe of today's games (with home/away team, date, odds)
    """

    # Load NFL-trained models
    xgb_ml = xgb.Booster()
    xgb_ml.load_model("Models/XGBoost_Models/XGBoost_NFL_ML.json")

    xgb_ou = xgb.Booster()
    xgb_ou.load_model("Models/XGBoost_Models/XGBoost_NFL_UO.json")

    dtest = xgb.DMatrix(X)

    # Moneyline (home win)
    ml_preds = xgb_ml.predict(dtest)
    ml_probs = [p[1] for p in ml_preds]  # probability home wins

    # Over/Under
    ou_preds = xgb_ou.predict(dtest)

    # Centralized output
    print_game_predictions(games, ml_probs=ml_probs, ou_probs=ou_preds)
