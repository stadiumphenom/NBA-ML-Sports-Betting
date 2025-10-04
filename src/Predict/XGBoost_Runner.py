import xgboost as xgb
from src.Utils.tools import print_game_predictions
from src.Utils.config_loader import load_config

config = load_config()

def xgb_runner(X, games):
    """
    Run NFL predictions with trained XGBoost models.
    Expects:
      X     = features as numpy array
      games = dataframe of today's games
    """
    # Load models from config
    xgb_ml = xgb.Booster()
    xgb_ml.load_model(config["models"]["xgb_ml"])

    xgb_ou = xgb.Booster()
    xgb_ou.load_model(config["models"]["xgb_ou"])

    dtest = xgb.DMatrix(X)

    # Predictions
    ml_preds = [p[1] for p in xgb_ml.predict(dtest)]   # home win prob
    ou_preds = [p[1] for p in xgb_ou.predict(dtest)]   # over prob

    print_game_predictions(games, ml_probs=ml_preds, ou_probs=ou_preds)
