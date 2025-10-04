import xgboost as xgb

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

    # Print results
    for i, game in enumerate(games.itertuples()):
        print(f"{game.away_team} @ {game.home_team} ({game.gameday})")
        print(f"   Home win probability: {ml_probs[i]:.2f}")
        print(f"   Over probability: {ou_preds[i]:.2f}")
        print("-" * 55)
