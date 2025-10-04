import tensorflow as tf

def nn_runner(X, games):
    """
    Run NFL predictions with trained Neural Network models.
    Expects:
      X     = normalized features as numpy array
      games = dataframe of today's games (with home/away team, date, odds)
    """

    # Load NFL-trained models
    nn_ml = tf.keras.models.load_model("Models/NN_Models/Trained-Model-NFL-ML.h5")
    nn_ou = tf.keras.models.load_model("Models/NN_Models/Trained-Model-NFL-OU.h5")

    # Moneyline (home win)
    ml_preds = nn_ml.predict(X, verbose=0)
    ml_probs = [p[1] for p in ml_preds]  # probability home wins

    # Over/Under
    ou_preds = nn_ou.predict(X, verbose=0)

    # Print results
    for i, game in enumerate(games.itertuples()):
        print(f"{game.away_team} @ {game.home_team} ({game.gameday})")
        print(f"   Home win probability: {ml_probs[i]:.2f}")
        print(f"   Over probability: {ou_preds[i]:.2f}")
        print("-" * 55)
