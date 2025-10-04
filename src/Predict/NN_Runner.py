import tensorflow as tf
import numpy as np

def nn_runner(X, games):
    """
    Runs NFL predictions using Neural Network models.
    Expects normalized features (X) and games DataFrame.
    """

    # Load trained NFL models
    nn_ml = tf.keras.models.load_model("Models/NN_Models/Trained-Model-NFL-ML.h5")
    nn_ou = tf.keras.models.load_model("Models/NN_Models/Trained-Model-NFL-OU.h5")

    # Predict Moneyline (Home Win)
    ml_preds = nn_ml.predict(X, verbose=0)
    ml_probs = [p[1] for p in ml_preds]  # home win probability

    # Predict Over/Under (Over probability)
    ou_preds = nn_ou.predict(X, verbose=0)

    # Print results
    for i, game in enumerate(games.itertuples()):
        print(f"{game.away_team} @ {game.home_team} ({game.gameday})")
        print(f"   Home win probability: {ml_probs[i]:.2f}")
        print(f"   Over probability: {ou_preds[i]:.2f}")
