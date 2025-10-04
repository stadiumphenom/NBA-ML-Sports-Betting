import tensorflow as tf
from src.Utils.tools import print_game_predictions
from src.Utils.config_loader import load_config

config = load_config()

def nn_runner(X, games):
    """
    Run NFL predictions with trained Neural Network models.
    Expects:
      X     = normalized features as numpy array
      games = dataframe of today's games
    """
    # Load models from config
    nn_ml = tf.keras.models.load_model(config["models"]["nn_ml"])
    nn_ou = tf.keras.models.load_model(config["models"]["nn_ou"])

    # Predictions
    ml_probs = [p[1] for p in nn_ml.predict(X, verbose=0)]
    ou_probs = [p[1] for p in nn_ou.predict(X, verbose=0)]

    print_game_predictions(games, ml_probs=ml_probs, ou_probs=ou_probs)
