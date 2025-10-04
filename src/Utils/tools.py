import sqlite3
import pandas as pd

DB_PATH = "Data/dataset.sqlite"

def load_table(table: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """Generic loader for any table in the SQLite DB."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df

def save_table(df: pd.DataFrame, table: str, db_path: str = DB_PATH, mode: str = "replace"):
    """Generic saver to SQLite."""
    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists=mode, index=False)
    conn.close()
    print(f"[tools] Saved {len(df)} rows to {db_path}:{table}")

def print_game_predictions(games: pd.DataFrame, ml_probs, ou_probs):
    """
    Nicely print game predictions.
    Expects:
      games    -> DataFrame with 'home_team', 'away_team', 'gameday'
      ml_probs -> list of home win probabilities
      ou_probs -> list of over probabilities
    """
    for i, game in enumerate(games.itertuples()):
        print(f"{game.away_team} @ {game.home_team} ({game.gameday})")
        print(f"   Home win probability: {ml_probs[i]:.2f}")
        print(f"   Over probability: {ou_probs[i]:.2f}")
        print("-" * 55)
