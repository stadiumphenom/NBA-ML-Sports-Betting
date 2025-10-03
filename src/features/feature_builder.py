"""
Feature Builder for NFL betting models
- Converts schedules + odds + stats into ML-friendly features
- Uses data from dataset.sqlite (populated by nflverse_fetcher)
"""

import sqlite3
import pandas as pd

DB_PATH = "Data/dataset.sqlite"

def load_todays_games():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM todays_games", conn)
    conn.close()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take raw games DataFrame and build engineered features
    Example: spread edges, EPA differentials, efficiency ratios
    """
    # Basic differentials
    df["epa_diff"] = df["home_epa"] - df["away_epa"]
    df["ppg_diff"] = df["home_ppg"] - df["away_ppg"]

    # Vegas implied win probability (rough estimate from moneyline)
    def implied_prob(moneyline):
        if moneyline > 0:
            return 100 / (moneyline + 100)
        else:
            return abs(moneyline) / (abs(moneyline) + 100)

    df["home_implied_prob"] = df["home_moneyline"].apply(implied_prob)
    df["away_implied_prob"] = df["away_moneyline"].apply(implied_prob)

    # Spread vs team strength (how much does EPA differential explain spread)
    df["spread_vs_epa"] = df["spread_line"] - df["epa_diff"]

    return df[
        [
            "gameday", "home_team", "away_team",
            "spread_line", "total_line",
            "home_moneyline", "away_moneyline",
            "epa_diff", "ppg_diff",
            "home_implied_prob", "away_implied_prob",
            "spread_vs_epa"
        ]
    ]


if __name__ == "__main__":
    games = load_todays_games()
    if games.empty:
        print("No games in DB")
    else:
        features = build_features(games)
        print(features)
