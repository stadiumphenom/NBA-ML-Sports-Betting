"""
Feature Builder for NFL betting models
- Converts schedules + odds + stats into ML-friendly features
- Works for both historical seasons and today's games
"""

import sqlite3
import pandas as pd
import nfl_data_py as nfl

DB_PATH = "Data/dataset.sqlite"

def implied_prob(moneyline):
    """Convert Vegas moneyline to implied probability"""
    if pd.isna(moneyline):
        return None
    if moneyline > 0:
        return 100 / (moneyline + 100)
    else:
        return abs(moneyline) / (abs(moneyline) + 100)


def build_historical_features(seasons=range(2012, 2025), save=True) -> pd.DataFrame:
    """
    Build historical features for multiple seasons.
    Includes labels: home_win, ou_cover
    """
    print(f"Fetching schedules for {min(seasons)}–{max(seasons)}")
    schedules = nfl.import_schedules(seasons)

    # Betting lines
    lines = nfl.import_lines(seasons)

    # Merge lines into schedules
    df = schedules.merge(
        lines[["game_id", "spread_line", "total_line", "away_moneyline", "home_moneyline"]],
        on="game_id", how="left"
    )

    # Add outcome labels
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # Over/Under label
    df["total_points"] = df["home_score"] + df["away_score"]
    df["ou_cover"] = (df["total_points"] > df["total_line"]).astype(int)

    # Team stats
    stats = nfl.import_team_stats(seasons)[["season", "team", "epa_per_play", "points_per_game"]]

    # Merge home team stats
    df = df.merge(
        stats, left_on=["season", "home_team"], right_on=["season", "team"], how="left"
    ).rename(columns={"epa_per_play": "home_epa", "points_per_game": "home_ppg"}).drop(columns="team")

    # Merge away team stats
    df = df.merge(
        stats, left_on=["season", "away_team"], right_on=["season", "team"], how="left"
    ).rename(columns={"epa_per_play": "away_epa", "points_per_game": "away_ppg"}).drop(columns="team")

    # Feature engineering
    df["epa_diff"] = df["home_epa"] - df["away_epa"]
    df["ppg_diff"] = df["home_ppg"] - df["away_ppg"]

    df["home_implied_prob"] = df["home_moneyline"].apply(implied_prob)
    df["away_implied_prob"] = df["away_moneyline"].apply(implied_prob)

    df["spread_vs_epa"] = df["spread_line"] - df["epa_diff"]

    features = df[
        [
            "season", "week", "gameday",
            "home_team", "away_team",
            "spread_line", "total_line",
            "home_moneyline", "away_moneyline",
            "epa_diff", "ppg_diff",
            "home_implied_prob", "away_implied_prob",
            "spread_vs_epa",
            "home_win", "ou_cover"
        ]
    ].dropna()

    if save:
        save_to_sqlite(features, "features_all")

    return features


def save_to_sqlite(df: pd.DataFrame, table: str, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved {len(df)} rows into {db_path}:{table}")


if __name__ == "__main__":
    features = build_historical_features(range(2012, 2025))
    print(features.head())
