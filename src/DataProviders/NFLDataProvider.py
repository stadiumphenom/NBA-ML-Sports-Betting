import nfl_data_py as nfl
import pandas as pd
from datetime import datetime
import sqlite3

DB_PATH = "Data/dataset.sqlite"

def get_todays_nfl_games(date: str = None) -> pd.DataFrame:
    """Fetch NFL games + Vegas lines + team stats for a specific date (default = today)."""
    if not date:
        date = datetime.today().strftime("%Y-%m-%d")

    target_date = pd.to_datetime(date)
    season = target_date.year

    # Load schedule
    schedules = nfl.import_schedules([season])
    schedules["gameday"] = pd.to_datetime(schedules["gameday"])
    games_today = schedules[schedules["gameday"] == target_date]

    if games_today.empty:
        print(f"No NFL games found for {date}")
        return pd.DataFrame()

    # Add lines
    lines = nfl.import_lines([season])
    games_today = games_today.merge(
        lines[["game_id", "spread_line", "total_line", "away_moneyline", "home_moneyline"]],
        on="game_id", how="left"
    )

    # Add team stats
    team_stats = nfl.import_team_stats([season])[["team", "epa_per_play", "points_per_game"]]
    games_today = _merge_team_stats(games_today, team_stats)

    result = games_today[
        ["gameday", "home_team", "away_team", "spread_line", "total_line",
         "home_moneyline", "away_moneyline", "home_epa", "away_epa", "home_ppg", "away_ppg"]
    ]

    save_to_sqlite(result, "todays_games")
    return result


def build_historical_features(seasons=range(2012, 2025)) -> pd.DataFrame:
    """Build historical dataset with outcomes (home_win, ou_cover)."""
    schedules = nfl.import_schedules(seasons)
    schedules["gameday"] = pd.to_datetime(schedules["gameday"])

    lines = nfl.import_lines(seasons)
    df = schedules.merge(
        lines[["game_id", "spread_line", "total_line", "away_moneyline", "home_moneyline"]],
        on="game_id", how="left"
    )

    # Labels
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["total_points"] = df["home_score"] + df["away_score"]
    df["ou_cover"] = (df["total_points"] > df["total_line"]).astype(int)

    # Team stats
    team_stats = nfl.import_team_stats(seasons)[["season", "team", "epa_per_play", "points_per_game"]]
    df = _merge_team_stats(df, team_stats)

    df["epa_diff"] = df["home_epa"] - df["away_epa"]
    df["ppg_diff"] = df["home_ppg"] - df["away_ppg"]
    df["spread_vs_epa"] = df["spread_line"] - df["epa_diff"]

    features = df[
        ["season", "week", "gameday",
         "home_team", "away_team",
         "spread_line", "total_line", "home_moneyline", "away_moneyline",
         "home_epa", "away_epa", "home_ppg", "away_ppg",
         "epa_diff", "ppg_diff", "spread_vs_epa",
         "home_win", "ou_cover"]
    ].dropna()

    save_to_sqlite(features, "features_all")
    return features


def _merge_team_stats(df, stats):
    """Helper to merge home/away team stats into dataframe."""
    df = df.merge(
        stats, left_on=["season", "home_team"], right_on=["season", "team"], how="left"
    ).rename(columns={"epa_per_play": "home_epa", "points_per_game": "home_ppg"}).drop(columns="team")

    df = df.merge(
        stats, left_on=["season", "away_team"], right_on=["season", "team"], how="left"
    ).rename(columns={"epa_per_play": "away_epa", "points_per_game": "away_ppg"}).drop(columns="team")
    return df


def save_to_sqlite(df: pd.DataFrame, table: str, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved {len(df)} rows into {db_path}:{table}")


if __name__ == "__main__":
    # Build today's games
    todays = get_todays_nfl_games()
    print(todays)

    # Build historical features
    hist = build_historical_features()
    print(hist.head())
