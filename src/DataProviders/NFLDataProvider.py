# mypy: disable-error-code=import, syntax
# pylint: disable=no-member, invalid-name, missing-module-docstring, import-error, wrong-import-order, line-too-long
"""
NFL Data Provider module
Fetches NFL schedules, betting lines, and team stats from nfl_data_py.
Auto-saves processed datasets to SQLite for use in model pipelines.
"""
from datetime import datetime
import sqlite3
import nfl_data_py as nfl
import pandas as pd
import inspect
import subprocess
import sys
from typing import Optional

DB_PATH = "Data/dataset.sqlite"

def _import_lines_fallback(seasons):
    """
    Compatibility wrapper for all nfl_data_py versions.
    Handles 2022–2025 API changes gracefully.
    """
    funcs = {
        "import_betting_data": "new (2024+) betting data API",
        "import_betting_lines": "legacy (2023) betting lines API",
        "import_lines": "very old (pre-2022) betting lines API"
    }

    for fn in funcs:
        if hasattr(nfl, fn) and inspect.isfunction(getattr(nfl, fn)):
            print(f"[NFLDataProvider] Using {fn} — {funcs[fn]}")
            return getattr(nfl, fn)(seasons)

    # self-heal mechanism: upgrade package if nothing works
    print("[NFLDataProvider] ❌ No betting line import function found in nfl_data_py.")
    print("[NFLDataProvider] Attempting to auto-upgrade package...")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "nfl_data_py"])
        import importlib
        importlib.reload(nfl)
        # Try again after upgrade
        for fn in funcs:
            if hasattr(nfl, fn) and inspect.isfunction(getattr(nfl, fn)):
                print(f"[NFLDataProvider] Using {fn} — {funcs[fn]} (after upgrade)")
                return getattr(nfl, fn)(seasons)
    except Exception as e:
        print(f"[NFLDataProvider] Auto-upgrade failed: {e}")

    raise AttributeError(
        "nfl_data_py has no betting data import function, even after upgrade. "
        "Please check your environment or manually run: pip install -U nfl_data_py"
    )


def get_todays_nfl_games(date: str = None) -> pd.DataFrame:
    """Fetch NFL games + Vegas lines + team stats for a specific date (default = today)."""
    if not date:
        date = datetime.today().strftime("%Y-%m-%d")

    target_date = pd.to_datetime(date)
    season = target_date.year if target_date.month >= 8 else target_date.year - 1

    print(f"[NFLDataProvider] Fetching games for {date} (season {season})")

    # Load schedule
    schedules = nfl.import_schedules([season])
    schedules["gameday"] = pd.to_datetime(schedules["gameday"])
    games_today = schedules[schedules["gameday"] == target_date]

    if games_today.empty:
        print(f"[NFLDataProvider] No NFL games found for {date}")
        return pd.DataFrame()

    # Load betting lines safely
    lines = _import_lines_fallback([season])
    games_today = games_today.merge(
        lines[["game_id", "spread_line", "total_line", "away_moneyline", "home_moneyline"]],
        on="game_id", how="left"
    )

    # Load team stats
    team_stats = nfl.import_team_stats([season])[["season", "team", "epa_per_play", "points_per_game"]]
    games_today = _merge_team_stats(games_today, team_stats)

    result = games_today[
        ["gameday", "home_team", "away_team", "spread_line", "total_line",
         "home_moneyline", "away_moneyline", "home_epa", "away_epa", "home_ppg", "away_ppg"]
    ]

    save_to_sqlite(result, "todays_games")
    return result


def build_historical_features(seasons=range(2012, 2025)) -> pd.DataFrame:
    """Build historical dataset with outcomes (home_win, ou_cover) and derived features."""
    print(f"[NFLDataProvider] Building features for seasons: {list(seasons)}")

    schedules = nfl.import_schedules(seasons)
    schedules["gameday"] = pd.to_datetime(schedules["gameday"])

    lines = _import_lines_fallback(seasons)
    df = schedules.merge(
        lines[["game_id", "spread_line", "total_line", "away_moneyline", "home_moneyline"]],
        on="game_id", how="left"
    )

    # Labels
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["total_points"] = df["home_score"] + df["away_score"]

    # Over/Under labels
    df["ou_cover"] = (df["total_points"] > df["total_line"]).astype(int)
    df.loc[df["total_points"] == df["total_line"], "ou_cover"] = -1

    # Team stats
    team_stats = nfl.import_team_stats(seasons)[["season", "team", "epa_per_play", "points_per_game"]]
    df = _merge_team_stats(df, team_stats)

    # Derived features
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
    print(f"[NFLDataProvider] ✅ Saved {len(features)} feature rows to SQLite.")
    return features


def _merge_team_stats(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Helper to merge home/away team stats into dataframe."""
    df = df.merge(
        stats, left_on=["season", "home_team"], right_on=["season", "team"], how="left"
    ).rename(columns={"epa_per_play": "home_epa", "points_per_game": "home_ppg"}).drop(columns="team")

    df = df.merge(
        stats, left_on=["season", "away_team"], right_on=["season", "team"], how="left"
    ).rename(columns={"epa_per_play": "away_epa", "points_per_game": "away_ppg"}).drop(columns="team")

    return df


def save_to_sqlite(df: pd.DataFrame, table: str, db_path=DB_PATH):
    """Save DataFrame into SQLite."""
    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()
    print(f"[NFLDataProvider] 💾 Saved {len(df)} rows into {db_path}:{table}")


if __name__ == "__main__":
    print("[NFLDataProvider] 🏈 Starting data build...\n")

    # Build today's games
    todays = get_todays_nfl_games()
    print(todays)

    # Build historical dataset
    hist = build_historical_features()
    print(hist.head())

    print("\n✅ Done building NFL data pipeline.")
