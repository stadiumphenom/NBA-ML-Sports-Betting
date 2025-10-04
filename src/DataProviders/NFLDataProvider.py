import nfl_data_py as nfl
import pandas as pd
from datetime import datetime
import sqlite3

DB_PATH = "Data/dataset.sqlite"

def _import_lines_fallback(seasons):
    """
    Compatibility wrapper for all nfl_data_py versions.
    Handles import_betting_data, import_betting_lines, or import_lines.
    Dynamically renames columns to expected names if necessary.
    """
    if hasattr(nfl, "import_betting_data"):
        print("[NFLDataProvider] Using import_betting_data()")
        df = nfl.import_betting_data(seasons)
    elif hasattr(nfl, "import_betting_lines"):
        print("[NFLDataProvider] Using import_betting_lines()")
        df = nfl.import_betting_lines(seasons)
    elif hasattr(nfl, "import_lines"):
        print("[NFLDataProvider] Using import_lines()")
        df = nfl.import_lines(seasons)
    else:
        raise ImportError(
            "[NFLDataProvider] No valid betting import function found. "
            "Try: pip install -U nfl_data_py"
        )

    # ✅ Normalize expected column names dynamically
    df.columns = [c.lower() for c in df.columns]

    rename_map = {
        "team_home": "home_team",
        "team_away": "away_team",
        "home_ml": "home_moneyline",
        "away_ml": "away_moneyline",
        "spread": "spread_line",
        "total": "total_line",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Ensure required columns exist (fill if missing)
    for col in ["spread_line", "total_line", "away_moneyline", "home_moneyline"]:
        if col not in df.columns:
            print(f"[NFLDataProvider] Warning: missing {col}, filling with NaN")
            df[col] = pd.NA

    return df


def build_historical_features(seasons=range(2012, 2025)) -> pd.DataFrame:
    """Build historical dataset with outcomes (home_win, ou_cover) and features."""
    print(f"[NFLDataProvider] Building features for seasons: {list(seasons)}")

    schedules = nfl.import_schedules(seasons)
    schedules["gameday"] = pd.to_datetime(schedules["gameday"])

    lines = _import_lines_fallback(seasons)
    print(f"[NFLDataProvider] Lines columns: {list(lines.columns)}")  # Debug info

    df = schedules.merge(
        lines[["game_id", "spread_line", "total_line", "away_moneyline", "home_moneyline"]],
        on="game_id", how="left"
    )

    # Labels
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["total_points"] = df["home_score"] + df["away_score"]
    df["ou_cover"] = (df["total_points"] > df["total_line"]).astype(int)
    df.loc[df["total_points"] == df["total_line"], "ou_cover"] = -1

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
    print(f"[NFLDataProvider] Saved {len(features)} feature rows to SQLite.")
    return features
