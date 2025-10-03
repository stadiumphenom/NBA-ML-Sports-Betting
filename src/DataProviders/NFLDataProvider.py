import nfl_data_py as nfl
import pandas as pd
from datetime import datetime
import sqlite3

def get_todays_nfl_games(date: str = None) -> pd.DataFrame:
    """
    Fetch NFL games + Vegas lines + team stats for a specific date (default = today).
    Saves results into SQLite.
    """
    if not date:
        date = datetime.today().strftime("%Y-%m-%d")

    try:
        target_date = pd.to_datetime(date)
        season = target_date.year

        # Load schedule
        schedules = nfl.import_schedules([season])
        schedules["gameday"] = pd.to_datetime(schedules["gameday"])
        games_today = schedules[schedules["gameday"] == target_date]

        if games_today.empty:
            print(f"No NFL games found for {date}")
            return pd.DataFrame()

        # Load betting lines
        lines = nfl.import_lines([season])
        games_today = games_today.merge(
            lines[["game_id", "spread_line", "total_line", "away_moneyline", "home_moneyline"]],
            on="game_id", how="left"
        )

        # Load team stats (cheaper than PBP for now)
        team_stats = nfl.import_team_stats([season])[["team", "epa_per_play", "points_per_game"]]

        games_today = games_today.merge(
            team_stats, left_on="home_team", right_on="team", how="left"
        ).rename(columns={"epa_per_play": "home_epa", "points_per_game": "home_ppg"}).drop(columns="team")

        games_today = games_today.merge(
            team_stats, left_on="away_team", right_on="team", how="left"
        ).rename(columns={"epa_per_play": "away_epa", "points_per_game": "away_ppg"}).drop(columns="team")

        result = games_today[
            ["gameday", "home_team", "away_team", "spread_line", "total_line",
             "home_moneyline", "away_moneyline", "home_epa", "away_epa", "home_ppg", "away_ppg"]
        ]

        save_games_to_db(result)  # save to SQLite
        return result

    except Exception as e:
        print(f"[NFLDataProvider] Error fetching data: {e}")
        return pd.DataFrame()


def save_games_to_db(df: pd.DataFrame, db_path="Data/dataset.sqlite", table="todays_games"):
    if df.empty:
        return
    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved {len(df)} games into {db_path}:{table}")


if __name__ == "__main__":
    df = get_todays_nfl_games()
    print(df)
