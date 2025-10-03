import nfl_data_py as nfl
import pandas as pd
from datetime import datetime

def get_todays_nfl_games(date: str = None) -> pd.DataFrame:
    """
    Fetch NFL games and minimal stats for a specific date (default = today).
    """
    if not date:
        date = datetime.today().strftime("%Y-%m-%d")

    try:
        # Load schedule for the current season
        schedules = nfl.import_schedules([2025])
        schedules["gameday"] = pd.to_datetime(schedules["gameday"])

        # Filter games for the given date
        target_date = pd.to_datetime(date)
        games_today = schedules[schedules["gameday"] == target_date]

        if games_today.empty:
            print(f"No NFL games found for {date}")
            return pd.DataFrame()

        # Load play-by-play for stats
        pbp = nfl.import_pbp_data([2025])
        team_stats = (
            pbp.groupby("posteam")[["epa"]]
            .mean()
            .reset_index()
            .rename(columns={"epa": "epa_per_play"})
        )

        # Merge stats into games
        games_today = games_today.merge(
            team_stats, left_on="home_team", right_on="posteam", how="left"
        ).rename(columns={"epa_per_play": "home_epa"})

        games_today = games_today.merge(
            team_stats, left_on="away_team", right_on="posteam", how="left"
        ).rename(columns={"epa_per_play": "away_epa"})

        return games_today[
            ["gameday", "home_team", "away_team", "home_epa", "away_epa"]
        ]

    except Exception as e:
        print(f"[NFLDataProvider] Error fetching data: {e}")
        return pd.DataFrame()

# Debug/test run
if __name__ == "__main__":
    df = get_todays_nfl_games()
    print(df)
