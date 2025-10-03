import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import toml

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.Utils.NFL_Dictionaries import (
    nfl_team_index_07, nfl_team_index_08, nfl_team_index_12,
    nfl_team_index_13, nfl_team_index_14, nfl_team_index_current
)

config = toml.load("../../config.toml")

df = pd.DataFrame()
scores = []
win_margin = []
OU = []
OU_Cover = []
games = []
days_rest_away = []
days_rest_home = []

teams_con = sqlite3.connect("../../Data/NFL_TeamData.sqlite")
odds_con = sqlite3.connect("../../Data/NFL_OddsData.sqlite")

for key, value in config['create-games'].items():
    print(key)
    odds_df = pd.read_sql_query(f"SELECT * FROM \"nfl_odds_{key}_new\"", odds_con, index_col="index")
    season = key

    for row in odds_df.itertuples():
        home_team = row[2]
        away_team = row[3]
        date = row[1]

        team_df = pd.read_sql_query(f"SELECT * FROM \"{date}\"", teams_con, index_col="index")
        if len(team_df.index) != 32:
            continue

        scores.append(row[8])
        OU.append(row[4])
        days_rest_home.append(row[10])
        days_rest_away.append(row[11])

        win_margin.append(1 if row[9] > 0 else 0)

        if row[8] < row[4]:
            OU_Cover.append(0)
        elif row[8] > row[4]:
            OU_Cover.append(1)
        else:
            OU_Cover.append(2)

        try:
            if season == '2007-08':
                home_team_series = team_df.iloc[nfl_team_index_07[home_team]]
                away_team_series = team_df.iloc[nfl_team_index_07[away_team]]
            elif season in ('2008-09', '2009-10', '2010-11', '2011-12'):
                home_team_series = team_df.iloc[nfl_team_index_08[home_team]]
                away_team_series = team_df.iloc[nfl_team_index_08[away_team]]
            elif season == '2012-13':
                home_team_series = team_df.iloc[nfl_team_index_12[home_team]]
                away_team_series = team_df.iloc[nfl_team_index_12[away_team]]
            elif season == '2013-14':
                home_team_series = team_df.iloc[nfl_team_index_13[home_team]]
                away_team_series = team_df.iloc[nfl_team_index_13[away_team]]
            elif season in ('2022-23', '2023-24'):
                home_team_series = team_df.iloc[nfl_team_index_current[home_team]]
                away_team_series = team_df.iloc[nfl_team_index_current[away_team]]
            else:
                home_team_series = team_df.iloc[nfl_team_index_14[home_team]]
                away_team_series = team_df.iloc[nfl_team_index_14[away_team]]

            game = pd.concat([
                home_team_series,
                away_team_series.rename(index={col: f"{col}.1" for col in team_df.columns})
            ])
            games.append(game)
        except KeyError as e:
            print(f"Team index error for {home_team} or {away_team} in season {season}: {e}")
            continue

odds_con.close()
teams_con.close()

if not games:
    raise ValueError("No valid games processed. Check data or team indices.")

season_df = pd.concat(games, ignore_index=True, axis=1).T
frame = season_df.drop(columns=['TEAM_ID', 'TEAM_ID.1'], errors='ignore')
frame['Score'] = np.asarray(scores)
frame['Home-Team-Win'] = np.asarray(win_margin)
frame['OU'] = np.asarray(OU)
frame['OU-Cover'] = np.asarray(OU_Cover)
frame['Days-Rest-Home'] = np.asarray(days_rest_home)
frame['Days-Rest-Away'] = np.asarray(days_rest_away)

for field in frame.columns:
    if 'TEAM_' in field or 'Date' in field or field not in frame:
        continue
    try:
        frame[field] = frame[field].astype(float)
    except ValueError as e:
        print(f"Could not convert {field} to float: {e}")

con = sqlite3.connect("../../Data/nfl_dataset.sqlite")
frame.to_sql("nfl_dataset_2012-24", con, if_exists="replace")
con.close()
