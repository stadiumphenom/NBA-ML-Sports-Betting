import argparse
from datetime import datetime, timedelta

import pandas as pd
import tensorflow as tf
from colorama import Fore, Style

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Predict import NN_Runner, XGBoost_Runner
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import create_todays_games_from_odds, get_json_data, to_data_frame, get_todays_games_json, create_todays_games

todays_games_url = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2024/scores/00_todays_scores.json'
data_url = 'https://stats.nba.com/stats/leaguedashteamstats?' \
           'Conference=&DateFrom=&DateTo=&Division=&GameScope=&' \
           'GameSegment=&LastNGames=0&LeagueID=00&Location=&' \
           'MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&' \
           'PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&' \
           'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&' \
           'Season=2024-25&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
           'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='

def createTodaysGames(games, df, odds):
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []
    home_team_days_rest = []
    away_team_days_rest = []

    schedule_df = pd.read_csv('Data/nba-2024-UTC.csv', parse_dates=['Date'])

    for game in games:
        home_team, away_team = game[0], game[1]
        if home_team not in team_index_current or away_team not in team_index_current:
            continue

        if odds:
            game_key = home_team + ':' + away_team
            if game_key not in odds:
                continue
            game_odds = odds[game_key]
            todays_games_uo.append(game_odds.get('under_over_odds', 0))
            home_team_odds.append(game_odds.get(home_team, {}).get('money_line_odds', 0))
            away_team_odds.append(game_odds.get(away_team, {}).get('money_line_odds', 0))
        else:
            todays_games_uo.append(float(input(f'{home_team} vs {away_team}: ')))
            home_team_odds.append(float(input(f'{home_team} odds: ')))
            away_team_odds.append(float(input(f'{away_team} odds: ')))

        today = datetime.today()
        home_games = schedule_df[(schedule_df['Home Team'] == home_team) | (schedule_df['Away Team'] == home_team)]
        away_games = schedule_df[(schedule_df['Home Team'] == away_team) | (schedule_df['Away Team'] == away_team)]

        previous_home = home_games[home_games['Date'] <= today].sort_values('Date', ascending=False).head(1)
        previous_away = away_games[away_games['Date'] <= today].sort_values('Date', ascending=False).head(1)

        home_days_off = (today - previous_home['Date'].values[0]) if not previous_home.empty else timedelta(days=7)
        away_days_off = (today - previous_away['Date'].values[0]) if not previous_away.empty else timedelta(days=7)

        home_team_days_rest.append(home_days_off.days if hasattr(home_days_off, 'days') else 7)
        away_team_days_rest.append(away_days_off.days if hasattr(away_days_off, 'days') else 7)

        home_team_series = df.iloc[team_index_current.get(home_team)]
        away_team_series = df.iloc[team_index_current.get(away_team)]
        stats = pd.concat([home_team_series, away_team_series])
        stats['Days-Rest-Home'] = home_team_days_rest[-1]
        stats['Days-Rest-Away'] = away_team_days_rest[-1]
        match_data.append(stats)

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1).T
    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'], errors='ignore')
    data = frame_ml.values.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds

def main():
    odds = None
    if args.odds:
        odds = SbrOddsProvider(sportsbook=args.odds).get_odds()
        games = create_todays_games_from_odds(odds)
        if not games:
            print("No games found.")
            return
        if (games[0][0] + ':' + games[0][1]) not in odds:
            print(f"{games[0][0]}:{games[0][1]}")
            print(Fore.RED + "--------------Games list not up to date for todays games!!! Scraping disabled until list is updated.--------------" + Style.RESET_ALL)
            odds = None
        else:
            print(f"------------------{args.odds} odds data------------------")
            for g in odds:
                home_team, away_team = g.split(":")
                print(f"{away_team} ({odds[g][away_team]['money_line_odds']}) @ {home_team} ({odds[g][home_team]['money_line_odds']})")
    else:
        data = get_todays_games_json(todays_games_url)
        games = create_todays_games(data)

    data = get_json_data(data_url)
    df = to_data_frame(data)
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = createTodaysGames(games, df, odds)

    if args.nn:
        print("------------Neural Network Model Predictions-----------")
        data = tf.keras.utils.normalize(data, axis=1)
        NN_Runner.nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")

    if args.xgb:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")

    if args.A:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")
        data = tf.keras.utils.normalize(data, axis=1)
        print("------------Neural Network Model Predictions-----------")
        NN_Runner.nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model to Run')
    parser.add_argument('-xgb', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-nn', action='store_true', help='Run with Neural Network Model')
    parser.add_argument('-A', action='store_true', help='Run all Models')
    parser.add_argument('-odds', help='Sportsbook to fetch from. (fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny')
    parser.add_argument('-kc', action='store_true', help='Calculates percentage of bankroll to bet based on model edge')
    args = parser.parse_args()
    main()
