from src.DataProviders.NFLDataProvider import get_todays_nfl_games, build_historical_features

def create_historical_dataset():
    """Build and save full historical NFL dataset into features_all."""
    print("[Create_Games] Building historical NFL dataset...")
    df = build_historical_features()
    print(f"[Create_Games] Done. Saved {len(df)} rows into features_all.")

def create_todays_games():
    """Fetch and save today's NFL games into todays_games."""
    print("[Create_Games] Fetching today's NFL games...")
    df = get_todays_nfl_games()
    if df.empty:
        print("[Create_Games] No games found today.")
    else:
        print("[Create_Games] Done.")
        print(df)

if __name__ == "__main__":
    # Run both for testing
    create_historical_dataset()
    create_todays_games()
