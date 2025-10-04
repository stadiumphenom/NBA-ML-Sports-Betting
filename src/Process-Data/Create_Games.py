from src.DataProviders.NFLDataProvider import build_historical_features, get_todays_nfl_games

def create_historical_dataset():
    """Build and save full historical NFL dataset."""
    print("Building historical NFL dataset...")
    df = build_historical_features(range(2012, 2025))
    print(f"Created historical dataset with {len(df)} rows")

def create_todays_games():
    """Fetch and save today's NFL games."""
    print("Fetching today's NFL games...")
    df = get_todays_nfl_games()
    print(df if not df.empty else "No games today.")

if __name__ == "__main__":
    create_historical_dataset()
    create_todays_games()
