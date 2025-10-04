import tomllib

def load_config(path: str = "config.toml") -> dict:
    """Load repo config file."""
    with open(path, "rb") as f:
        return tomllib.load(f)
