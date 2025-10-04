import os
import pandas as pd
from flask import Flask, render_template, jsonify

# Local utilities
from src.Utils.tools import load_table
from src.Utils.config_loader import load_config

# Load configuration
config = load_config()

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# ✅ Home route — show dashboard
@app.route("/")
def index():
    try:
        # Load today’s NFL games from SQLite
        games = load_table("todays_games")

        if games is None or games.empty:
            return render_template(
                "index.html",
                message="No NFL games found for today. Please run Create_Games first.",
                games=[]
            )

        # Convert to list of dictionaries for Jinja
        games_list = games.to_dict(orient="records")

        return render_template("index.html", games=games_list)

    except Exception as e:
        # Handle any errors gracefully
        return render_template("index.html", message=f"⚠️ Error loading games: {str(e)}", games=[])


# ✅ API route — return data as JSON for optional frontend/chart usage
@app.route("/api/games")
def api_games():
    try:
        games = load_table("todays_games")
        if games is None or games.empty:
            return jsonify({"message": "No games found"}), 404

        games_list = games.to_dict(orient="records")
        return jsonify(games_list)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Health check
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


# ✅ Entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Flask server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
