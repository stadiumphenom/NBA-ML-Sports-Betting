# 🏈 NFL Machine Learning Betting Models

This project provides an **end-to-end pipeline for NFL betting predictions** using machine learning.  
It fetches **schedules, lines, and team stats** from the [nflverse](https://github.com/nflverse) repos via [`nfl_data_py`](https://github.com/nflverse/nfl_data_py), builds feature sets, trains models, and generates predictions for **moneyline (home win)** and **over/under totals**.

---

##_🚀_Features
- **Automated NFL data ingestion** via `nfl_data_py`  
- **SQLite-backed pipeline** for storing `features_all` (historical) and `todays_games` (daily matchups)  
- **Model training** with Logistic Regression, XGBoost, and Neural Networks  
- **Prediction runners** for XGBoost & NN models  
- **Streamlit dashboard** for interactive viewing of today’s games & predictions  
- **Bankroll tools** (Expected Value, Kelly Criterion) for betting edge analysis  

---

## 📂 Project Structure

```txt
src/
├── DataProviders/
│   └── NFLDataProvider.py        # fetch NFL schedules, odds, stats → SQLite
│
├── Process-Data/
│   └── Create_Games.py           # wrapper for building historical/today games
│
├── Predict/
│   ├── NN_Runner.py              # Neural Net predictions (NFL)
│   ├── XGBoost_Runner.py         # XGB predictions (NFL)
│
├── Train-Models/
│   ├── Logistic_Regression_ML.py
│   ├── Logistic_Regression_OU.py
│   ├── NN_Model_ML.py
│   ├── NN_Model_OU.py
│   ├── XGBoost_Model_ML.py
│   ├── XGBoost_Model_OU.py
│
├── Utils/
│   ├── Dictionaries.py           # NFL team lookups
│   ├── Expected_Value.py
│   ├── Kelly_Criterion.py
│   ├── tools.py                  # DB + print helpers
│
app.py                            # Streamlit dashboard (NFL predictions)
main.py                           # CLI runner for predictions

git clone https://github.com/stadiumphenom/NFL_ML_Sports_Betting.git
cd NFL_ML_Sports_Betting
pip install -r requirements.txt

python main.py -hist

python src/Train-Models/XGBoost_Model_ML.py
python src/Train-Models/XGBoost_Model_OU.py
python src/Train-Models/NN_Model_ML.py
python src/Train-Models/NN_Model_OU.py

python main.py -xgb   # XGBoost only
python main.py -nn    # Neural Net only
python main.py -A     # All models

streamlit run app.py

BUF @ KC (2025-10-03)
   Home win probability: 0.62
   Over probability: 0.55
-------------------------------------------------------

---



---

👉 Do you want me to also generate a **requirements.txt** for you (with `nfl_data_py`, `xgboost`, `tensorflow`, `scikit-learn`, `streamlit`, etc. pinned), so Streamlit Cloud will build and deploy cleanly?  

[Catch the Quantum Wave... Password: spinor](https://pulsr.co.uk/spinor.html)
