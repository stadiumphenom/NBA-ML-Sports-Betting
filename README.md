# 🏈 NFL-ML-Sports-Betting

Machine learning + simulation framework for NFL betting.  
Built with live data from [nflverse](https://github.com/nflverse) and odds APIs.  
CLI-first design, extensible to UI later.

---

## 🚀 Features
- Pulls live **NFL play-by-play data** via `nfl_data_py`.
- Builds feature sets (EPA/play, PROE, Vegas totals, spreads).
- Trains ML models (`scikit-learn`, `xgboost`) for win + totals prediction.
- Simulates **parlays and game outcomes**.
- CLI interface: `predict`, `simulate`, `train`.

---

## 📦 Installation
```bash
git clone https://github.com/stadiumphenom/NFL-ML-Sports-Betting.git
cd NFL-ML-Sports-Betting
pip install -r requirements.txt
