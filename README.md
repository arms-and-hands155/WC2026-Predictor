# FIFA World Cup 2026 Simulator

A Monte Carlo simulation of the 2026 FIFA World Cup built from scratch using historical match data, a custom Elo rating system, and Poisson regression models. The app simulates the full 48-team tournament thousands of times to estimate each team's probability of winning, reaching the final, and advancing through each knockout round.

---

## How it works

### 1. Elo ratings
A custom Elo system is trained on ~47,000 historical international matches sourced from Kaggle. Matches are weighted by tournament tier (World Cup > continental finals > qualifiers > friendlies) and goal differential. Ratings are initialized from a 2025 baseline and updated through March 2026.

### 2. Goal prediction models
Two Poisson GLMs (one for home goals, one for away goals) are trained on competitive international matches. Features include:
- Home and away Elo ratings
- Neutral venue flag
- Tournament weight (importance of the match)
- Home and away confederation (one-hot encoded)

Expected goals from both models are combined into a full scoreline probability matrix, from which win/draw/loss probabilities are derived analytically.

### 3. Tournament simulation
The full 2026 format is simulated:
- **Group stage**: 12 groups of 4, each team plays 3 matches, top 2 from each group qualify automatically
- **Third-place qualification**: best 8 third-placed teams also advance (ranked by points, goal difference, goals scored, then FIFA rank)
- **Knockout stage**: Round of 32 → Round of 16 → Quarter-finals → Semi-finals → Final, with penalty shootouts resolved via normalized win probabilities when matches are level after 90 minutes

### 4. Monte Carlo
The full tournament is simulated up to 2,000 times. Results are aggregated into stage-reach probabilities for each of the 48 teams.

---

## Results (1,000 simulations)

| Team | Win % | Final % | SF % |
|------|-------|---------|------|
| Argentina | ~25% | ~35% | ~50% |
| Spain | ~21% | ~37% | ~46% |
| France | ~8% | ~16% | ~30% |
| Brazil | ~8% | ~17% | ~31% |
| Colombia | ~6% | ~12% | ~23% |

*Results vary across runs due to the stochastic nature of the simulation.*

---

## Project structure

```
WC2026/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── FIFA_confederations.csv
├── models/
│   ├── away_goals_model.joblib
│   ├── df_groups_fixtures.joblib
│   ├── df_groups.joblib
│   ├── final_elo.joblib
│   ├── home_goals_model.joblib
│   ├── model_features.joblib
│   └── team_to_confed.joblib
├── notebooks/
│   ├── 01_data.ipynb
│   ├── 02_model.ipynb
│   ├── 03_simulation.ipynb
│   └── 04_multiple_tournament_sims.ipynb
└── src/
    ├── __init__.py
    ├── data.py
    ├── elo.py
    ├── features.py
    └── simulation.py
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Kaggle API key

Match data is downloaded via the Kaggle API. Set up your credentials:

```bash
# Create ~/.kaggle/kaggle.json with your API key
# Or set environment variables:
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

Or create a `.env` file at the project root:
```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

### 3. Required data file

`data/FIFA_confederations.csv` is not from Kaggle — it maps each of the 48 WC 2026 teams to their FIFA confederation. You need this file to run the simulation. It has two columns:

```
nation,confederation
France,UEFA
Argentina,CONMEBOL
...
```

This file is included in the repository.

### 4. Run the notebooks

Run in order:

```
01_data.ipynb → 02_model.ipynb → 03_simulation.ipynb
```

This generates the `.joblib` model files used by the app. These are not committed to the repo and must be generated locally.

### 5. Launch the app

```bash
streamlit run app.py
```

---

## Streamlit app

The app has two tabs:

- **Win Probabilities** — run Monte Carlo simulations (100–2,000) and view each team's probability of reaching each stage, with a progress bar and estimated time remaining
- **Simulate Tournament** — simulate a single tournament and view the full group stage tables and knockout bracket with connector lines, scores, and penalty shootout indicators

---

## Disclaimer

The Poisson GLM approach for goal prediction and parts of the bracket simulation structure were inspired by [anesriad/football_WorldCup_2026_predictions](https://github.com/anesriad/football_WorldCup_2026_predictions). 


---

## Data sources

- [International Football Results 1872–2025](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) — Kaggle (martj42)
- [International Football Elo Ratings](https://www.kaggle.com/datasets/saifalnimri/international-football-elo-ratings) — Kaggle (saifalnimri)
