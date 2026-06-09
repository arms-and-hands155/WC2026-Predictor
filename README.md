# FIFA World Cup 2026 Simulator

A Monte Carlo simulation of the FIFA World Cup 2026 built from scratch using match history, a custom Elo rating system, and machine learning.

## How it works

1. **Elo ratings** — custom Elo system trained on ~47,000 historical international matches, weighted by tournament tier (World Cup → continental finals → qualifiers → friendlies) and goal differential
2. **Match outcome model** — Logistic Regression classifier predicting win/draw/loss from Elo difference, recent form, head-to-head record, and squad value differential (60% accuracy on held-out 2025 data)
3. **Score model** — XGBoost Poisson regressor predicting expected goals for each team independently
4. **Tournament simulation** — full 48-team bracket (group stage → R32 → R16 → QF → SF → Final) simulated 10,000 times via Monte Carlo; penalty shootouts resolved via ELO-weighted probability

## Results (10,000 simulations)

| Team | Win probability |
|------|----------------|
| Brazil | ~15% |
| France | ~12% |
| England | ~10% |
| Spain | ~9% |
| Argentina | ~8% |

## Project structure

```
├── notebooks/
│   ├── 01_data.ipynb                    # ELO training on historical match data and sets up CSV files
│   ├── 02_model.ipynb                   # Trains goal count models
│   ├── 03_simulation.ipynb              # Step by step going through one simulation
│   └── 04_multiple_tournament_sims.ipynb # Full tournament Monte Carlo simulation
├── src/
│   ├── data.py             # Data loading
│   ├── elo.py              # Elo rating system
│   ├── features.py         # Feature engineering (form, h2h, squad value)
│   └── simulation.py       # Match prediction + tournament runner
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Run the notebooks in order: `01_data` → `02_model` → `03_simulation` → `04_multiple_tournament_sims`.

The training data is sourced from Kaggle ([International Football Results 1872–2025](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)). Download it and place the CSV in `data/`.

## Key design decisions


