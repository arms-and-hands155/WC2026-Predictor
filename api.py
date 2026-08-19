from fastapi import FastAPI, HTTPException
import joblib

from src.predictor import predict_match, UnknownTeamError
from src.simulation import simulate_tournament, run_monte_carlo

app = FastAPI(title="World Cup Predictor API")

# Load once at startup
country_elo = joblib.load("models/final_elo.joblib")
model_h = joblib.load("models/home_goals_model.joblib")
model_a = joblib.load("models/away_goals_model.joblib")
features = joblib.load("models/model_features.joblib")
team_to_confederation = joblib.load("models/team_to_confed.joblib")
df_groups = joblib.load("models/df_groups.joblib")
df_group_fixtures = joblib.load("models/df_groups_fixtures.joblib")


@app.get("/")
def root():
    return {"status": "ok", "message": "World Cup Predictor API is running"}


@app.get("/predict/match")
def predict_match_endpoint(home_team: str, away_team: str):
    try:
        return predict_match(home_team, away_team, country_elo,
                              team_to_confederation, model_h, model_a, features)
    except UnknownTeamError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/simulate/tournament")
def simulate_full_tournament():
    result = simulate_tournament(
        model_h, model_a, country_elo, team_to_confederation,
        features, df_groups, df_group_fixtures
    )
    return {
        "winner": result["summary"]["winner"],
        "runner_up": result["summary"]["runner_up"],
        "sf_teams": result["summary"]["sf_teams"],
        "qf_teams": result["summary"]["qf_teams"],
    }


@app.get("/simulate/monte-carlo")
def monte_carlo_odds(n_sims: int = 1000):
    if n_sims > 10000:
        raise HTTPException(status_code=400, detail="Max 10,000 simulations per request")
    df = run_monte_carlo(
    n_sims, model_h, model_a, country_elo, team_to_confederation,
    features, df_groups, df_group_fixtures
)
    return df.to_dict(orient="records")