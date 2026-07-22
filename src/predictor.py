from src.features import build_features
from src.simulation import predict_game

class UnknownTeamError(Exception):
    pass

def predict_match(home_team, away_team, country_elo, team_to_confederation,
                   model_h, model_a, features):
    if home_team not in country_elo or away_team not in country_elo:
        raise UnknownTeamError(f"Unknown team: {home_team} or {away_team}")

    X = build_features(home_team, away_team, country_elo, team_to_confederation, features)
    home_goals_pred = model_h.predict(X)[0]
    away_goals_pred = model_a.predict(X)[0]
    p_home, p_draw, p_away = predict_game(home_goals_pred, away_goals_pred)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_home_goals": round(float(home_goals_pred), 2),
        "predicted_away_goals": round(float(away_goals_pred), 2),
        "home_win_prob": round(float(p_home), 3),
        "draw_prob": round(float(p_draw), 3),
        "away_win_prob": round(float(p_away), 3),
    }