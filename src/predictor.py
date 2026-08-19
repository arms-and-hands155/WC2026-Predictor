from src.features import build_features
import numpy as np

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

def predict_game(lambda_h, lambda_a, n=10000):
    h_goals = np.random.poisson(lambda_h, n)
    a_goals = np.random.poisson(lambda_a, n)
    
    p_home = (h_goals > a_goals).mean()
    p_draw = (h_goals == a_goals).mean()
    p_away = (h_goals < a_goals).mean()
    
    return p_home, p_draw, p_away

def simulate_match(home_team, away_team, home_goal_model, away_goal_model, country_elo, team_to_confederation, feature):
    np.random.seed()
    predict = predict_match(
        home_team, away_team, country_elo, team_to_confederation, home_goal_model, away_goal_model, feature)
    
    h_goals = np.random.poisson(predict['predicted_home_goals'])
    a_goals = np.random.poisson(predict['predicted_away_goals'])
    
    if h_goals > a_goals:
        result = 'home_win'
        winner = home_team
    elif a_goals > h_goals:
        result = 'away_win'
        winner = away_team
    else:
        result = 'draw'
        winner = None
    
    return {
        "home_team": home_team,
        'away_team': away_team,
        'home_goals': h_goals,
        'away_goals': a_goals,
        'result': result,
        'winner': winner
    }