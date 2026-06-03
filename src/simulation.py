from dataclasses import dataclass
from src.features import build_features

@dataclass
class  Matchprediction:
    outcome: str        # "Home Win", "Draw", "Away Win"
    home_points: int
    away_points: int
    probability: float  
    diff: float

def predict_match(home_team, away_team, model, draw_threshold, history_dict, h2h_dict,country_elo):
    model_features = ['elo_diff', 'home_form', 'away_form', 'h2h', 'home_gd', 'away_gd']
    X = build_features(home_team, away_team, history_dict, h2h_dict, country_elo )
    X=X[model_features]
    
    probability = model.predict_proba(X)[0]
    
    diff = probability[2] - probability[0]
    
    if abs(diff) < draw_threshold:
        outcome, ap, hp = "Draw", 1, 1
        prob = probability[1]
    elif diff > 0:
        outcome, hp, ap = "Home Win", 3, 0
        prob = probability[2]
    else:
        outcome, hp, ap = "Away Win", 0, 3
        prob = probability[0]
    
    return Matchprediction(outcome, hp, ap, round(prob, 4), round(abs(diff), 4))
    
    
    