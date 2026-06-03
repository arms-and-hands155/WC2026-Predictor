from dataclasses import dataclass
from src.features import build_features
import joblib
import numpy as np

goal_model_h = joblib.load('home_goals_model.joblib')
goal_model_a = joblib.load('away_goals_model.joblib')
classifier = joblib.load('world_cup_model.joblib')

@dataclass
class  Matchprediction:
    outcome: str        # "Home Win", "Draw", "Away Win"
    home_points: int
    away_points: int
    probability: float  
    diff: float
    home_score: int
    away_score: int

def predict_match(home_team, 
                  away_team, 
                  model, 
                  draw_threshold, 
                  history_dict, 
                  h2h_dict,
                  country_elo, 
                  goal_model_h,
                  goal_model_a,
                  scalar):
    
    model_features = ['elo_diff', 'home_form', 'away_form', 'h2h', 'home_gd', 'away_gd']
    X = build_features(home_team, away_team, history_dict, h2h_dict, country_elo )
    X=X[model_features]
    X_scaled = scalar.transform(X)
    
    probability = model.predict_proba(X_scaled)[0]
    
    calibration_factor = .4 #Model was predicting too many goals so i had to scale down
    lambda_h = goal_model_h.predict(X_scaled)[0] * calibration_factor
    lambda_a = goal_model_a.predict(X_scaled)[0] * calibration_factor
    
    lambda_h = max(0.5, lambda_h)
    lambda_a = max(0.5, lambda_a)
    
    h_goals = np.random.poisson(lambda_h)
    a_goals = np.random.poisson(lambda_a)
    
    diff = probability[2] - probability[0]
    
    if abs(diff) < draw_threshold:
        h_goals = a_goals = (1 if (lambda_h + lambda_a)/2 > 1.2 else 0)
        outcome, ap, hp = "Draw", 1, 1
        prob = probability[1]
    elif diff > 0:
        if h_goals <= a_goals: h_goals = a_goals + 1 #incase lambdas give out incorrect numbers in correspondence to result
        outcome, hp, ap = "Home Win", 3, 0
        prob = probability[2]
    else:
        if a_goals <= h_goals: a_goals = h_goals + 1
        outcome, hp, ap = "Away Win", 0, 3
        prob = probability[0]
    return Matchprediction(outcome, hp, ap, round(prob, 4), round(abs(diff), 4), int(h_goals), int(a_goals))

def assign_thirds(best8_third, thirds_slot_map):

    available_thirds = {row['Group']: row['Team'] 
                        for _, row in best8_third.iterrows()}
    
    assignments = {}  # winner_group -> third_place_team
    used = set()
    
    for winner_group, eligible_groups in thirds_slot_map.items():
        for g in eligible_groups:
            if g in available_thirds and g not in used:
                assignments[winner_group] = available_thirds[g]
                used.add(g)
                break
    
    return assignments
    
    
    
    