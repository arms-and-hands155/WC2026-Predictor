from dataclasses import dataclass
from src.features import build_features, update_team_history, update_h2h
import joblib
import numpy as np
import copy
import pandas as pd
from src.elo import update_elo


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
                  scalar,
                  squad_values,
                  home_elo,
                  away_elo):
    
    model_features = ['elo_diff', 'home_elo', 'away_elo', 'home_form', 'away_form', 'h2h', 'home_gd', 'away_gd', 'squad_value_diff']
    X = build_features(home_team, away_team, history_dict, h2h_dict, country_elo, squad_values, home_elo, away_elo)
    X=X[model_features]
    X_scaled = scalar.transform(X)
    
    probability = model.predict_proba(X_scaled)[0]
    
    calibration_factor = .25
    lambda_h = goal_model_h.predict(X_scaled)[0] * calibration_factor
    lambda_a = goal_model_a.predict(X_scaled)[0] * calibration_factor
    
    h_goals = np.random.poisson(lambda_h)
    a_goals = np.random.poisson(lambda_a)
    
    outcome_idx = np.random.choice([0, 1, 2], p=probability)
    
    diff = probability[2] - probability[0]
    
    # if abs(diff) < draw_threshold:
    #     avg_goals = np.random.poisson((lambda_h + lambda_a) / 2)
    #     h_goals = a_goals = avg_goals
    #     outcome, hp, ap = "Draw", 1, 1
    #     prob = probability[1]
        
    # elif diff > 0:
    #     if h_goals <= a_goals: h_goals = a_goals + 1 #incase lambdas give out incorrect numbers in correspondence to result
    #     outcome, hp, ap = "Home Win", 3, 0
    #     prob = probability[2]
    # else:
    #     if a_goals <= h_goals: a_goals = h_goals + 1
    #     outcome, hp, ap = "Away Win", 0, 3
    #     prob = probability[0]
    
    if outcome_idx == 2: #Home_win
        outcome, hp, ap = "Home Win", 3, 0
        if h_goals <= a_goals:
            h_goals = a_goals + 1
    elif outcome_idx == 0:        # Away win
        outcome, hp, ap = "Away Win", 0, 3
        if a_goals <= h_goals:
            a_goals = h_goals + 1
    else:                         # Draw
        outcome, hp, ap = "Draw", 1, 1
        avg = (h_goals + a_goals) // 2
        h_goals = a_goals = avg

    prob = probability[outcome_idx]
    
    return Matchprediction(outcome, hp, ap, round(prob, 4), round(abs(diff), 4), int(h_goals), int(a_goals))

def assign_thirds(thirds_slot_map, available_thirds):
    assignments = {}
    used = set()
    remaining = dict(thirds_slot_map)

    while remaining:
        winner_group, eligible_groups = min(
            remaining.items(),
            key=lambda item: sum(1 for g in item[1] if g in available_thirds and g not in used)
        )

        for g in eligible_groups:
            if g in available_thirds and g not in used:
                assignments[winner_group] = available_thirds[g]
                used.add(g)
                break
        else:
            # Fallback: grab any remaining unassigned third
            remaining_thirds = {g: t for g, t in available_thirds.items() if g not in used}
            if remaining_thirds:
                g, team = next(iter(remaining_thirds.items()))
                assignments[winner_group] = team
                used.add(g)

        del remaining[winner_group]

    return assignments
    


def run_tournament(wc_model, draw_threshold, history_dict, h2h_dict,
                   country_elo, model_h, model_a, scaler, group_stage_matches):

    # Deep copy mutable state so each simulation starts fresh
    country_elo  = copy.deepcopy(country_elo)
    history_dict = copy.deepcopy(history_dict)
    h2h_dict     = copy.deepcopy(h2h_dict)

    wc_groups = {
    "A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

    squad_values = {
        "Mexico": 226, "South Korea": 154, "South Africa": 50, "Czech Republic": 200,
        "Canada": 230, "Switzerland": 360, "Qatar": 23, "Bosnia and Herzegovina": 80,
        "Brazil": 1100, "Morocco": 350, "Haiti": 10, "Scotland": 200,
        "United States": 310, "Paraguay": 100, "Australia": 40, "Turkey": 320,
        "Germany": 850, "Curaçao": 5, "Ivory Coast": 300, "Ecuador": 230,
        "Netherlands": 600, "Japan": 290, "Sweden": 210, "Tunisia": 45,
        "Belgium": 450, "Egypt": 130, "Iran": 50, "New Zealand": 20,
        "Spain": 1000, "Cape Verde": 30, "Saudi Arabia": 30, "Uruguay": 480,
        "France": 1200, "Senegal": 250, "Iraq": 10, "Norway": 450,
        "Argentina": 800, "Algeria": 180, "Austria": 240, "Jordan": 15,
        "Portugal": 1050, "DR Congo": 110, "Uzbekistan": 35, "Colombia": 280,
        "England": 1500, "Croatia": 300, "Ghana": 200, "Panama": 20
    }
    #--------------GROUP STAGE--------------
    rows=[]
    for group, teams in wc_groups.items():
        for team in teams:
            rows.append({"Group": group,
                        "Team": team,
                        "Points": 0,
                        "GD": 0,
                        'GF': 0,
                        "GA": 0})
            
    group_stage_result = pd.DataFrame(rows) #Gives us the starting group stage
    for match in group_stage_matches.itertuples(index = False):
        
        
        if match.home_team not in country_elo or pd.isna(country_elo[match.home_team]):
            country_elo[match.home_team] = 1500.0
        if match.away_team not in country_elo or pd.isna(country_elo[match.away_team]):
            country_elo[match.away_team] = 1500.0
        
        x = predict_match(match.home_team, match.away_team, wc_model, draw_threshold, history_dict, h2h_dict, country_elo, model_h, model_a, scaler, squad_values, country_elo[match.home_team], country_elo[match.away_team])
        #print(f"{match.home_team} vs {match.away_team} ({x.home_score}-{x.away_score}, result: {x.outcome}), Prob: {x.probability}, diff: {x.diff}")
        group_stage_result.loc[group_stage_result['Team'] == match.home_team, 'Points'] += x.home_points
        group_stage_result.loc[group_stage_result['Team'] == match.away_team, 'Points'] += x.away_points
        
        group_stage_result.loc[group_stage_result['Team'] == match.home_team, 'GD'] += x.home_score - x.away_score
        group_stage_result.loc[group_stage_result['Team'] == match.away_team, 'GD'] += x.away_score - x.home_score
        
        group_stage_result.loc[group_stage_result['Team'] == match.home_team, 'GF'] += x.home_score
        group_stage_result.loc[group_stage_result['Team'] == match.away_team, 'GF'] += x.away_score
        
        group_stage_result.loc[group_stage_result['Team'] == match.home_team, 'GA'] += x.away_score
        group_stage_result.loc[group_stage_result['Team'] == match.away_team, 'GA'] += x.home_score
        
        if x.outcome == "Home Win": S = 1
        elif x.outcome == "Away Win": S = 0
        else: S = 0.5
        new_away, new_home = update_elo(S, match.neutral, match.K_factor,x.home_score, x.away_score, 
                                        country_elo[match.away_team], 
                                        country_elo[match.home_team])
        
        country_elo[match.home_team] = new_home
        country_elo[match.away_team] = new_away
        update_team_history(match, x.home_score, x.away_score, history_dict)
        update_h2h(match, h2h_dict)
        
    group_stage_result = group_stage_result.sort_values(['Group', 'Points', 'GD', 'GF'], ascending=[True, False, False, False]).reset_index(drop=True)    

    #--------------ROUND OF 32---------------
    
    top2 = group_stage_result.groupby('Group').head(2).copy() #Teams that placed top 2 in their group which qualifies
    third = group_stage_result.groupby('Group').nth(2).copy() #All teams that placed third (only 8 of them move on)

    best8_third = third.sort_values(
        ['Points', 'GD', 'GF'], 
        ascending=[False, False, False]
    ).head(8)

    thirds_slot_map = { #Needs to be fixed
        'E': 'ABCDEFGHIJKL',
        'I': 'ABCDEFGHIJKL', 
        'A': 'ABCDEFGHIJKL',
        'L': 'ABCDEFGHIJKL',
        'D': 'ABCDEFGHIJKL',
        'G': 'ABCDEFGHIJKL',
        'B': 'ABCDEFGHIJKL',
        'K': 'ABCDEFGHIJKL',
    }

    winners = {g: teams.iloc[0]['Team'] for g, teams in group_stage_result.groupby('Group')}
    runners = {g: teams.iloc[1]['Team'] for g, teams in group_stage_result.groupby('Group')}

    available_thirds = {row.Group: row.Team for row in best8_third.itertuples()}
    assignments = {}
    used = set()
    for winner_group, eligible_groups in thirds_slot_map.items():
        for g in eligible_groups:
            if g in available_thirds and g not in used:
                assignments[winner_group] = available_thirds[g]
                used.add(g)
                break

    r32_matchups = {
        73: (runners['A'], runners['B']),
        74: (winners['E'], assignments['E']),
        75: (winners['F'], runners['C']),
        76: (winners['C'], runners['F']),
        77: (winners['I'], assignments['I']),
        78: (runners['E'], runners['I']),
        79: (winners['A'], assignments['A']),
        80: (winners['L'], assignments['L']),
        81: (winners['D'], assignments['D']),
        82: (winners['G'], assignments['G']),
        83: (runners['K'], runners['L']),
        84: (winners['H'], runners['J']),
        85: (winners['B'], assignments['B']),
        86: (winners['J'], runners['H']),
        87: (winners['K'], assignments['K']),
        88: (runners['D'], runners['G'])
    }

    r32_rows = []
    r16_teams = {}

    for match, teams in r32_matchups.items():
        #teams[0] = home team, team[1] = away team
        x=predict_match(teams[0], teams[1], wc_model, draw_threshold, history_dict, h2h_dict, country_elo, model_h, model_a, scaler, squad_values, country_elo[teams[0]], country_elo[teams[1]])
        
        if x.outcome == "Home Win": 
            S, winner = 1, teams[0]
        elif x.outcome == "Away Win": 
            S, winner = 0, teams[1]
        else: 
            S = .5
            winner = teams[0] if x.diff > 0 else teams[1]

        r16_teams[match] = winner
        r32_rows.append({
            'home_team': teams[0],
            'away_team': teams[1],
            'home_score': x.home_score,
            'away_score': x.away_score,
            'result': S,
            'neutral': 1,
            'K_factor': 50,
            'winner': winner
        })
        

        new_away, new_home = update_elo(S, 1, 50, x.home_score, x.away_score, 
                                        country_elo[teams[0]], 
                                        country_elo[teams[1]])
        
        country_elo[teams[0]] = new_home
        country_elo[teams[1]] = new_away

    #------------ROUND OF 16-----------
    r16_matchups = {
    89: (r16_teams[74], r16_teams[77]),
    90: (r16_teams[73], r16_teams[75]),
    91: (r16_teams[76], r16_teams[78]),
    92: (r16_teams[79], r16_teams[80]),
    93: (r16_teams[83], r16_teams[84]),
    94: (r16_teams[81], r16_teams[82]),
    95: (r16_teams[86], r16_teams[88]),
    96: (r16_teams[85], r16_teams[87])
}

    r16_rows = []
    r8_teams = {}

    for match, teams in r16_matchups.items():
        #teams[0] = home team, team[1] = away team
        x=predict_match(teams[0], teams[1], wc_model, draw_threshold, history_dict, h2h_dict, country_elo, model_h, model_a, scaler, squad_values, country_elo[teams[0]], country_elo[teams[1]])

        if x.outcome == "Home Win": 
            S, winner = 1, teams[0]
        elif x.outcome == "Away Win": 
            S, winner = 0, teams[1]
        else: 
            S = .5
            winner = teams[0] if x.diff > 0 else teams[1]

        r8_teams[match] = winner
        r16_rows.append({
            'home_team': teams[0],
            'away_team': teams[1],
            'home_score': x.home_score,
            'away_score': x.away_score,
            'result': S,
            'neutral': 1,
            'K_factor': 50,
            'winner': winner
        })
        

        new_away, new_home = update_elo(S, 1, 50, x.home_score, x.away_score, 
                                        country_elo[teams[0]], 
                                        country_elo[teams[1]])
        
        country_elo[teams[0]] = new_home
        country_elo[teams[1]] = new_away
    
    #---------QUARTER FINALS-----------
    QF_matchups = {
  97: (r8_teams[89], r8_teams[90]),
  98: (r8_teams[93], r8_teams[94]), 
  99: (r8_teams[91], r8_teams[92]), 
  100: (r8_teams[95], r8_teams[96]), 
}

    QF_rows = []
    SF_teams = {}

    for match, teams in QF_matchups.items():
        #teams[0] = home team, team[1] = away team
        x=predict_match(teams[0], teams[1], wc_model, draw_threshold, history_dict, h2h_dict, country_elo, model_h, model_a, scaler, squad_values, country_elo[teams[0]], country_elo[teams[1]])

        if x.outcome == "Home Win": 
            S, winner = 1, teams[0]
        elif x.outcome == "Away Win": 
            S, winner = 0, teams[1]
        else: 
            S = .5
            winner = teams[0] if x.diff > 0 else teams[1]

        SF_teams[match] = winner
        QF_rows.append({
            'home_team': teams[0],
            'away_team': teams[1],
            'home_score': x.home_score,
            'away_score': x.away_score,
            'result': S,
            'neutral': 1,
            'K_factor': 50,
            'winner': winner
        })
        

        new_away, new_home = update_elo(S, 1, 50, x.home_score, x.away_score, 
                                        country_elo[teams[0]], 
                                        country_elo[teams[1]])
        
        country_elo[teams[0]] = new_home
        country_elo[teams[1]] = new_away
        
    #---------SEMI FINALS------------
    SF_matchups = {
  101: (SF_teams[97], SF_teams[98]),
  102: (SF_teams[99], SF_teams[100]) 
}

    SF_rows = []
    FINAL_teams = {}

    for match, teams in SF_matchups.items():
        #teams[0] = home team, team[1] = away team
        x=predict_match(teams[0], teams[1], wc_model, draw_threshold, history_dict, h2h_dict, country_elo, model_h, model_a, scaler, squad_values, country_elo[teams[0]], country_elo[teams[1]])

        if x.outcome == "Home Win": 
            S, winner = 1, teams[0]
        elif x.outcome == "Away Win": 
            S, winner = 0, teams[1]
        else: 
            S = .5
            winner = teams[0] if x.diff > 0 else teams[1]

        FINAL_teams[match] = winner
        SF_rows.append({
            'home_team': teams[0],
            'away_team': teams[1],
            'home_score': x.home_score,
            'away_score': x.away_score,
            'result': S,
            'neutral': 1,
            'K_factor': 50,
            'winner': winner
        })
        

        new_away, new_home = update_elo(S, 1, 50, x.home_score, x.away_score, 
                                        country_elo[teams[0]], 
                                        country_elo[teams[1]])
        
        country_elo[teams[0]] = new_home
        country_elo[teams[1]] = new_away
        
    #-----------FINALS-----------
    for match, teams in {103: (FINAL_teams[101], FINAL_teams[102])}.items():
        #teams[0] = home team, team[1] = away team
        x=predict_match(teams[0], teams[1], wc_model, draw_threshold, history_dict, h2h_dict, country_elo, model_h, model_a, scaler, squad_values, country_elo[teams[0]], country_elo[teams[1]])

        if x.outcome == "Home Win": 
            S, winner = 1, teams[0]
        elif x.outcome == "Away Win": 
            S, winner = 0, teams[1]
        else: 
            S = .5
            winner = teams[0] if x.diff > 0 else teams[1]

    return winner

