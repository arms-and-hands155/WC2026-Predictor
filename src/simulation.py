from dataclasses import dataclass
from src.features import build_features, assign_third_place_slots, update_team_history, update_h2h
import joblib
import numpy as np
import copy
import pandas as pd
from src.elo import update_elo
from scipy.stats import poisson


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

def predict_match(home_team, away_team, model_home, model_away, team_to_confederation, country_elo, feature):
    X = build_features(home_team, away_team, country_elo, team_to_confederation, feature)
    
    home_goals_predict = model_home.predict(X)[0]
    away_goals_predict = model_away.predict(X)[0]
    
    score_probability = []
    
    home_win = 0
    away_win = 0
    draw = 0
    
    for home_goals in range(9):
        for away_goals in range(9):
            
            prob = poisson.pmf(away_goals, away_goals_predict) * poisson.pmf(home_goals, home_goals_predict)
            
            score_probability.append({
                'home_goals': home_goals,
                'away_goals': away_goals,
                'probability': prob
            })
            
            if home_goals > away_goals:
                home_win += prob
            elif home_goals == away_goals:
                draw += prob
            else:
                away_win += prob
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_xg': home_goals_predict,
        'away_xg': away_goals_predict,
        'home_win': home_win,
        'away_win': away_win,
        'draw': draw,
        'score_probs': pd.DataFrame(score_probability)
    }
            
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
    #Squad value of countries based off trasnfermarket (in euros)
    squad_values = {
        "Mexico": 191, "South Korea": 139, "South Africa": 50, "Czech Republic": 188,
        "Canada": 196, "Switzerland": 332, "Qatar": 22, "Bosnia and Herzegovina": 151,
        "Brazil": 923, "Morocco": 498, "Haiti": 55, "Scotland": 170,
        "United States": 385, "Paraguay": 153, "Australia": 77, "Turkey": 473,
        "Germany": 947, "Curaçao": 25, "Ivory Coast": 522, "Ecuador": 368,
        "Netherlands": 804, "Japan": 270, "Sweden": 406, "Tunisia": 70,
        "Belgium": 547, "Egypt": 116, "Iran": 32, "New Zealand": 34,
        "Spain": 1220, "Cape Verde": 54, "Saudi Arabia": 40, "Uruguay": 360,
        "France": 1520, "Senegal": 478, "Iraq": 21, "Norway": 589,
        "Argentina": 782, "Algeria": 256, "Austria": 242, "Jordan": 20,
        "Portugal": 1010, "DR Congo": 140, "Uzbekistan": 85, "Colombia": 300,
        "England": 1360, "Croatia": 387, "Ghana": 234, "Panama": 34
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
        # new_away, new_home = update_elo(S, match.neutral, match.K_factor,x.home_score, x.away_score, 
        #                                 country_elo[match.away_team], 
        #                                 country_elo[match.home_team])
        
        # country_elo[match.home_team] = new_home
        # country_elo[match.away_team] = new_away
        # update_team_history(match, x.home_score, x.away_score, history_dict)
        # update_h2h(match, h2h_dict)
        
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
            p_home_wins_pens = 0.5 + (country_elo[teams[0]] - country_elo[teams[1]]) / 4000
            winner = teams[0] if np.random.random() < np.clip(p_home_wins_pens, 0.3, 0.7) else teams[1]

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
        

        # new_away, new_home = update_elo(S, 1, 50, x.home_score, x.away_score, 
        #                                 country_elo[teams[0]], 
        #                                 country_elo[teams[1]])
        
        # country_elo[teams[0]] = new_home
        # country_elo[teams[1]] = new_away

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
            p_home_wins_pens = 0.5 + (country_elo[teams[0]] - country_elo[teams[1]]) / 4000
            winner = teams[0] if np.random.random() < np.clip(p_home_wins_pens, 0.3, 0.7) else teams[1]

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
        

        # new_away, new_home = update_elo(S, 1, 50, x.home_score, x.away_score, 
        #                                 country_elo[teams[0]], 
        #                                 country_elo[teams[1]])
        
        # country_elo[teams[0]] = new_home
        # country_elo[teams[1]] = new_away
    
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
            p_home_wins_pens = 0.5 + (country_elo[teams[0]] - country_elo[teams[1]]) / 4000
            winner = teams[0] if np.random.random() < np.clip(p_home_wins_pens, 0.3, 0.7) else teams[1]

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
        

        # new_away, new_home = update_elo(S, 1, 50, x.home_score, x.away_score, 
        #                                 country_elo[teams[0]], 
        #                                 country_elo[teams[1]])
        
        # country_elo[teams[0]] = new_home
        # country_elo[teams[1]] = new_away
        
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
            p_home_wins_pens = 0.5 + (country_elo[teams[0]] - country_elo[teams[1]]) / 4000
            winner = teams[0] if np.random.random() < np.clip(p_home_wins_pens, 0.3, 0.7) else teams[1]

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
        

        # new_away, new_home = update_elo(S, 1, 50, x.home_score, x.away_score, 
        #                                 country_elo[teams[0]], 
        #                                 country_elo[teams[1]])
        
        # country_elo[teams[0]] = new_home
        # country_elo[teams[1]] = new_away
        
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
            p_home_wins_pens = 0.5 + (country_elo[teams[0]] - country_elo[teams[1]]) / 4000
            winner = teams[0] if np.random.random() < np.clip(p_home_wins_pens, 0.3, 0.7) else teams[1]

    return winner

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
        home_team, away_team, home_goal_model, away_goal_model, team_to_confederation, country_elo, feature)
    
    h_goals = np.random.poisson(predict['home_xg'])
    a_goals = np.random.poisson(predict['away_xg'])
    
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
    
def create_empty_group_table(group_teams):
    table = pd.DataFrame({
        "team": group_teams,
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_difference": 0,
        "points": 0
    })

    return table

def update_group_table(table, match):

    home_team = match["home_team"]
    away_team = match["away_team"]
    home_goals = match["home_goals"]
    away_goals = match["away_goals"]

    # Update played
    table.loc[table["team"] == home_team, "played"] += 1
    table.loc[table["team"] == away_team, "played"] += 1

    # Update goals
    table.loc[table["team"] == home_team, "goals_for"] += home_goals
    table.loc[table["team"] == home_team, "goals_against"] += away_goals

    table.loc[table["team"] == away_team, "goals_for"] += away_goals
    table.loc[table["team"] == away_team, "goals_against"] += home_goals

    # Update result stats and points
    if home_goals > away_goals:
        table.loc[table["team"] == home_team, "wins"] += 1
        table.loc[table["team"] == away_team, "losses"] += 1

        table.loc[table["team"] == home_team, "points"] += 3

    elif home_goals < away_goals:
        table.loc[table["team"] == away_team, "wins"] += 1
        table.loc[table["team"] == home_team, "losses"] += 1

        table.loc[table["team"] == away_team, "points"] += 3

    else:
        table.loc[table["team"] == home_team, "draws"] += 1
        table.loc[table["team"] == away_team, "draws"] += 1

        table.loc[table["team"] == home_team, "points"] += 1
        table.loc[table["team"] == away_team, "points"] += 1

    # Recalculate goal difference
    table["goal_difference"] = table["goals_for"] - table["goals_against"]

    return table

def rank_group_table(table):
    table = table.copy()

    # Temporary random tie-breaker for exact ties
    table["random_tiebreaker"] = np.random.random(len(table))

    table = (
        table
        .sort_values(
            by=["points", "goal_difference", "goals_for", "random_tiebreaker"],
            ascending=[False, False, False, False]
        )
        .reset_index(drop=True)
    )

    table["group_rank"] = table.index + 1

    table = table.drop(columns=["random_tiebreaker"])

    return table

def simulate_group(group_name, df_groups, df_group_fixtures, home_goal_model, away_goal_model, 
            country_elo, 
            team_to_confederation, 
            feature):
    group_teams = (
        df_groups[df_groups["group"] == group_name]
        .sort_values("position")["nation"]
        .tolist()
    )

    group_matches = df_group_fixtures[df_group_fixtures["group"] == group_name]

    table = create_empty_group_table(group_teams)

    simulated_matches = []

    for _, row in group_matches.iterrows():
        match = simulate_match(
            row["home_team"],
            row["away_team"],
            home_goal_model,
            away_goal_model, 
            country_elo, 
            team_to_confederation, 
            feature
        )

        simulated_matches.append(match)
        table = update_group_table(table, match)

    ranked_table = rank_group_table(table)
    ranked_table["group"] = group_name

    return ranked_table, pd.DataFrame(simulated_matches)

def simulate_group_stage(df_groups, df_group_fixture, home_goal_model, away_goal_model, 
            country_elo, 
            team_to_confederation, 
            feature):

    all_group_tables = []
    all_group_matches = []

    for group_name in sorted(df_groups["group"].unique()):
        group_table, group_matches = simulate_group(group_name, 
                                                    df_groups,
                                                    df_group_fixture, 
                                                    home_goal_model, 
                                                    away_goal_model, 
                                                    country_elo, 
                                                    team_to_confederation, 
                                                    feature)

        all_group_tables.append(group_table)
        all_group_matches.append(group_matches)

    df_group_tables = pd.concat(all_group_tables, ignore_index=True)
    df_group_matches = pd.concat(all_group_matches, ignore_index=True)

    return df_group_tables, df_group_matches

def construct_round32(teams_dict):
    r32_matchups = {
    73: (teams_dict['2A'], teams_dict['2B']),
    74: (teams_dict['1E'], teams_dict['3ABCDF']),
    75: (teams_dict['1F'], teams_dict['2C']),
    76: (teams_dict['1C'], teams_dict['2F']),
    77: (teams_dict['1I'], teams_dict['3CDFGH']),
    78: (teams_dict['2E'], teams_dict['2I']),
    79: (teams_dict['1A'], teams_dict['3CEFHI']),
    80: (teams_dict['1L'], teams_dict['3EHIJK']),
    81: (teams_dict['1D'], teams_dict['3BEFIJ']),
    82: (teams_dict['1G'], teams_dict['3AEHIJ']),
    83: (teams_dict['2K'], teams_dict['2L']),
    84: (teams_dict['1H'], teams_dict['2J']),
    85: (teams_dict['1B'], teams_dict['3EFGIJ']),
    86: (teams_dict['1J'], teams_dict['2H']),
    87: (teams_dict['1K'], teams_dict['3DEIJL']),
    88: (teams_dict['2D'], teams_dict['2G'])
}
    
    return r32_matchups

def construct_round16(winners):
    r16_matchups = {
    89: (winners[74], winners[77]),
    90: (winners[73], winners[75]),
    91: (winners[76], winners[78]),
    92: (winners[79], winners[80]),
    93: (winners[83], winners[84]),
    94: (winners[81], winners[82]),
    95: (winners[86], winners[88]),
    96: (winners[85], winners[87])
}
    
    return r16_matchups

def construct_QF(winners):
    QF_matchups = {
  97: (winners[89], winners[90]),
  98: (winners[93], winners[94]), 
  99: (winners[91], winners[92]), 
  100: (winners[95], winners[96]), 
}
    
    return QF_matchups

def construct_SF(SF_teams):
    SF_matchups = {
    101: (SF_teams[97], SF_teams[98]),
    102: (SF_teams[99], SF_teams[100]) 
    }
    
    return SF_matchups


def simulate_knockout_match(home_team, away_team, home_goal_model, away_goal_model, country_elo, team_to_confederation, feature):

    predict = predict_match(
        home_team, away_team, home_goal_model, away_goal_model, team_to_confederation, country_elo, feature)
    
    h_goals = np.random.poisson(predict['home_xg'])
    a_goals = np.random.poisson(predict['away_xg'])
    
    if h_goals > a_goals:
        result = 'home_win'
        winner, loser = home_team, away_team
        result_type = 'regular_time'
    elif a_goals > h_goals:
        result = 'away_win'
        winner, loser = away_team, home_team
        result_type = 'regular_time'
    else:
        home_strength = predict['home_win']
        away_strength = predict['away_win']
        result_type = 'OT/Pens'
        
        home_advance_prob = home_strength / (home_strength + away_strength)

        if np.random.random() < home_advance_prob:
            result = 'home_win'
            winner = home_team
            loser = away_team
        else:
            result = 'away_win'
            winner = away_team
            loser = home_team
            

    
    return {
        "home_team": home_team,
        'away_team': away_team,
        'home_goals': h_goals,
        'away_goals': a_goals,
        'result': result,
        'result_type': result_type,
        'winner': winner,
        'loser': loser,
        'home_win_prob': predict['home_win'],
        'draw_prob': predict['draw'],
        'away_win_prob': predict['away_win']
    }

def simulate_knockout_round(teams_dict, home_goal_model, away_goal_model, country_elo, team_to_confederation, feature):
    results= []
    winners = {}
    
    for k,v in teams_dict.items():
        match_result = simulate_knockout_match(
            v[0], v[1],home_goal_model, away_goal_model, country_elo, team_to_confederation, feature)
        
        results.append({
            'home_team': v[0],
            'away_team': v[1],
            'home_goals': match_result['home_goals'],
            'away_goals': match_result['away_goals'],
            'result_type': match_result['result_type'],
            'winner': match_result['winner'],
            'loser': match_result['loser'],
            'home_win_prob': match_result['home_win_prob'],
            'draw_prob': match_result['draw_prob'],
            'away_win_prob': match_result['away_win_prob']
        })

        result = pd.DataFrame(results)
        winners[k] = match_result['winner']
        
    return winners, result

def simulate_tournament(home_goal_model, away_goal_model, country_elo, team_to_confederation, feature, df_groups, df_group_fixture):
    
    df_group_stage = simulate_group_stage(df_groups, 
                                                             df_group_fixture, 
                                                             home_goal_model, 
                                                             away_goal_model, 
                                                             country_elo, 
                                                             team_to_confederation, 
                                                             feature)
    
    group_stage_result = df_group_stage[0]
    
    top2 = group_stage_result.groupby('group').head(2).copy() #Teams that placed top 2 in their group which qualifies
    third = group_stage_result.groupby('group').nth(2).copy() #All teams that placed third (only 8 of them move on)
    best8_third = third.sort_values(
        ['points', 'goal_difference', 'goals_for'], 
        ascending=[False, False, False]
    ).head(8).reset_index(drop=True)

    best8_third['third_place_rank'] = best8_third.index + 1
    
    round_of_32 = pd.concat([top2, best8_third])
    
    rd32_teams = {}
    for team in top2.itertuples(index=False):
            rd32_teams[f'{team.group_rank}{team.group}'] = team.team
            
    thirds = assign_third_place_slots(best8_third)
    rd32_teams |= thirds

    x = construct_round32(rd32_teams)
    
    r32_winners, r32_results = simulate_knockout_round(x, home_goal_model, away_goal_model, country_elo, team_to_confederation, feature)
    
    r16_matchups = construct_round16(r32_winners)

    r16_winners, r16_results = simulate_knockout_round(r16_matchups, home_goal_model, away_goal_model, country_elo, team_to_confederation, feature)
    
    QF_matchsups = construct_QF(r16_winners)

    QF_winners, QF_results = simulate_knockout_round(QF_matchsups, home_goal_model, away_goal_model, country_elo, team_to_confederation, feature)
    
    SF_matchsups = construct_SF(QF_winners)

    SF_winners, SF_results = simulate_knockout_round(SF_matchsups, home_goal_model, away_goal_model, country_elo, team_to_confederation, feature)
    
    winner, results = simulate_knockout_round({103: (SF_winners[101], SF_winners[102])}, home_goal_model, away_goal_model, country_elo, team_to_confederation, feature)
    
    summary = {
        "winner": results['winner'].iloc[0],
        "runner_up": results['loser'].iloc[0],
        "r32_teams": list(rd32_teams.values()),
        "r16_teams": list(r32_winners.values()),
        "qf_teams": list(r16_winners.values()),
        "sf_teams": list(QF_winners.values()),
        "final_teams": list(SF_winners.values())
    }
    
    r32_results['round'] = 'R32'
    r16_results['round'] = 'R16'
    QF_results['round'] = 'QF'
    SF_results['round'] = 'SF'
    results['round'] = 'Final'
    
    
    return {
        "summary": summary,
        "group_tables": group_stage_result,
        "group_matches": df_group_stage[1],
        "rounds": pd.DataFrame(x),
        'bracket': pd.concat([r32_results, r16_results, QF_results, SF_results, results], ignore_index=True)
    }
    
    

    
    