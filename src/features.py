import pandas as pd
from src.elo import update_elo, assign_k, get_result
from collections import defaultdict
from collections import deque
from itertools import combinations

team_history = defaultdict(lambda: deque(maxlen=10))

def update_team_history(row, team_history):
    # Home Team Stats
    h_pts = 3 if row.result == 1 else (1 if row.result == 0.5 else 0)
    team_history[row.home_team].append((h_pts, row.home_score - row.away_score))
    
    # Away Team Stats
    a_pts = 3 if row.result == 0 else (1 if row.result == 0.5 else 0)
    team_history[row.away_team].append((a_pts, row.away_score - row.home_score))


def get_stats(team, history_dict):
    history = history_dict[team]
    if not history:
        return 0.0, 0.0
    
    n = len(history)
    total_gd = sum(float(match[1]) for match in history)
    total_pts = sum(float(match[0]) for match in history)
    
    return round(total_pts / n, 2), round(total_gd / n, 2)

def update_h2h(row, h2h_dict):
    matchup = frozenset([row.home_team, row.away_team])
    leader = min(row.home_team, row.away_team)
    
    if row.home_team == leader:
        outcome = 1 if row.result == 1 else (0.5 if row.result == 0.5 else 0)
    else:
        outcome = 1 if row.result == 0 else (0.5 if row.result == 0.5 else 0)
    
    h2h_dict[matchup].append(outcome)

def get_h2h_stat(home_team, away_team, h2h_dict):
    matchup = frozenset([home_team, away_team])
    history = h2h_dict[matchup]
    
    if not history:
        return 0.5
    
    avg_outcome = sum(history) / len(history)
    
    leader = min(home_team, away_team)
    if home_team == leader:
        return round(avg_outcome, 2)
    else:
        return round(1 - avg_outcome, 2)


features = ['elo_diff', 'home_form', 'away_form', 'h2h', 'home_gd', 'away_gd']


def build_features(home_team, away_team, history_dict, h2h_dict, country_elo):
    home_form, home_gd = get_stats(home_team, history_dict)
    away_form, away_gd = get_stats(away_team, history_dict)
    
    h2h = get_h2h_stat(home_team, away_team, h2h_dict)
    
    return pd.DataFrame([{
        'elo_diff': country_elo[home_team]-country_elo[away_team],
        'home_form': home_form,
        'away_form': away_form,
        'home_gd': home_gd,
        'away_gd': away_gd,
        'h2h': h2h
    }])