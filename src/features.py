import pandas as pd
from collections import defaultdict
from collections import deque
import numpy as np
import statsmodels.api as sm

team_history = defaultdict(lambda: deque(maxlen=10))

def update_team_history(row, home_score, away_score, team_history):
    # Home Team Stats
    h_pts = 3 if row.result == 1 else (1 if row.result == 0.5 else 0)
    team_history[row.home_team].append((h_pts, home_score - away_score))
    
    # Away Team Stats
    a_pts = 3 if row.result == 0 else (1 if row.result == 0.5 else 0)
    team_history[row.away_team].append((a_pts, away_score - home_score))


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
    
    h2h_dict.setdefault(matchup, []).append(outcome)

def get_h2h_stat(home_team, away_team, h2h_dict):
    matchup = frozenset([home_team, away_team])
    history = h2h_dict.get(matchup)
    
    if not history:
        return 0.5
    
    avg_outcome = sum(history) / len(history)
    
    leader = min(home_team, away_team)
    if home_team == leader:
        return round(avg_outcome, 2)
    else:
        return round(1 - avg_outcome, 2)


def build_features(home_team, away_team, country_elo, team_to_confederation, feature, neutral = True, tournament_weight=5):
    home_elo = country_elo[home_team]
    away_elo = country_elo[away_team]
    
    home_conf = team_to_confederation[home_team]
    away_conf = team_to_confederation[away_team]
    
    if home_elo is None:
        home_elo = 1500.0

    if away_elo is None:
        home_elo = 1500.0

    if home_conf is None:
        home_conf = 'Unknown'

    if away_conf is None:
        home_conf = 'Unknown'

    row = pd.DataFrame([{
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "tournament_weight": tournament_weight,
        "neutral": int(neutral),
        "home_confederation": home_conf,
        "away_confederation": away_conf
    }])
    
    row = row.reindex(columns=feature, fill_value=0)
    
    for col in feature:
        if col.startswith('home_confederation_'):
            row[col] = 1.0 if home_conf == col.replace('home_confederation_', '') else 0.0
        elif col.startswith('away_confederation_'):
            row[col] = 1.0 if away_conf == col.replace('away_confederation_', '') else 0.0

    if 'const' in row.columns:
        row['const'] = 1.0

    return row.astype(float)

from itertools import permutations

def assign_third_place_slots(best_third_placed, third_place_slots=['3ABCDF', '3CDFGH', '3CEFHI', '3EHIJK', '3BEFIJ', '3AEHIJ', '3EFGIJ', '3DEIJL']):

    third_teams = best_third_placed[["team", "group", "third_place_rank"]].copy()

    if len(third_place_slots) != len(third_teams):
        raise ValueError(
            f"Number of third-place slots ({len(third_place_slots)}) does not match "
            f"number of third-place teams ({len(third_teams)})."
        )

    # Try every possible order of the 8 third-placed teams
    for perm in permutations(third_teams.to_dict("records")):
        assignment = {}
        valid = True

        for slot, team_row in zip(third_place_slots, perm):
            allowed_groups = list(str(slot).replace("3", ""))

            if team_row["group"] not in allowed_groups:
                valid = False
                break

            assignment[slot] = team_row["team"]

        if valid:
            return assignment

    raise ValueError("No valid third-place assignment found.")


