import pandas as pd
from src.elo import update_elo, assign_k, get_result
from collections import defaultdict
from itertools import combinations

def rolling_form(df, team, date):
    past = df[(df['home_team']==team) | (df['away_team']==team)] #Df with all the games played by a team regardless of home or away
    past = past[past['date']<date].tail(10) #Get the last ten games before the wc
    points = 0
    for _ in past.itertuples(index=False):
        if _.home_team == team:
            if _.result == 1: points += 3
            elif _.result == .5: points += 1
        else:
            if _.result == 0: points += 3
            elif _.result == .5: points += 1
    
    return points/10 #Gives average pts per game over the last 10 games

def h2h_last_five(df, team1, team2):
    past = df[
        ((df['home_team'] == team1) | (df['away_team'] == team1)) & 
        ((df['home_team'] == team2) | (df['away_team'] == team2))]   
    
    wins = 0
    for _ in past.itertuples(index=False):
        if _.home_team == team1 and _.result == 1: wins += 1
        elif _.away_team == team1 and _.result == 0: wins += 1
    return round(wins / len(past), 2) if len(past) > 0 else 0.5, 1- round(wins / len(past), 2) if len(past) > 0 else 0.5

def average_goal_diff(df, team):
    past = df[
        ((df['home_team'] == team) | (df['away_team'] == team))]  
    gd = 0
    for _ in past.tail(10).itertuples(index=False):
        if _.home_team == team:
            gd = gd+_.home_score-_.away_score
        else:
            gd = gd+_.away_score-_.home_score
    return gd / 10

def elo_diff(country_elo,home_team, away_team):
    return country_elo[home_team] - country_elo[away_team]

def build_prediction_features(df, country_elo, wc_teams, cutoff):
    rows = []
    for home_team, away_team in combinations(wc_teams, 2):
        rows.append({
            "home_team": home_team,
            "away_team": away_team,
            "elo_diff": elo_diff(country_elo, home_team, away_team),
            "home_form": rolling_form(df, home_team, cutoff),
            "away_form": rolling_form(df, away_team, cutoff),
            "h2h": h2h_last_five(df, home_team, away_team),
            "home_gd": average_goal_diff(df, home_team),
            "away_gd": average_goal_diff(df, away_team),
            #Need to fix scenario where mexico plays in US since they aren't hosts
            "neutral": 0 if home_team in ["United States", "Mexico", "Canada"] or away_team in ["United States", "Mexico", "Canada"] else 1
        })
    return pd.DataFrame(rows)
    