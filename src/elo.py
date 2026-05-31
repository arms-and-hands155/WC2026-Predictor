import pandas as pd
import numpy as np


def get_result(home_score, away_score): 
    if home_score > away_score:
        return 1 #Home win
    elif away_score > home_score:
        return 0 #Away win
    else:
        return 0.5 #Draw
    
def assign_k(tournament): #Adding a weight constant column for future ELO caculations
    t = tournament.lower()
    
    # Tier 1: World Cup Finals
    if 'fifa world cup' in t and 'qualification' not in t:
        return 60
    
    # Tier 2: Continental Finals
    major_finals = ['uefa euro', 'copa américa', 'african cup of nations', 'confederations cup', 'asian cup']
    if any(x in t for x in major_finals) and 'qualification' not in t:
        return 50
    
    # Tier 3: Qualifiers and Nations League
    if 'qualification' in t or 'nations league' in t:
        return 40
    
    # Tier 4: Friendlies
    if 'friendly' in t:
        return 20
    
    # Tier 5: Everything else (Minor cups/tournaments)
    return 30


#Elo Caculation
def update_elo(S, loc, k, home_team_score,away_team_score, R_away, R_home):
    #Is there home field advantage
    H = 0 if loc else 100
    
    diff = (R_away - (R_home + H)) / 400
    diff = np.clip(diff, -20, 20)
    E = 1 / (1 + np.exp(diff * np.log(10)))

    #Whats the goal difference
    gols = home_team_score-away_team_score
    if gols <=1:
        G =1
    elif gols == 2:
        G = 1.5
    else:
        G = 1.75 + (gols-3)/8

    #Caculate Elo Shift
    shift = (k * G)*(S - E)
    #Caculate away and home teams elo and return
    return R_home + shift, R_away - shift

#looping through matches and updating elo
def run_elo_updates(relevant_matches_df, country_elo):
    for row in relevant_matches_df.itertuples(index=False):
        away_elo,home_elo = update_elo(row.home_win, 
                                       row.neutral, 
                                       row.K_factor, 
                                       row.home_score, 
                                       row.away_score, 
                                       country_elo[row.away_team], 
                                       country_elo[row.home_team])
        
        country_elo[row.home_team] = home_elo
        country_elo[row.away_team] = away_elo
        
    return country_elo



def prepare_matches(df, start_date, end_date): #Filter matches to a date range and add result + K-factor columns.
    mask = (df['date'] > start_date) & (df['date'] < end_date)
    out = df.loc[mask].copy()
    out['result'] = out.apply(
        lambda row: get_result(row['home_score'], row['away_score']), axis=1
    )
    out['K_factor'] = out['tournament'].apply(assign_k)
    return out
