import kagglehub
import os 
import pandas as pd
from collections import defaultdict
import numpy as np
from dotenv import load_dotenv


load_dotenv()
secret = os.getenv("KAGGLE_API_TOKEN")

#Setting up Kaggle and the datasets
matches = kagglehub.dataset_download("martj42/international-football-results-from-1872-to-2017")
elo = kagglehub.dataset_download("saifalnimri/international-football-elo-ratings")

#Function that finds the files that contains the csvs
def find_file(root_folder, filename):
    for root, dirs, files in os.walk(root_folder):
        if filename in files:
            return os.path.join(root, filename)
    return None
#Getting results for matches
matches_path = find_file(matches, "results.csv")
if matches_path:
    df = pd.read_csv(matches_path)
else:
    print("Could not find the file anywhere in the downloaded folder.")
    
#Getting Elo data from 2025
elo_path = find_file(elo, "eloratings.csv")
if elo_path:
    elo_df = pd.read_csv(elo_path)
else:
    print("Could not find the file anywhere in the downloaded folder.")

elo_2025 = elo_df[elo_df['date'] == '12/13/2025']
country_elo = defaultdict(lambda:1500)
for row in elo_2025.itertuples(index=False):
    country_elo[row.team] = row.rating #Gives us a dictionary with every teams elo in 2025

df['date'] = pd.to_datetime(df['date'])
mask = ((df['date']>"2025-12-13") & (df['date'] < '2026-06-11'))
relevant_matches_df = df.loc[mask].copy()

def home_win(home_score, away_score): #Dataset desont include home_win column so must add it
    if home_score > away_score:
        return 1
    elif away_score > home_score:
        return 0
    else:
        return 0.5
relevant_matches_df['home_win'] = relevant_matches_df.apply(lambda row: home_win(row['home_score'], row['away_score']), 
    axis=1)

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

relevant_matches_df['K_factor'] = relevant_matches_df['tournament'].apply(assign_k)


# #Elo caculation
def elo(S, loc, k, home_team_score,away_team_score, R_away, R_home):
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
    return R_away - shift, R_home + shift


for row in relevant_matches_df.itertuples(index=False):
    away_elo,home_elo = elo(row.home_win, row.neutral, row.K_factor, 
                            row.home_score, row.away_score, country_elo[row.away_team], 
                            country_elo[row.home_team])
    
    country_elo[row.away_team] = away_elo
    country_elo[row.home_team] = home_elo
