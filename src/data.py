import os 
from dotenv import load_dotenv
import kagglehub
import pandas as pd
from collections import defaultdict

load_dotenv()

#Function that finds the files that contains the csvs
def find_file(root_folder, filename):
    for root, dirs, files in os.walk(root_folder):
        if filename in files:
            return os.path.join(root, filename)
    return None

def load_matches():
    #Download and return international matches df
    matches = kagglehub.dataset_download("martj42/international-football-results-from-1872-to-2017")
    matches_path = find_file(matches, "results.csv")
    if not matches_path:
        raise FileNotFoundError("File not found")
    df = pd.read_csv(matches_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # This replaces the hidden \xa0 characters with normal spaces
    df['home_team'] = df['home_team'].str.replace('\xa0', ' ', regex=True).str.strip()
    df['away_team'] = df['away_team'].str.replace('\xa0', ' ', regex=True).str.strip()
    
    return df

def load_elo_baseline(date="12/13/2025"):
    elo = kagglehub.dataset_download("saifalnimri/international-football-elo-ratings")
    elo_path = find_file(elo, "eloratings.csv")
    if not elo_path:
        raise FileNotFoundError("File not found")
    elo_df = pd.read_csv(elo_path)
    elo_2025 = elo_df[elo_df['date'] == date]
    elo_2025['team'] = elo_2025['team'].str.replace('\xa0', ' ', regex=True).str.strip()
    country_elo = defaultdict(lambda:1500)
    for row in elo_2025.itertuples(index=False):
        country_elo[row.team] = row.rating #Gives us a dictionary with every teams elo in 2025
    
    return country_elo  

