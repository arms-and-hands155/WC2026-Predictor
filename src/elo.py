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
    
    diff = (R_home + H - R_away) / 400
    diff = np.clip(diff, -20, 20)
    E = 1 / (1 + np.exp(-diff * np.log(10)))

    #Whats the goal difference
    gols = abs(home_team_score - away_team_score)
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

def prepare_matches(df, start_date, end_date): #Filter matches to a date range and add result + K-factor columns.
    mask = (df['date'] > start_date) & (df['date'] < end_date)
    out = df.loc[mask].copy()
    out['result'] = out.apply(
        lambda row: get_result(row['home_score'], row['away_score']), axis=1
    )
    out['K_factor'] = out['tournament'].apply(assign_k)
    out['tournament_weight'] = out['tournament'].apply(tournament_weight)
    return out

#looping through matches and updating elo
def run_elo_updates(relevant_matches_df, country_elo):
    for row in relevant_matches_df.itertuples(index=False):
        away_elo,home_elo = update_elo(row.result, 
                                       row.neutral, 
                                       row.K_factor, 
                                       row.home_score, 
                                       row.away_score, 
                                       country_elo[row.away_team], 
                                       country_elo[row.home_team])
        
        country_elo[row.home_team] = home_elo
        country_elo[row.away_team] = away_elo
        
    return country_elo

def tournament_weight(t):
    if t in {'FIFA World Cup'}:          return 5
    if t in {'UEFA Euro', 'Copa América', 'African Cup of Nations', 'AFC Asian Cup',
    'Gold Cup', 'CONCACAF Championship', 'Oceania Nations Cup',
    'Confederations Cup',}:        return 4
    if t in {'FIFA World Cup qualification', 'UEFA Euro qualification',
    'African Cup of Nations qualification', 'AFC Asian Cup qualification',
    'Gold Cup qualification', 'CONCACAF Championship qualification',
    'Copa América qualification', 'Oceania Nations Cup qualification',
    'UEFA Nations League', 'CONCACAF Nations League',
    'CONCACAF Nations League qualification',}: return 3
    if t in {
    'CECAFA Cup', 'COSAFA Cup', 'COSAFA Cup qualification', 'WAFF Championship',
    'Amílcar Cabral Cup', 'All-African Games', 'UDEAC Cup', 'UNIFFAC Cup',
    'West African Cup', 'Nile Basin Tournament', 'African Friendship Games',
    # Asia / Oceania
    'Gulf Cup', 'Arab Cup', 'Arab Cup qualification', 'SAFF Cup',
    'AFF Championship', 'AFF Championship qualification', 'EAFF Championship',
    'EAFF Championship qualification', 'ASEAN Championship',
    'ASEAN Championship qualification', 'AFC Challenge Cup',
    'AFC Challenge Cup qualification', 'Asian Games', 'CAFA Nations Cup',
    'Southeast Asian Games', 'South Asian Games', 'Dynasty Cup',
    'Pacific Games', 'South Pacific Games', 'Melanesia Cup',
    'Indian Ocean Island Games', 'Afro-Asian Games',
    # Europe
    'British Home Championship', 'Nordic Championship', 'Baltic Cup',
    'Balkan Cup', 'Central European International Cup',
    # Americas
    'CFU Caribbean Cup', 'CFU Caribbean Cup qualification', 'UNCAF Cup',
    'Central American and Caribbean Games', 'Pan American Championship',
    'CCCF Championship', 'Bolivarian Games', 'NAFC Championship',
    # Multi-sport
    'Olympic Games',}:           return 2
    return 1   

