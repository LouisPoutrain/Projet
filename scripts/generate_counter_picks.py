import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

# Load all raw CSVs from 2020-2025
raw_data_dir = "../raw_data"
years = [2020, 2021, 2022, 2023, 2024, 2025]

all_dfs = []
for year in years:
    file_path = os.path.join(raw_data_dir, f"{year}_LoL_esports_match_data_from_OraclesElixir.csv")
    if os.path.exists(file_path):
        print(f"Loading {year}...")
        df = pd.read_csv(file_path, usecols=['gameid', 'side', 'position', 'champion', 'result'])
        df = df.dropna(subset=['position', 'champion'])  # Remove team/aggregated rows
        df = df[df['position'].isin(['top', 'jng', 'mid', 'bot', 'sup'])]  # Keep only player rows
        all_dfs.append(df)
        print(f"  -> {len(df)} player rows")

df_combined = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal player records: {len(df_combined)}")

# Build matchups: for each game, get blue winners vs red picks, and red winners vs blue picks
matchup_records = []

for game_id, game_data in tqdm(df_combined.groupby('gameid'), desc="Processing matchups"):
    blue_data = game_data[game_data['side'] == 'Blue']
    red_data = game_data[game_data['side'] == 'Red']
    
    if blue_data.empty or red_data.empty:
        continue
    
    # Determine which team won
    blue_result = blue_data.iloc[0]['result']
    red_result = red_data.iloc[0]['result']
    
    if blue_result == 1:  # Blue won
        # Record each blue champion vs each red champion matchup
        for _, blue_row in blue_data.iterrows():
            blue_champ = blue_row['champion']
            blue_pos = blue_row['position']
            for _, red_row in red_data.iterrows():
                red_champ = red_row['champion']
                red_pos = red_row['position']
                matchup_records.append({
                    'champion': blue_champ,
                    'position': blue_pos,
                    'opponent': red_champ,
                    'opp_position': red_pos,
                    'won': 1
                })
    
    elif red_result == 1:  # Red won
        # Record each red champion vs each blue champion matchup
        for _, red_row in red_data.iterrows():
            red_champ = red_row['champion']
            red_pos = red_row['position']
            for _, blue_row in blue_data.iterrows():
                blue_champ = blue_row['champion']
                blue_pos = blue_row['position']
                matchup_records.append({
                    'champion': red_champ,
                    'position': red_pos,
                    'opponent': blue_champ,
                    'opp_position': blue_pos,
                    'won': 1
                })

df_matchups = pd.DataFrame(matchup_records)
print(f"\nCreated {len(df_matchups)} winning matchup records")

# Calculate win rates: for all matchups (no filters)
# Combine wins and losses perspective

# Create all matchups (both wins and losses from champion perspective)
all_matchups = []

print("Expanding matchups to wins/losses perspective...")
for _, row in tqdm(df_matchups.iterrows(), total=len(df_matchups), desc="Building all_matchups"):
    all_matchups.append({
        'champion': row['champion'],
        'position': row['position'],
        'opponent': row['opponent'],
        'opponent_pos': row['opp_position'],
        'result': 'win'
    })
    all_matchups.append({
        'champion': row['opponent'],
        'position': row['opp_position'],
        'opponent': row['champion'],
        'opponent_pos': row['position'],
        'result': 'loss'
    })

df_all = pd.DataFrame(all_matchups)
print(f"Total matchups (all positions): {len(df_all)}")

# Calculate win rates for each (champion, position, opponent, opponent_pos) combination
win_rate_data = []
print("Calculating all win rates...")
for (champ, pos, opp, opp_pos), group in tqdm(df_all.groupby(['champion', 'position', 'opponent', 'opponent_pos']), desc="Computing win rates"):
    total_games = len(group)
    wins = (group['result'] == 'win').sum()
    win_rate = wins / total_games
    win_rate_data.append({
        'champion': champ,
        'position': pos,
        'opponent': opp,
        'opponent_pos': opp_pos,
        'wins': int(wins),
        'losses': int(total_games - wins),
        'total_games': total_games,
        'win_rate': win_rate
    })

df_win_rates = pd.DataFrame(win_rate_data)
print(f"Calculated {len(df_win_rates)} total matchup win rates")

# Save all matchup win rates to CSV (no filtering, no top-N selection)
output_path = "../processed_data/all_matchup_winrates.csv"
df_win_rates.to_csv(output_path, index=False)
print(f"\nAll matchup win rates saved to: {output_path}")
print(f"Total matchup records: {len(df_win_rates)}")
print("\nFirst 10 entries:")
print(df_win_rates.head(10).to_string())
