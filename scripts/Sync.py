import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np

# --- CONFIGURATION ---
FILE_PATH = '../processed_data/csv_games_fusionnes.csv'
MIN_GAMES_PERCENTILE = 0.95 # On garde le top 5% des paires les plus jouées (fiabilité statistique)

# 1. Chargement
print("Chargement des données...")
df = pd.read_csv(FILE_PATH)

# 2. Structure pour compter les paires
pair_stats = {} 
blue_picks_cols = [f'bp{i}' for i in range(1, 6)]
red_picks_cols = [f'rp{i}' for i in range(1, 6)]

# 3. Calcul des synergies (Optimisé)
print("Calcul des synergies en cours...")
blue_data = df[blue_picks_cols].values
red_data = df[red_picks_cols].values
results = df['result'].values 

for i in range(len(df)):
    # Blue Side
    b_team = [c for c in blue_data[i] if pd.notna(c)]
    b_win = 1 if results[i] == 'b' else 0
    for pair in combinations(set(b_team), 2):
        pair_key = tuple(sorted(pair))
        if pair_key not in pair_stats: pair_stats[pair_key] = {'wins': 0, 'games': 0}
        pair_stats[pair_key]['games'] += 1
        pair_stats[pair_key]['wins'] += b_win

    # Red Side
    r_team = [c for c in red_data[i] if pd.notna(c)]
    r_win = 1 if results[i] == 'r' else 0
    for pair in combinations(set(r_team), 2):
        pair_key = tuple(sorted(pair))
        if pair_key not in pair_stats: pair_stats[pair_key] = {'wins': 0, 'games': 0}
        pair_stats[pair_key]['games'] += 1
        pair_stats[pair_key]['wins'] += r_win

# 4. Création du DataFrame
synergy_list = []
for pair, stats in pair_stats.items():
    synergy_list.append({
        'Duo': f"{pair[0]} + {pair[1]}",
        'Champion A': pair[0],
        'Champion B': pair[1],
        'Games': stats['games'],
        'Wins': stats['wins'],
        'Winrate': stats['wins'] / stats['games']
    })

df_synergy = pd.DataFrame(synergy_list)

# 5. Filtrage et Tri
threshold = df_synergy['Games'].quantile(MIN_GAMES_PERCENTILE)
if threshold < 50: threshold = 50 # Minimum de sécurité
print(f"Filtre activé : Un duo doit avoir joué au moins {int(threshold)} parties.")

# On ne garde que les paires significatives et on trie par WINRATE
df_best = df_synergy[df_synergy['Games'] >= threshold].copy()
df_best = df_best.sort_values(by='Winrate', ascending=False)

# 6. Sauvegarde et Affichage
# Sauvegarder les 100 meilleures synergies pour analyse externe
df_best.head(100).to_csv('top_100_synergies.csv', index=False)
print("Fichier 'top_100_synergies.csv' généré.")

# Affichage console des Top 10
print("\n--- TOP 10 DUOS GAGNANTS ---")
print(df_best[['Duo', 'Games', 'Winrate']].head(10).to_string(index=False))

# 7. Visualisation (Bar Chart)
top_20 = df_best.head(20)

plt.figure(figsize=(10, 8))
# Barplot horizontal
sns.barplot(data=top_20, x='Winrate', y='Duo', palette='viridis')

# Esthétique
plt.title(f'Top 20 Synergies (Winrate) - Min {int(threshold)} Games')
plt.xlabel('Taux de Victoire')
plt.ylabel('Duo de Champions')
plt.xlim(0.5, top_20['Winrate'].max() + 0.02) # On zoome sur la partie > 50%
plt.axvline(0.5, color='red', linestyle='--', alpha=0.5) # Ligne des 50%
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('top_20_synergies.png')
print("\nGraphique 'top_20_synergies.png' généré.")
plt.show()