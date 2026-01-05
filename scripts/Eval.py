import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# 1. Chargement des données
df = pd.read_csv('../processed_data/csv_games_fusionnes.csv')

# --- ANALYSE UNIVARIÉE ---
print("--- Analyse Univariée ---")

# Distribution de la variable cible (result)
target_counts = df['result'].value_counts(normalize=True) * 100
print(f"\nDistribution de la cible 'result' (%):\n{target_counts}")

# Analyse de la "normalisation" (équilibre des classes/fréquences)
# On regarde si certains champions dominent outrageusement la meta (skewness catégoriel)
def analyze_top_champions(df, slots, title, top_n=10):
    combined = pd.concat([df[col] for col in slots], ignore_index=True)
    counts = combined.value_counts(normalize=True).head(top_n) * 100
    return counts

def champion_distribution(df, cols, top_n=20, as_percent=True):
    """Retourne la distribution des champions sur une liste de colonnes (picks/bans)."""
    combined = pd.concat([df[c] for c in cols], ignore_index=True).dropna()
    if as_percent:
        return (combined.value_counts(normalize=True).head(top_n) * 100)
    return combined.value_counts().head(top_n)

blue_bans = [f'bb{i}' for i in range(1, 6)]
blue_picks = [f'bp{i}' for i in range(1, 6)]
red_bans = [f'rb{i}' for i in range(1, 6)]
red_picks = [f'rp{i}' for i in range(1, 6)]

all_bans = blue_bans + red_bans
all_picks = blue_picks + red_picks

print("\nTop 10 Champions les plus bannis (Blue Side) (%):")
print(analyze_top_champions(df, blue_bans, "Bans Blue", top_n=10))

# --- NOUVEAU : Distribution des personnages ---
print("\n--- Distribution des personnages (Champions) ---")

print("\nTop 20 Champions les plus pick (% - Blue+Red) :")
top_picks = champion_distribution(df, all_picks, top_n=20, as_percent=True)
print(top_picks)

print("\nTop 20 Champions les plus ban (% - Blue+Red) :")
top_bans = champion_distribution(df, all_bans, top_n=20, as_percent=True)
print(top_bans)

print("\nTop 20 Champions les plus pick (% - Blue uniquement) :")
print(champion_distribution(df, blue_picks, top_n=20, as_percent=True))

print("\nTop 20 Champions les plus pick (% - Red uniquement) :")
print(champion_distribution(df, red_picks, top_n=20, as_percent=True))

# --- ANALYSE BIVARIÉE & CORRÉLATION ---
print("\n--- Analyse Bivariée & Corrélation ---")

def cramers_v(x, y):
    """ Calcule le V de Cramér pour mesurer l'association entre deux variables catégorielles """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Étude de corrélation avec la cible 'result'
correlations_with_target = {}
cols_to_test = blue_picks + red_picks  # On se concentre sur les picks pour le winrate
for col in cols_to_test:
    correlations_with_target[col] = cramers_v(df[col], df['result'])

corr_series = pd.Series(correlations_with_target).sort_values(ascending=False)
print("\nAssociation (Cramér's V) entre les Picks et le Résultat (0 à 1) :")
print(corr_series)

# --- VISUALISATIONS ---

# 1. Distribution de la cible
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='result', hue='result', palette='viridis', legend=False)
plt.title('Distribution des Victoires (b=Blue, r=Red)')
plt.tight_layout()
plt.savefig('distribution_cible.png')

# 1bis. Top picks (distribution personnages)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_picks.values, y=top_picks.index, palette='Blues_r')
plt.xlabel('Pourcentage (%)')
plt.ylabel('Champion')
plt.title('Top 20 Champions les plus pick (Blue+Red)')
plt.tight_layout()
plt.savefig('distribution_champions_picks_top20.png')

# 1ter. Top bans (distribution personnages)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_bans.values, y=top_bans.index, palette='Reds_r')
plt.xlabel('Pourcentage (%)')
plt.ylabel('Champion')
plt.title('Top 20 Champions les plus ban (Blue+Red)')
plt.tight_layout()
plt.savefig('distribution_champions_bans_top20.png')

# 2. Heatmap de corrélation (Cramér's V) entre les premiers picks
plt.figure(figsize=(10, 8))
selected_cols = ['bp1', 'rp1', 'bp2', 'rp2', 'bp3', 'rp3', 'bp4', 'rp4', 'bp5', 'rp5', 'result']
corr_matrix = pd.DataFrame(index=selected_cols, columns=selected_cols)

for c1 in selected_cols:
    for c2 in selected_cols:
        corr_matrix.loc[c1, c2] = cramers_v(df[c1], df[c2])

sns.heatmap(corr_matrix.astype(float), annot=True, cmap='coolwarm')
plt.title('Association entre les premiers Picks et le Résultat')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')

print(
    "\nAnalyses terminées. Graphiques sauvegardés sous :\n"
    "- 'distribution_cible.png'\n"
    "- 'distribution_champions_picks_top20.png'\n"
    "- 'distribution_champions_bans_top20.png'\n"
    "- 'correlation_heatmap.png'\n"
)