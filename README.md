# Projet Draft LoL — Prédiction des bans/picks et des rôles

Ce projet construit un système de prédiction pour la phase de draft de League of Legends (bans/picks), en s'appuyant sur des historiques de matchs (Oracle's Elixir) et un modèle ML supervisé. L'algorithme propose les bans/picks suivants à chaque étape et tient compte des rôles à compléter par équipe.

## Table des matières
- [Objectif](#objectif)
- [Données](#données)
- [Algorithme](#algorithme-vue-densemble)
- [Installation](#installation)
- [Préparation des données](#préparation-des-données)
- [Utilisation](#utilisation)
- [Évaluation](#évaluation)
- [Structure des fichiers](#structure-des-fichiers)
- [Notes et bonnes pratiques](#notes-et-bonnes-pratiques)
- [Idées d'amélioration](#idées-damélioration)

## Objectif
- Prédire les bans et picks successifs pour les équipes Blue/Red.
- Éviter les doublons en tenant compte des champions déjà utilisés (bans + picks des deux équipes).
- Favoriser des picks qui complètent des rôles manquants (`top`, `jng`, `mid`, `bot`, `sup`).
- Affecter automatiquement un rôle à chaque champion pické, y compris les picks flexibles (champions pouvant occuper plusieurs rôles).

## Données

### Source principale
- **CSV Oracle's Elixir** (2014 → 2025)
- Fichiers présents dans le dossier `CSV/` et à la racine
- Colonnes pertinentes : `gameid`, champions des bans/picks, positions, résultats

### Fichiers générés
- `csv_games_fusionnes.csv` : Dataset d'entraînement fusionné (Blue/Red par `gameid`)
  - Colonnes : `bb1..bb5` (Blue bans), `rb1..rb5` (Red bans), `bp1..bp5` (Blue picks), `rp1..rp5` (Red picks)
  - Cible : `result` ∈ {`b`, `r`} (Blue win / Red win)
- `champions_by_position.csv` : Mapping des rôles par champion
  - Généré depuis les données Oracle's Elixir
  - Utilisé pour l'affectation des rôles et la pénalisation

### Pipeline de préparation
1. **[Filtre.py](Filtre.py)** → Réduit les colonnes des CSV et génère `draft_dataset_bans_picks.csv`
2. **[Fusion.py](Fusion.py)** → Fusionne Blue/Red par `gameid`, crée `csv_games_fusionnes.csv`
3. **[PrepaRoles.py](PrepaRoles.py)** → Agrège les positions et génère `champions_by_position.csv`
4. **[Division.py](Division.py)** → (Optionnel) Produit des sous-ensembles pour analyses

## Algorithme (vue d'ensemble)

### Modèle principal
Implémenté dans **[Roles.py](Roles.py)** :
- **Modèle** : `LinearSVC` (scikit-learn)
- **Encodage** : `OneHotEncoder` pour les features catégorielles
- **Entraînement séquentiel** : Un classifieur par étape de draft (20 classifieurs au total)
- **Filtrage** : Classes (champions) observées ≥ 2 fois uniquement

### Prédiction intelligente
1. **Scores normalisés** : `softmax` appliqué sur `decision_function` de SVM
2. **Anti-doublon global** : Masquage des champions déjà bannis/pickés
3. **Pénalisation par rôle** :
   - Favorise les champions comblant des rôles manquants
   - Pénalise modérément les champions inutiles pour les rôles restants
   - Pénalise fortement les champions sans rôle connu
4. **Mécanisme de secours** : Si tous les scores sont nuls, choix du premier champion disponible

### Affectation des rôles
- **`assign_champion_role()`** :
  - 1 rôle possible → Assignation directe
  - Plusieurs rôles possibles → Pick "flex" (`PROV-...`)
  - Tous les rôles pris → Double rôle (premier disponible)
- **`resolve_flexible_assignments()`** :
  - Résout les picks provisoires
  - Optimise l'affectation pour éviter les conflits

## Installation

### Prérequis
- **Python 3.10+** (recommandé : 3.11 ou 3.12)
- **Git** (optionnel, pour le versioning)
- **Git LFS** (optionnel, pour les gros fichiers CSV)

### Installation des dépendances
```bash
pip install -r requirements.txt
```

## Préparation des données

### Étape 1 : Filtrage et fusion
```bash
python Filtre.py
python Fusion.py
```
Génère `csv_games_fusionnes.csv` avec les 20 colonnes de draft + cible `result`.

### Étape 2 : Mapping des rôles
```bash
python PrepaRoles.py
```
Crée `champions_by_position.csv` avec les statistiques de rôles par champion.

### Étape 3 : Division (optionnel)
```bash
python Division.py
```
Produit des sous-ensembles progressifs pour l'analyse d'ablation.

## Utilisation

### Mode Self-Play (par défaut)
Le modèle prédit l'intégralité du draft automatiquement :
```bash
python Roles.py
```

### Mode Interactif
Pour interagir et saisir vos propres bans/picks :
1. Ouvrir `Roles.py`
2. Modifier la ligne : `SELF_PLAY = False`
3. Relancer :
```bash
python Roles.py
```

### Scripts de test ML
Plusieurs variantes de modèles sont disponibles pour comparaison :
- `TestML.py` : RandomForest
- `TestML2.py` : LightGBM
- `TestML3.py` : XGBoost avec hyperparamètres basiques
- `TestML4.py` : XGBoost optimisé
- `TestML5.py` : Ensemble de modèles
- `TestML6.py` : Neural Network avec TensorFlow/Keras

```bash
python TestML.py  # Exemple avec RandomForest
```

## Évaluation

### Métrique principale
**Exactitude Top-1** : Proportion de prédictions correctes par étape de draft
- Calculée pour chaque cible (`bb1`, `rb1`, ..., `bp5`, `rp5`)
- Moyenne globale sur les 20 étapes

### Protocole recommandé

#### Split temporel
```python
# Entraînement : 2014-2023
# Test : 2024-2025
```
Respecte la dérive de méta du jeu.

#### Entraînement par étape
- Chaque cible utilise uniquement les features disponibles à son moment
- Exemple : `rb2` s'entraîne avec `bb1`, `rb1`, `bb2`

#### Gestion des classes rares
- Filtrage ≥ 2 occurrences pour éviter l'overfitting
- Amélioration possible : seuil adaptatif selon la saison

### Analyses complémentaires

#### 1. Impact du masque anti-doublon
Comparer avec/sans contrainte d'unicité des champions.

#### 2. Efficacité de la pénalisation par rôle
Mesurer l'amélioration de la complétion des compositions.

#### 3. Confusions et Top-K
- Examiner les top-5 suggestions
- Calculer l'exactitude Top-3, Top-5
- Analyser les erreurs fréquentes (champions similaires, rôles flexibles)

### Exemple de procédure d'évaluation
```python
# 1. Charger les données
df_train = pd.read_csv('csv_games_fusionnes.csv')
df_train = df_train[df_train['year'] <= 2023]
df_test = pd.read_csv('csv_games_fusionnes.csv')
df_test = df_test[df_test['year'] >= 2024]

# 2. Boucler sur les 20 cibles
for target_col in ['bb1', 'rb1', ..., 'bp5', 'rp5']:
    # Entraîner le modèle
    # Prédire sur test
    # Calculer exactitude
    
# 3. Reporter les résultats
print(f"Moyenne globale : {mean_accuracy:.2%}")
```

## Structure des fichiers

```
Projet/
├── CSV/
│   ├── 2020_LoL_esports_match_data_from_OraclesElixir.csv
│   ├── 2021_LoL_esports_match_data_from_OraclesElixir.csv
│   └── ... (2014-2025)
├── 2014_LoL_esports_match_data_from_OraclesElixir.csv
├── 2015_LoL_esports_match_data_from_OraclesElixir.csv
├── ... (années à la racine)
├── csv_filtre_concatenes.csv (intermédiaire)
├── draft_dataset_bans_picks.csv (intermédiaire)
├── csv_games_fusionnes.csv (dataset principal)
├── champions_by_position.csv (mapping rôles)
├── Filtre.py (préparation)
├── Fusion.py (préparation)
├── PrepaRoles.py (préparation)
├── Division.py (optionnel)
├── Roles.py (exécution principale)
├── TestML.py (RandomForest)
├── TestML2.py (LightGBM)
├── TestML3.py (XGBoost basique)
├── TestML4.py (XGBoost optimisé)
├── TestML5.py (Ensemble)
├── TestML6.py (Neural Network)
├── requirements.txt
└── README.md
```

## Notes et bonnes pratiques

### Gestion des gros fichiers
- Les CSV peuvent dépasser 100 Mo
- Utiliser **Git LFS** pour les fichiers > 50 Mo :
```bash
git lfs install
git lfs track "*.csv"
git add .gitattributes
```

### Limitations du modèle
- `LinearSVC` ne produit pas de probabilités calibrées
- Le `softmax` sur `decision_function` fournit des scores comparatifs uniquement
- La pénalisation par rôle est heuristique, non optimale mathématiquement

### Performance et optimisation
- Les CSV volumineux peuvent ralentir le chargement
- Considérer `pandas.read_csv(chunksize=...)` pour très gros fichiers
- Utiliser `joblib` ou `pickle` pour sauvegarder les modèles entraînés

### Dépendance aux données
- Les résultats varient selon la saison, le patch, la méta
- Régénérer `champions_by_position.csv` après chaque mise à jour des données
- Considérer un système de versioning des datasets

## Idées d'amélioration

### Court terme
- [ ] Implémenter la calibration des probabilités (Platt scaling, isotonic regression)
- [ ] Ajouter la sauvegarde/chargement des modèles entraînés
- [ ] Créer un script d'évaluation automatique avec métriques complètes
- [ ] Générer des graphiques d'analyse (confusion matrices, courbes Top-K)

### Moyen terme
- [ ] Features contextuelles :
  - Synergies entre champions pickés
  - Contre-picks basés sur l'historique
  - Informations de patch/méta
  - Statistiques par ligue/équipe/joueur
- [ ] Modèles alternatifs :
  - `LogisticRegression` avec régularisation
  - `CatBoost` pour données catégorielles
  - Réseaux de neurones récurrents (LSTM) pour séquence de draft
- [ ] Split de validation :
  - Validation croisée temporelle
  - Hyperparameter tuning avec Optuna/GridSearch

### Long terme
- [ ] Interface web interactive pour simuler des drafts
- [ ] API REST pour intégration dans d'autres outils
- [ ] Système de mise à jour automatique des données Oracle's Elixir
- [ ] Analyse de méta en temps réel basée sur les patchs récents
- [ ] Système de recommandation multi-objectifs (win rate + style de jeu)

---

## Support et contribution

Pour toute question ou suggestion d'amélioration, n'hésitez pas à ouvrir une issue ou à proposer une pull request.

**Auteur** : Louis Poutrain  
**Dernière mise à jour** : Janvier 2026