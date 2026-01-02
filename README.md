# Projet Draft LoL — Prédiction des bans/picks et des rôles

Ce projet construit un système de prédiction pour la phase de draft de League of Legends (bans/picks), en s'appuyant sur des historiques de matchs (Oracle's Elixir) et un modèle ML supervisé. L'algorithme propose les bans/picks suivants à chaque étape et tient compte des rôles à compléter par équipe.

## Objectif
- Prédire les bans et picks successifs pour les équipes Blue/Red.
- Éviter les doublons en tenant compte des champions déjà utilisés (bans + picks des deux équipes).
- Favoriser des picks qui complètent des rôles manquants (`top`, `jng`, `mid`, `bot`, `sup`).
- Affecter automatiquement un rôle à chaque champion pické, y compris les picks flexibles (champions pouvant occuper plusieurs rôles).

## Données
- Source principale: CSV Oracle's Elixir (2014 → 2025), présents dans le dossier `CSV/` et à la racine.
- Fichier fusionné d'entraînement: `csv_games_fusionnes.csv` (issue de la fusion des côtés Blue/Red, colonnes `bb1..bb5`, `rb1..rb5`, `bp1..bp5`, `rp1..rp5`, et la cible `result` ∈ {`b`,`r`}).
- Mapping des rôles par champion: `champions_by_position.csv` (généré depuis les données Oracle's Elixir).

Fichiers clés du pipeline:
- [Filtre.py](Filtre.py) → réduit les colonnes des CSV filtrés concaténés vers `draft_dataset_bans_picks.csv`.
- [Fusion.py](Fusion.py) → fusionne les lignes Blue/Red par `gameid`, crée `csv_games_fusionnes.csv` et la cible `result` (`b` si Blue gagne, sinon `r`).
- [PrepaRoles.py](PrepaRoles.py) → agrège les apparitions par position et génère `champions_by_position.csv`.
- [Division.py](Division.py) → optionnel, produit des sous-ensembles progressifs pour analyses/ablation.

## Algorithme (vue d'ensemble)
Implémenté principalement dans [Roles.py](Roles.py):
- Modèle: `LinearSVC` (scikit-learn) avec features encodées via `OneHotEncoder`.
- Entraînement par étape: un classifieur est appris pour chaque colonne cible (`bb1`, `rb1`, ..., `bp5`, `rp5`) avec comme features les bans/picks déjà connus à ce moment.
- Filtrage des classes rares: seules les classes (champions) observées ≥ 2 fois sont utilisées pour l'entraînement.
- Score → pseudo-proba: on applique `softmax` sur `decision_function` pour obtenir un score par champion.
- Anti-doublon global: on met à zéro les scores des champions déjà utilisés (bans + picks totaux jusque-là).
- Pénalisation consciente des rôles:
  - Si des rôles de l'équipe sont encore vides, on pénalise les champions qui ne peuvent combler aucun rôle manquant (modérément s'ils ont des rôles connus mais non utiles; plus fortement s'ils n'ont aucun rôle connu).
- Secours: si tous les scores sont nuls (masqués/penalisés), on choisit le premier champion restant disponible.

### Affectation des rôles
- `assign_champion_role()` affecte un rôle:
  - Si un champion ne peut que 1 rôle non encore pris → assignation directe.
  - S'il peut plusieurs rôles → pick «flex» (marqué `PROV-...`) à résoudre plus tard.
  - Si tous ses rôles possibles sont déjà pris → on assigne le premier possible (double rôle).
- `resolve_flexible_assignments()` résout ensuite les `PROV-...` en tentant d'assigner de manière unique les rôles restants; sinon on choisit le premier rôle disponible.

## Utilisation
Deux modes sont possibles dans [Roles.py](Roles.py):
- `SELF_PLAY = True` (par défaut): le modèle prédit tout le draft (modèle vs modèle).
- `SELF_PLAY = False`: interaction utilisateur, vous saisissez les bans/picks au fur et à mesure.

### Prérequis
- Python 3.10+
- Paquets: `pandas`, `numpy`, `scikit-learn`
- Données: `csv_games_fusionnes.csv` et `champions_by_position.csv` présents à la racine du projet.

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Préparation des données
1) Filtrer puis fusionner:
```bash
python Filtre.py
python Fusion.py
```
2) Préparer le mapping rôles:
```bash
python PrepaRoles.py
```
3) (Optionnel) Générer des sous-ensembles:
```bash
python Division.py
```

### Exécution
Lancer la prédiction complète (self-play):
```bash
python Roles.py
```
Pour passer en mode interactif, éditez `SELF_PLAY` à `False` dans [Roles.py](Roles.py) et relancez.

## Évaluation
- **Métrique principale:** exactitude Top-1 par étape (pour chaque cible `bb1`, `rb1`, …, `bp5`, `rp5`), c’est-à-dire la proportion de fois où le champion prédit correspond au champion observé dans `csv_games_fusionnes.csv`.
- **Protocole conseillé:**
  - **Split temporel:** entraîner sur les saisons antérieures (ex. 2014–2023) et tester sur 2024–2025 pour respecter les dérives de méta.
  - **Par-étape:** pour chaque cible, utiliser exactement les features disponibles au moment de la draft (ex. `rb2` s’entraîne avec `bb1`, `rb1`, `bb2`).
  - **Classes rares:** conserver le filtrage ≥ 2 occurrences pour limiter l’overfitting sur des picks anecdotiques.
  - **Comparaison de variantes:** mesurer la même métrique pour `LinearSVC` vs alternatives ([TestML.py](TestML.py), [TestML2.py](TestML2.py), [TestMl3.py](TestMl3.py), [TestML4.py](TestML4.py), [TestML5.py](TestML5.py), [TestML6.py](TestML6.py)).
- **Analyse complémentaire:**
  - **Impact du masque anti-doublon:** vérifier l’amélioration par rapport à une prédiction naïve sans contrainte.
  - **Pénalisation des rôles:** comparer l’exactitude des picks avec et sans pénalisation rôle-aware (utile en fin de draft).
  - **Confusions:** top-5 suggestions et taux de réussite Top-k pour voir si le bon champion est souvent dans le haut du classement.

Exemple de démarche (pseudo-procédure):
- **Préparer les données:** [Fusion.py](Fusion.py) pour `csv_games_fusionnes.csv`, [PrepaRoles.py](PrepaRoles.py) pour `champions_by_position.csv`.
- **Boucler sur les 20 cibles:** pour chaque cible, entraîner `LinearSVC`, prédire sur le split de test, calculer l’exactitude.
- **Reporter les résultats:** tableau par cible et moyenne globale, puis répéter avec les variantes `RandomForest`, `LightGBM`, `XGBoost`.

## Structure des fichiers
- Données brutes: `CSV/20XX_LoL_esports_match_data_from_OraclesElixir.csv` + années 2014–2019 à la racine.
- Intermédiaires: `csv_filtre_concatenes.csv`, `draft_dataset_bans_picks.csv`, `csv_games_fusionnes.csv`.
- Rôles: `champions_by_position.csv`.
- Scripts ML exploratoires: [TestML.py](TestML.py), [TestML2.py](TestML2.py), [TestMl3.py](TestMl3.py), [TestML4.py](TestML4.py), [TestML5.py](TestML5.py), [TestML6.py](TestML6.py) — variantes `RandomForest`, `LightGBM`, `XGBoost`, et `LinearSVC`.
- Exécution principale: [Roles.py](Roles.py).

## Notes et bonnes pratiques
- Les CSV peuvent être volumineux. Si vous poussez sur GitHub, utilisez Git LFS pour les fichiers > 50 Mo.
- Le modèle `LinearSVC` ne produit pas de probabilités calibrées; l'usage de `softmax` sur `decision_function` fournit des scores comparatifs exploités par l'algorithme.
- La pénalisation par rôle est heuristique: elle guide les picks vers la complétion des rôles manquants, mais ne garantit pas une optimalité globale.
- Les résultats dépendent de la qualité/quantité des données (saisons, métas, patchs). Pensez à régénérer `champions_by_position.csv` si vous mettez à jour les données.

## Idées d'amélioration
- Essayer des modèles probabilistes (e.g., `LogisticRegression`, `CatBoost`), calibration (`Platt scaling`).
- Ajouter des features contextuelles (synergies, contre-picks, patch, ligue, joueur, équipe).
- Évaluer systématiquement avec un split temporel, validation croisée, ou benchmarking entre variantes `TestML*`.

---
Si vous souhaitez, je peux ajouter un `requirements.txt`, des scripts d'exécution rapide, ou une évaluation automatique. N'hésitez pas à me le demander !