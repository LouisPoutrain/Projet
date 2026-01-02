# 🎯 Guide d'Amélioration des Modèles de Draft LoL

Ce dossier contient 3 scripts avancés pour entraîner et améliorer vos modèles de prédiction de draft.

## 📦 Installation des Dépendances

Avant de lancer les scripts, installez les dépendances optionnelles :

```bash
pip install lightgbm xgboost optuna
```

Ou individuellement :
```bash
pip install lightgbm  # Gradient Boosting haute performance
pip install xgboost   # Extreme Gradient Boosting
pip install optuna    # Optimisation Bayésienne des hyperparamètres
```

## 🚀 Scripts Disponibles

### 1. `train_and_improve.py` - Comparaison Complète des Modèles
**Description**: Compare LinearSVC, RandomForest, LightGBM et XGBoost avec validation croisée.

**Exécution**:
```bash
python scripts/train_and_improve.py
```

**Fonctionnalités**:
- ✅ Évalue 5 modèles différents en parallèle
- ✅ Validation croisée 5-fold pour robustesse
- ✅ Affiche métriques détaillées (Accuracy, F1, Precision, Recall)
- ✅ Sauvegarde les meilleurs modèles dans `models/improved_models/`
- ✅ Teste sur 5 cibles principales (rb1, bb2, rb2, bp1, rp1)

**Output Exemple**:
```
======================================================================
🎯 TARGET: rb1
======================================================================
  📊 Samples: 5000 | Train: 4000 | Test: 1000
  🏷️  Classes: 10
  
📈 Test Set Performance (80/20 split):
  LinearSVC (baseline)     | Acc: 0.456 | F1: 0.412 | Prec: 0.420 | Rec: 0.410
  LinearSVC (C=10)         | Acc: 0.478 | F1: 0.438 | Prec: 0.445 | Rec: 0.435
  RandomForest             | Acc: 0.512 | F1: 0.465 | Prec: 0.475 | Rec: 0.460
  LightGBM                 | Acc: 0.528 | F1: 0.482 | Prec: 0.490 | Rec: 0.480
  XGBoost                  | Acc: 0.535 | F1: 0.492 | Prec: 0.501 | Rec: 0.490

🔄 Cross-Validation (5-fold) pour: XGBoost
  Accuracy: 0.530 ± 0.015
  F1-Macro: 0.487 ± 0.018
```

---

### 2. `hyperparameter_optimizer.py` - Optimisation des Hyperparamètres
**Description**: Optimise les hyperparamètres avec Optuna (Bayesian Optimization) ou GridSearchCV.

**Exécution**:
```bash
python scripts/hyperparameter_optimizer.py
```

**Fonctionnalités**:
- ✅ Optimisation Bayésienne (Optuna) - plus rapide et efficace
- ✅ GridSearchCV en fallback si Optuna n'est pas installé
- ✅ Teste 50 combinaisons d'hyperparamètres par modèle
- ✅ Cherche les meilleurs paramètres pour LinearSVC et RandomForest
- ✅ Utilise validation croisée 5-fold

**Hyperparamètres Optimisés**:
- **LinearSVC**: C (regularization strength) [0.01 - 100]
- **RandomForest**: 
  - n_estimators [50 - 300]
  - max_depth [5 - 50]
  - min_samples_split [2 - 20]
  - min_samples_leaf [1 - 10]

**Output Exemple**:
```
🔧 Optimisation pour: rb1
  📊 Samples: 5000, Features: 156
  
  🔍 LinearSVC optimization...
    ✅ Best C: 8.4523
  
  🔍 RandomForest optimization...
    ✅ Best params: {'n_estimators': 250, 'max_depth': 35, 'min_samples_split': 3, 'min_samples_leaf': 2}
```

---

### 3. `ensemble_optimizer.py` - Ensemble Learning (Voting & Stacking)
**Description**: Combine plusieurs modèles avec Voting Classifier et Stacking Classifier.

**Exécution**:
```bash
python scripts/ensemble_optimizer.py
```

**Fonctionnalités**:
- ✅ Voting Classifier (soft voting) - combine prédictions par moyenne
- ✅ Stacking Classifier - utilise méta-learner (LogisticRegression)
- ✅ Combine LinearSVC, RandomForest, AdaBoost, LightGBM, XGBoost
- ✅ Souvent meilleur que les modèles individuels
- ✅ Sauvegarde dans `models/ensemble_models/`

**Comment ça marche**:
1. **Voting**: Chaque modèle vote et les votes sont moyennés
2. **Stacking**: Prédictions des modèles → métamodèle → prédiction finale

**Output Exemple**:
```
🎯 Ensemble Learning pour: rb1
  📊 Samples: 4000 train, 1000 test

  🗳️  Voting Classifier (soft voting)...
    ✅ Acc: 0.545 | F1: 0.502

  📚 Stacking Classifier...
    ✅ Acc: 0.552 | F1: 0.510

📋 RÉSUMÉ - MEILLEUR ENSEMBLE PAR CIBLE
======================================================================
rb1: Stacking        | F1: 0.510 | Acc: 0.552
```

---

## 📊 Stratégie de Test Recommandée

### Phase 1: Comparaison (30 min)
```bash
python scripts/train_and_improve.py
```
→ Identifie le meilleur modèle simple

### Phase 2: Optimisation (1-2 heures)
```bash
python scripts/hyperparameter_optimizer.py
```
→ Affine les hyperparamètres du meilleur modèle

### Phase 3: Ensemble (30 min)
```bash
python scripts/ensemble_optimizer.py
```
→ Combine les modèles optimisés

### Phase 4: Intégration
Charger les meilleurs modèles dans votre `predictor.py` :
```python
# Dans predictor.py
import pickle

# Charger l'ensemble optimal
with open('../models/ensemble_models/rb1_Stacking_ensemble.pkl', 'rb') as f:
    best_ensemble = pickle.load(f)

# Utiliser pour les prédictions
prediction = best_ensemble.predict(X_encoded)
```

---

## 🎛️ Personalisation

### Augmenter le nombre de cibles testées
Modifier `num_targets` dans `main()`:
```python
comparator.run_full_evaluation(num_targets=10)  # au lieu de 5
```

### Augmenter le nombre d'essais d'optimisation
Modifier `n_trials` dans `run_optimization()`:
```python
optimizer.run_optimization(targets, n_trials=100)  # au lieu de 50
```

### Ajouter d'autres modèles
Dans `train_and_improve.py`, ajouter dans le dictionnaire `models`:
```python
from sklearn.neighbors import KNeighborsClassifier
models['KNN'] = KNeighborsClassifier(n_neighbors=5)
```

---

## 📈 Métriques Expliquées

- **Accuracy**: % de prédictions correctes
- **F1-Macro**: Moyenne harmonique par classe, utile si classes imbalancées
- **Precision**: % de prédictions positives correctes
- **Recall**: % de vrais positifs détectés
- **Cross-Validation Std**: Variance - plus bas = plus stable

---

## ✅ Checklist de Résultats

Après avoir exécuté les 3 scripts, vous devriez avoir:
- [ ] `models/improved_models/` avec les meilleurs modèles simples
- [ ] Rapports de validation croisée pour chaque cible
- [ ] `models/ensemble_models/` avec Voting et Stacking
- [ ] Identification du meilleur modèle par cible

---

## 🔗 Intégration avec Roles.py

Pour utiliser les modèles améliorés dans votre script principal:

1. **Charger les modèles sauvegardés** plutôt que les réentraîner
2. **Remplacer LinearSVC par le meilleur modèle** dans `predictor.py`
3. **Garder le cache** pour les modèles améliorés

```python
# Dans predictor.py, remplacer le modèle LinearSVC
model_dir = "../models/ensemble_models"  # ou improved_models
# Charger le modèle pré-optimisé
```

---

## 🐛 Dépannage

**"ModuleNotFoundError: No module named 'lightgbm'"**
→ Installer: `pip install lightgbm`

**"Erreur: Données insuffisantes pour rb1"**
→ Vérifier que `csv_games_fusionnes.csv` contient assez de données

**Script lent?**
→ Réduire `num_targets` ou `n_trials` pour test rapide

---

## 📝 Notes

- Les modèles sont sauvegardés au format pickle
- Validation croisée 5-fold = plus robuste mais plus lent
- Ensemble Learning est généralement meilleur qu'un seul modèle
- Temps d'exécution typique:
  - `train_and_improve.py`: 5-10 min
  - `hyperparameter_optimizer.py`: 30-60 min
  - `ensemble_optimizer.py`: 5-15 min
