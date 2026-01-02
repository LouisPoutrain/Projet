"""
Script avancé pour optimisation fine des hyperparamètres.
Utilise Bayesian Optimization pour trouver les meilleurs paramètres plus rapidement.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# Optionnel mais recommandé pour optimisation
try:
    from optuna import create_study, Trial, visualization
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠️  Optuna non installé. Installer avec: pip install optuna")


class HyperparameterOptimizer:
    """
    Optimise les hyperparamètres des modèles avec Optuna (ou GridSearch en fallback).
    """
    
    def __init__(self, df):
        self.df = df
        self.best_params = {}
    
    def prepare_data(self, target_col, feature_cols):
        """Prépare les données."""
        df_f = self.df.dropna(subset=feature_cols + [target_col])
        if df_f.empty:
            return None, None, None, None

        X = df_f[feature_cols]
        y = df_f[target_col]

        # Garde classes >= 2 occurrences
        freq = y.value_counts()
        valid = freq[freq >= 2].index
        mask = y.isin(valid)
        X = X[mask]
        y = y[mask]

        if X.shape[0] < 10 or y.nunique() < 2:
            return None, None, None, None

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_encoded = encoder.fit_transform(X)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        return X_encoded, y_encoded, encoder, label_encoder
    
    def objective_linearsvc(self, trial: Trial, X, y):
        """Objectif d'optimisation pour LinearSVC."""
        C = trial.suggest_float('C', 0.01, 100, log=True)
        
        model = LinearSVC(C=C, max_iter=5000, dual=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
        
        return scores.mean()
    
    def objective_randomforest(self, trial: Trial, X, y):
        """Objectif d'optimisation pour RandomForest."""
        n_estimators = trial.suggest_int('n_estimators', 30, 100)  # Réduit de 50-300 à 30-100
        max_depth = trial.suggest_int('max_depth', 10, 30)  # Réduit de 5-50 à 10-30
        min_samples_split = trial.suggest_int('min_samples_split', 5, 20)  # Augmente min de 2 à 5
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 3, 10)  # Augmente min de 1 à 3
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features='sqrt',  # Réduit features considérées
            random_state=42,
            n_jobs=2  # Limite parallélisme
        )
        scores = cross_val_score(model, X, y, cv=3, scoring='f1_macro', n_jobs=1)  # CV=3 au lieu de 5
        
        return scores.mean()
    
    def optimize_target(self, target_col, feature_cols, n_trials=50):
        """
        Optimise les hyperparamètres pour une cible spécifique.
        """
        print(f"\n🔧 Optimisation pour: {target_col}")
        
        X, y, encoder, label_encoder = self.prepare_data(target_col, feature_cols)
        if X is None:
            print(f"  ⚠️  Données insuffisantes")
            return None
        
        print(f"  📊 Samples: {X.shape[0]}, Features: {X.shape[1]}")
        
        results = {}
        
        # 1. LinearSVC Optimization
        print(f"  🔍 LinearSVC optimization...")
        try:
            if HAS_OPTUNA:
                study_svc = create_study(direction='maximize')
                study_svc.optimize(
                    lambda trial: self.objective_linearsvc(trial, X, y),
                    n_trials=n_trials,
                    show_progress_bar=False
                )
                best_svc = study_svc.best_params
                print(f"    ✅ Best C: {best_svc['C']:.4f}")
                results['LinearSVC_best'] = best_svc
            else:
                # Fallback: simple grid search
                from sklearn.model_selection import GridSearchCV
                grid = GridSearchCV(LinearSVC(max_iter=5000, dual=True, random_state=42),
                                  {'C': [0.01, 0.1, 1, 10, 100]},
                                  cv=3, scoring='f1_macro', n_jobs=1)  # CV=3, n_jobs=1
                grid.fit(X, y)
                print(f"    ✅ Best C: {grid.best_params_['C']:.4f}")
                results['LinearSVC_best'] = grid.best_params_
        except Exception as e:
            print(f"    ❌ Erreur: {e}")
        
        # 2. RandomForest Optimization
        print(f"  🔍 RandomForest optimization...")
        try:
            if HAS_OPTUNA:
                study_rf = create_study(direction='maximize')
                study_rf.optimize(
                    lambda trial: self.objective_randomforest(trial, X, y),
                    n_trials=n_trials,
                    show_progress_bar=False
                )
                best_rf = study_rf.best_params
                print(f"    ✅ Best params: {best_rf}")
                results['RandomForest_best'] = best_rf
            else:
                from sklearn.model_selection import GridSearchCV
                grid = GridSearchCV(
                    RandomForestClassifier(
                        random_state=42, 
                        n_jobs=2, 
                        max_features='sqrt'
                    ),
                    {
                        'n_estimators': [50, 100],  # Réduit de [100, 200]
                        'max_depth': [15, 25],  # Réduit de [10, 20, 30]
                        'min_samples_split': [5, 10]
                    },
                    cv=3, scoring='f1_macro', n_jobs=1  # CV=3, n_jobs=1
                )
                grid.fit(X, y)
                print(f"    ✅ Best params: {grid.best_params_}")
                results['RandomForest_best'] = grid.best_params_
        except Exception as e:
            print(f"    ❌ Erreur: {e}")
        
        return results
    
    def run_optimization(self, targets_and_features, n_trials=50):
        """
        Lance l'optimisation pour plusieurs cibles.
        """
        print("\n" + "="*70)
        print("🚀 OPTIMISATION DES HYPERPARAMÈTRES")
        print("="*70)
        
        for target, features in targets_and_features:
            self.optimize_target(target, features, n_trials=n_trials)
        
        print("\n" + "="*70)
        print("✨ Optimisation terminée!")
        print("="*70)


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Optimisation des hyperparamètres")
    parser.add_argument(
        "--csv",
        type=str,
        default="../processed_data/csv_games_fusionnes.csv",
        help="Chemin vers le fichier csv_games_fusionnes.csv"
    )
    args = parser.parse_args()
    
    print("📥 Chargement des données...")
    print(f"📁 Fichier: {args.csv}")
    
    if not os.path.exists(args.csv):
        print(f"❌ Erreur: Le fichier {args.csv} n'existe pas")
        sys.exit(1)
    
    df = pd.read_csv(args.csv)
    print(f"✅ Données chargées: {df.shape[0]} lignes")
    
    optimizer = HyperparameterOptimizer(df)
    
    # Cibles à optimiser
    targets = [
        ("rb1", ["bb1"]),
        ("bb2", ["bb1", "rb1"]),
        ("rb2", ["bb1", "rb1", "bb2"]),
        ("bp1", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3"]),
    ]
    
    optimizer.run_optimization(targets, n_trials=50)


if __name__ == "__main__":
    main()
