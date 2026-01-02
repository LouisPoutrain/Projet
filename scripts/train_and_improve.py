"""
Script complet pour entraîner et améliorer les modèles de prédiction de draft LoL.
Teste et compare plusieurs algorithmes avec validation croisée et optimisation d'hyperparamètres.
"""

import os 
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Imports scikit-learn
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Imports gradient boosting (optionnels mais recommandés)
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️  LightGBM non installé. Installer avec: pip install lightgbm")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost non installé. Installer avec: pip install xgboost")


class ModelComparator:
    """
    Compare plusieurs modèles ML pour prédiction de draft LoL.
    """
    
    def __init__(self, df, model_dir="../models/saved_models"):
        self.df = df
        self.model_dir = model_dir
        self.results = {}
        self.best_models = {}
        
    def prepare_data(self, target_col, feature_cols):
        """
        Prépare les données pour l'entraînement.
        
        Returns:
            X_encoded, y_encoded, encoder, valid_indices, label_encoder
        """
        # Filtre les lignes avec données manquantes
        df_f = self.df.dropna(subset=feature_cols + [target_col])
        if df_f.empty:
            return None, None, None, None
        
        X = df_f[feature_cols]
        y = df_f[target_col]
        
        # Garde uniquement les classes apparaissant au moins 2 fois
        freq = y.value_counts()
        valid = freq[freq >= 2].index
        mask = y.isin(valid)
        X = X[mask]
        y = y[mask]
        
        if X.shape[0] < 10 or y.nunique() < 2:
            return None, None, None, None
        
        # One-hot encode
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_encoded = encoder.fit_transform(X)

        # Encode labels as integers (required by XGBoost >=2.0)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        return X_encoded, y_encoded, encoder, df_f[mask].index, label_encoder
    
    def evaluate_model(self, model_name, model, X_train, X_test, y_train, y_test):
        """
        Évalue un modèle et retourne les métriques.
        """
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calcul des métriques
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            
            return {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'precision': precision,
                'recall': recall,
                'model': model
            }
        except Exception as e:
            print(f"  ❌ Erreur avec {model_name}: {e}")
            return None
    
    def cross_validate_model(self, model_name, model, X, y, cv=5):
        """
        Validation croisée sur un modèle.
        """
        try:
            scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
            # Réduit n_jobs pour éviter les problèmes mémoire
            cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, 
                                       return_train_score=False, n_jobs=1)
            
            return {
                'accuracy_mean': cv_results['test_accuracy'].mean(),
                'accuracy_std': cv_results['test_accuracy'].std(),
                'f1_macro_mean': cv_results['test_f1_macro'].mean(),
                'f1_macro_std': cv_results['test_f1_macro'].std(),
                'precision_mean': cv_results['test_precision_macro'].mean(),
                'recall_mean': cv_results['test_recall_macro'].mean(),
            }
        except Exception as e:
            print(f"  ❌ Erreur CV avec {model_name}: {e}")
            return None
    
    def optimize_hyperparameters(self, model_name, model, X_train, y_train, param_grid, cv=5):
        """
        Optimise les hyperparamètres avec GridSearchCV.
        """
        try:
            grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, 
                                      scoring='f1_macro', verbose=0)
            grid_search.fit(X_train, y_train)
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_model': grid_search.best_estimator_
            }
        except Exception as e:
            print(f"  ❌ Erreur tuning avec {model_name}: {e}")
            return None
    
    def compare_models_for_target(self, target_col, feature_cols, verbose=True):
        """
        Compare tous les modèles pour une cible donnée.
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🎯 TARGET: {target_col}")
            print(f"{'='*70}")
        
        # Prépare les données
        X_encoded, y_encoded, encoder, valid_idx, label_encoder = self.prepare_data(target_col, feature_cols)
        if X_encoded is None:
            if verbose:
                print(f"  ⚠️  Données insuffisantes pour {target_col}")
            return None

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        if verbose:
            print(f"  📊 Samples: {len(X_encoded)} | Train: {len(X_train)} | Test: {len(X_test)}")
            print(f"  🏷️  Classes: {len(label_encoder.classes_)}")
        
        results_for_target = {}
        
        # Modèles à tester
        models = {
            'LinearSVC (baseline)': LinearSVC(C=1.0, max_iter=5000, dual=True, random_state=42),
            'LinearSVC (C=10)': LinearSVC(C=10.0, max_iter=5000, dual=True, random_state=42),
            'RandomForest': RandomForestClassifier(
                n_estimators=50,         # Réduit de 100 à 50
                max_depth=20,            # Limite la profondeur des arbres
                min_samples_split=10,    # Augmente le seuil de split
                min_samples_leaf=5,      # Augmente le nombre min de samples par feuille
                max_features='sqrt',     # Réduit les features considérées
                n_jobs=2,                # Limite le parallélisme pour économiser mémoire
                random_state=42
            ),
        }
        
        if HAS_LIGHTGBM:
            models['LightGBM'] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        
        if HAS_XGBOOST:
            models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, verbose=0)
        
        # Évaluation simple
        if verbose:
            print(f"\n📈 Test Set Performance (80/20 split):")
        
        for model_name, model in models.items():
            metrics = self.evaluate_model(model_name, model, X_train, X_test, y_train, y_test)
            
            if metrics:
                results_for_target[model_name] = metrics
                if verbose:
                    print(f"  {model_name:25} | "
                          f"Acc: {metrics['accuracy']:.3f} | "
                          f"F1: {metrics['f1_macro']:.3f} | "
                          f"Prec: {metrics['precision']:.3f} | "
                          f"Rec: {metrics['recall']:.3f}")
        
        # Validation croisée pour le meilleur modèle
        if results_for_target:
            best_model_name = max(results_for_target.keys(), 
                                 key=lambda x: results_for_target[x]['f1_macro'])
            best_model = results_for_target[best_model_name]['model']
            
            if verbose:
                print(f"\n🔄 Cross-Validation (5-fold) pour: {best_model_name}")
            
            cv_results = self.cross_validate_model(best_model_name, best_model, X_encoded, y_encoded, cv=5)
            
            if cv_results and verbose:
                print(f"  Accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
                print(f"  F1-Macro: {cv_results['f1_macro_mean']:.3f} ± {cv_results['f1_macro_std']:.3f}")
            
            results_for_target[best_model_name]['cv_results'] = cv_results
        
        return results_for_target
    
    def run_full_evaluation(self, num_targets=5):
        """
        Évalue les modèles sur les principales cibles.
        """
        print("\n" + "="*70)
        print("🚀 COMPARAISON COMPLÈTE DES MODÈLES")
        print("="*70)
        
        # Cibles principales
        configs = [
            ("rb1", ["bb1"]),
            ("bb2", ["bb1", "rb1"]),
            ("rb2", ["bb1", "rb1", "bb2"]),
            ("bp1", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3"]),
            ("rp1", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1"]),
        ]
        
        for i, (target, features) in enumerate(configs[:num_targets]):
            result = self.compare_models_for_target(target, features, verbose=True)
            self.results[target] = result
        
        # Résumé final
        self.print_summary()
    
    def print_summary(self):
        """
        Affiche un résumé des résultats.
        """
        print("\n" + "="*70)
        print("📋 RÉSUMÉ FINAL")
        print("="*70)
        
        for target, models_results in self.results.items():
            if models_results:
                best_model = max(models_results.keys(), 
                               key=lambda x: models_results[x]['f1_macro'])
                best_f1 = models_results[best_model]['f1_macro']
                print(f"{target}: {best_model} (F1: {best_f1:.3f})")
    
    def save_best_models(self, output_dir="../models/improved_models"):
        """
        Sauvegarde les meilleurs modèles.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for target, models_results in self.results.items():
            if models_results:
                best_model_name = max(models_results.keys(), 
                                     key=lambda x: models_results[x]['f1_macro'])
                best_model = models_results[best_model_name]['model']
                
                filepath = os.path.join(output_dir, f"{target}_model.pkl")
                with open(filepath, 'wb') as f:
                    pickle.dump(best_model, f)
        
        print(f"\n✅ Modèles sauvegardés dans: {output_dir}")


def main():
    """
    Fonction principale.
    """
    parser = argparse.ArgumentParser(description="Entraînement et amélioration des modèles LoL")
    parser.add_argument(
        "--csv",
        type=str,
        default="../processed_data/csv_games_fusionnes.csv",
        help="Chemin vers le fichier csv_games_fusionnes.csv"
    )
    args = parser.parse_args()
    
    # Charge les données
    print("📥 Chargement des données...")
    print(f"📁 Fichier: {args.csv}")
    
    if not os.path.exists(args.csv):
        print(f"❌ Erreur: Le fichier {args.csv} n'existe pas")
        sys.exit(1)
    
    df = pd.read_csv(args.csv)
    print(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Lance la comparaison
    comparator = ModelComparator(df)
    comparator.run_full_evaluation(num_targets=5)
    
    # Sauvegarde les meilleurs modèles
    comparator.save_best_models()
    
    print("\n" + "="*70)
    print("✨ Script d'amélioration terminé!")
    print("="*70)


if __name__ == "__main__":
    main()
