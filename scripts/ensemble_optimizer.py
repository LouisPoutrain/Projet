"""
Script pour Ensemble Learning - Combine plusieurs modèles pour meilleure performance.
Utilise Voting et Stacking pour améliorer les prédictions.
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier
)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class EnsembleOptimizer:
    """
    Crée et optimise des ensembles de modèles (Voting & Stacking).
    """
    
    def __init__(self, df):
        self.df = df
        self.results = {}
    
    def prepare_data(self, target_col, feature_cols):
        """Prépare les données."""
        df_f = self.df.dropna(subset=feature_cols + [target_col])
        if df_f.empty:
            return None, None, None, None, None

        X = df_f[feature_cols]
        y = df_f[target_col]

        freq = y.value_counts()
        valid = freq[freq >= 2].index
        mask = y.isin(valid)
        X = X[mask]
        y = y[mask]

        if X.shape[0] < 10 or y.nunique() < 2:
            return None, None, None, None, None

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_encoded = encoder.fit_transform(X)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        return X_encoded, y_encoded, encoder, df_f[mask].index, label_encoder
    
    def create_voting_classifier(self):
        """
        Crée un Voting Classifier avec plusieurs algorithmes.
        """
        estimators = [
            ('svc', LinearSVC(C=1.0, max_iter=5000, dual=True, random_state=42)),
            ('rf', RandomForestClassifier(
                n_estimators=50,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                n_jobs=2,
                random_state=42
            )),
            ('ada', AdaBoostClassifier(n_estimators=50, random_state=42))
        ]
        
        if HAS_LIGHTGBM:
            estimators.append(('lgb', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)))
        
        if HAS_XGBOOST:
            estimators.append(('xgb', XGBClassifier(n_estimators=100, random_state=42, verbose=0)))
        
        return VotingClassifier(estimators=estimators, voting='soft')
    
    def create_stacking_classifier(self):
        """
        Crée un Stacking Classifier.
        Les prédictions des modèles de base sont combinées par un meta-learner.
        """
        base_estimators = [
            ('svc', LinearSVC(C=1.0, max_iter=5000, dual=True, random_state=42, probability=False)),
            ('rf', RandomForestClassifier(
                n_estimators=50,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                n_jobs=2,
                random_state=42
            )),
            ('ada', AdaBoostClassifier(n_estimators=50, random_state=42))
        ]
        
        if HAS_LIGHTGBM:
            base_estimators.append(('lgb', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)))
        
        # Meta-learner
        meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        
        return StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=3,  # Réduit de 5 à 3
            n_jobs=1  # Limite parallélisme
        )
    
    def evaluate_ensemble(self, name, ensemble, X_train, X_test, y_train, y_test):
        """
        Évalue un ensemble.
        """
        try:
            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_test)
            
            return {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
                'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'model': ensemble
            }
        except Exception as e:
            print(f"  ❌ Erreur avec {name}: {e}")
            return None
    
    def optimize_target(self, target_col, feature_cols, verbose=True):
        """
        Optimise les ensembles pour une cible donnée.
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🎯 Ensemble Learning pour: {target_col}")
            print(f"{'='*70}")
        
        X_encoded, y_encoded, encoder, valid_idx, label_encoder = self.prepare_data(target_col, feature_cols)
        if X_encoded is None:
            if verbose:
                print(f"  ⚠️  Données insuffisantes")
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        if verbose:
            print(f"  📊 Samples: {X_train.shape[0]} train, {X_test.shape[0]} test")
        
        results = {}
        
        # 1. Voting Classifier
        if verbose:
            print(f"\n  🗳️  Voting Classifier (soft voting)...")
        voting_ensemble = self.create_voting_classifier()
        voting_results = self.evaluate_ensemble("Voting", voting_ensemble, X_train, X_test, y_train, y_test)
        
        if voting_results:
            results['Voting'] = voting_results
            if verbose:
                print(f"    ✅ Acc: {voting_results['accuracy']:.3f} | "
                      f"F1: {voting_results['f1_macro']:.3f}")
        
        # 2. Stacking Classifier
        if verbose:
            print(f"\n  📚 Stacking Classifier...")
        stacking_ensemble = self.create_stacking_classifier()
        stacking_results = self.evaluate_ensemble("Stacking", stacking_ensemble, X_train, X_test, y_train, y_test)
        
        if stacking_results:
            results['Stacking'] = stacking_results
            if verbose:
                print(f"    ✅ Acc: {stacking_results['accuracy']:.3f} | "
                      f"F1: {stacking_results['f1_macro']:.3f}")
        
        return results
    
    def run_full_optimization(self, num_targets=5):
        """
        Exécute l'optimisation complète sur plusieurs cibles.
        """
        print("\n" + "="*70)
        print("🚀 ENSEMBLE LEARNING - OPTIMISATION COMPLÈTE")
        print("="*70)
        
        configs = [
            ("rb1", ["bb1"]),
            ("bb2", ["bb1", "rb1"]),
            ("rb2", ["bb1", "rb1", "bb2"]),
            ("bp1", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3"]),
            ("rp1", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1"]),
        ]
        
        for target, features in configs[:num_targets]:
            result = self.optimize_target(target, features, verbose=True)
            self.results[target] = result
        
        self.print_summary()
    
    def print_summary(self):
        """
        Affiche un résumé comparatif.
        """
        print("\n" + "="*70)
        print("📋 RÉSUMÉ - MEILLEUR ENSEMBLE PAR CIBLE")
        print("="*70)
        
        for target, ensembles in self.results.items():
            if ensembles:
                best_name = max(ensembles.keys(), 
                               key=lambda x: ensembles[x]['f1_macro'])
                best_f1 = ensembles[best_name]['f1_macro']
                best_acc = ensembles[best_name]['accuracy']
                print(f"{target}: {best_name:15} | F1: {best_f1:.3f} | Acc: {best_acc:.3f}")
    
    def save_best_ensembles(self, output_dir="../models/ensemble_models"):
        """
        Sauvegarde les meilleurs ensembles.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for target, ensembles in self.results.items():
            if ensembles:
                best_name = max(ensembles.keys(), 
                               key=lambda x: ensembles[x]['f1_macro'])
                best_model = ensembles[best_name]['model']
                
                filepath = os.path.join(output_dir, f"{target}_{best_name}_ensemble.pkl")
                with open(filepath, 'wb') as f:
                    pickle.dump(best_model, f)
        
        print(f"\n✅ Ensembles sauvegardés dans: {output_dir}")


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Ensemble Learning pour modèles LoL")
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
    
    optimizer = EnsembleOptimizer(df)
    optimizer.run_full_optimization(num_targets=5)
    optimizer.save_best_ensembles()
    
    print("\n" + "="*70)
    print("✨ Ensemble Learning terminé!")
    print("="*70)


if __name__ == "__main__":
    main()
