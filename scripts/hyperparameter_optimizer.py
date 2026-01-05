"""
Script avancé pour optimisation fine des hyperparamètres.
Utilise Bayesian Optimization pour trouver les meilleurs paramètres plus rapidement.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
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

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[1]

    @staticmethod
    def get_draft_target_configs():
        """Liste ordonnée des cibles et des features disponibles à chaque étape."""
        return [
            ("rb1", ["bb1"]),
            ("bb2", ["bb1", "rb1"]),
            ("rb2", ["bb1", "rb1", "bb2"]),
            ("bb3", ["bb1", "rb1", "bb2", "rb2"]),
            ("rb3", ["bb1", "rb1", "bb2", "rb2", "bb3"]),
            ("bp1", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3"]),
            ("rp1", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1"]),
            ("rp2", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1"]),
            ("bp2", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2"]),
            ("bp3", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2"]),
            ("rp3", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3"]),
            ("rb4", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3"]),
            ("bb4", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3", "rb4"]),
            ("rb5", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3", "rb4", "bb4"]),
            ("bb5", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3", "rb4", "bb4", "rb5"]),
            ("rp4", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3", "rb4", "bb4", "rb5", "bb5"]),
            ("bp4", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3", "rb4", "bb4", "rb5", "bb5", "rp4"]),
            ("bp5", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3", "rb4", "bb4", "rb5", "bb5", "rp4", "bp4"]),
            ("rp5", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3", "rb4", "bb4", "rb5", "bb5", "rp4", "bp4", "bp5"]),
        ]
    
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

    def train_and_save_best_for_target(
        self,
        target_col,
        feature_cols,
        n_trials=50,
        output_dir=None,
        prefer_metric="cv_f1_macro",
    ):
        """Optimise puis entraîne le meilleur modèle et sauvegarde un bundle compatible predictor."""
        X, y, encoder, label_encoder = self.prepare_data(target_col, feature_cols)
        if X is None:
            print(f"  ⚠️  Données insuffisantes pour {target_col}")
            return None

        # Optimisation des hyperparamètres
        results = self.optimize_target(target_col, feature_cols, n_trials=n_trials)
        if not results:
            return None

        # Construire candidats (modèle + params) et scorer en CV
        candidates = []
        if 'LinearSVC_best' in results:
            C = float(results['LinearSVC_best'].get('C', 1.0))
            candidates.append((
                'LinearSVC_tuned',
                LinearSVC(C=C, max_iter=5000, dual=True, random_state=42),
            ))
        if 'RandomForest_best' in results:
            p = results['RandomForest_best']
            candidates.append((
                'RandomForest_tuned',
                RandomForestClassifier(
                    n_estimators=int(p.get('n_estimators', 50)),
                    max_depth=int(p.get('max_depth', 20)),
                    min_samples_split=int(p.get('min_samples_split', 10)),
                    min_samples_leaf=int(p.get('min_samples_leaf', 5)),
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=2,
                ),
            ))

        if not candidates:
            return None

        best_name = None
        best_model = None
        best_cv = -1.0
        for name, model in candidates:
            try:
                scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro', n_jobs=1)
                score = float(scores.mean())
                print(f"  🔁 CV check {name}: f1_macro={score:.4f}")
                if score > best_cv:
                    best_cv = score
                    best_name = name
                    best_model = model
            except Exception as e:
                print(f"  ❌ CV error {name}: {e}")

        if best_model is None:
            return None

        # Entraînement final sur toutes les données
        try:
            best_model.fit(X, y)
        except Exception as e:
            print(f"  ❌ Entraînement final échoué pour {target_col}: {e}")
            return None

        # Chemin output
        if output_dir is None:
            output_dir = str(self._project_root() / "models" / "improved_models")
        os.makedirs(output_dir, exist_ok=True)

        bundle = {
            'target_col': target_col,
            'feature_cols': list(feature_cols),
            'model_name': best_name,
            'model': best_model,
            'encoder': encoder,
            'label_encoder': label_encoder,
            'metrics': {
                'cv_f1_macro': float(best_cv),
                'n_samples': int(X.shape[0]),
                'n_classes': int(len(label_encoder.classes_)),
            },
        }

        out_path = os.path.join(output_dir, f"{target_col}_bundle.pkl")
        with open(out_path, 'wb') as f:
            import pickle
            pickle.dump(bundle, f)

        print(f"  ✅ Bundle sauvegardé: {out_path}")
        return bundle

    def run_full_optimization_and_save(self, n_trials=50, num_targets=None, output_dir=None):
        configs = self.get_draft_target_configs()
        if num_targets is not None:
            configs = configs[:num_targets]

        print("\n" + "="*70)
        print("🚀 OPTIMISATION + EXPORT BUNDLES (toutes les étapes)")
        print("="*70)

        saved = 0
        for target, features in configs:
            res = self.train_and_save_best_for_target(
                target,
                features,
                n_trials=n_trials,
                output_dir=output_dir,
            )
            if res is not None:
                saved += 1

        print(f"\n✅ Terminé. Bundles sauvegardés: {saved}/{len(configs)}")
    
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
        default=None,
        help="Chemin vers le fichier csv_games_fusionnes.csv"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Nombre de trials Optuna (si dispo) par cible",
    )
    parser.add_argument(
        "--num-targets",
        type=int,
        default=None,
        help="Limiter le nombre de cibles (par défaut: toutes)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Dossier de sortie pour les bundles (par défaut: models/improved_models)",
    )
    args = parser.parse_args()

    project_root = HyperparameterOptimizer._project_root()
    default_csv = project_root / "processed_data" / "csv_games_fusionnes.csv"
    csv_path = Path(args.csv) if args.csv else default_csv

    print("📥 Chargement des données...")
    print(f"📁 Fichier: {csv_path}")

    if not csv_path.exists():
        print(f"❌ Erreur: Le fichier {csv_path} n'existe pas")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"✅ Données chargées: {df.shape[0]} lignes")
    
    optimizer = HyperparameterOptimizer(df)

    optimizer.run_full_optimization_and_save(
        n_trials=args.n_trials,
        num_targets=args.num_targets,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
