"""scripts/train_and_improve.py

Script pour entraîner et améliorer les modèles de prédiction de draft LoL.

Objectif:
- Comparer plusieurs algorithmes (baseline + alternatives) pour CHAQUE étape de draft.
- Sélectionner le meilleur modèle par cible (rb1, bb2, ..., bp5, rp5).
- Sauvegarder un "bundle" réutilisable pour l'inférence: modèle + OneHotEncoder + LabelEncoder
    + feature_cols + métadonnées.
- Enregistrer tous les résultats de comparaison (par cible et par modèle) dans un CSV.
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
from sklearn.base import clone
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
    
    def __init__(self, df, model_dir="../models/saved_models", use_xgboost: bool = True):
        self.df = df
        self.model_dir = model_dir
        self.results = {}
        self.best_models = {}
        self.use_xgboost = bool(use_xgboost)

    @staticmethod
    def _project_root() -> Path:
        # scripts/ -> project root
        return Path(__file__).resolve().parents[1]

    @staticmethod
    def _recreate_estimator(estimator):
        """Best-effort recreation of an estimator with same params (safer than clone for some libs)."""
        try:
            return clone(estimator)
        except Exception:
            try:
                return estimator.__class__(**estimator.get_params())
            except Exception:
                return None

    @staticmethod
    def _model_keys(models_results: dict):
        return [k for k in models_results.keys() if k != '_artifacts']

    def _save_best_bundle_for_target(self, target: str, models_results: dict, output_dir: str):
        if not models_results:
            return False

        artifacts = models_results.get('_artifacts')
        if not artifacts:
            return False

        model_keys = self._model_keys(models_results)
        if not model_keys:
            return False

        best_model_name = max(model_keys, key=lambda x: models_results[x]['f1_macro'])
        best_entry = models_results[best_model_name]
        fitted_model = best_entry['model']

        X_all = artifacts.get('X_encoded')
        y_all = artifacts.get('y_encoded')
        encoder = artifacts.get('encoder')
        label_encoder = artifacts.get('label_encoder')
        feature_cols = artifacts.get('feature_cols')

        final_model = self._recreate_estimator(fitted_model)
        if final_model is None:
            print(f"  ⚠️  Impossible de re-créer l'estimateur pour {target} ({best_model_name}); sauvegarde du modèle tel quel.")
            final_model = fitted_model
        else:
            try:
                final_model.fit(X_all, y_all)
            except Exception as e:
                print(f"  ⚠️  Ré-entraînement impossible pour {target} ({best_model_name}): {e}; sauvegarde du modèle tel quel.")
                final_model = fitted_model

        bundle = {
            'target_col': target,
            'feature_cols': feature_cols,
            'model_name': best_model_name,
            'model': final_model,
            'encoder': encoder,
            'label_encoder': label_encoder,
            'metrics': {
                'accuracy': float(best_entry.get('accuracy')),
                'f1_macro': float(best_entry.get('f1_macro')),
                'precision': float(best_entry.get('precision')),
                'recall': float(best_entry.get('recall')),
                'n_samples': int(artifacts.get('n_samples', 0)),
                'n_classes': int(artifacts.get('n_classes', 0)),
            },
            'cv_results': best_entry.get('cv_results'),
        }

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{target}_bundle.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(bundle, f)

        print(f"  💾 Bundle sauvegardé: {filepath}")
        return True
        
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

    @staticmethod
    def get_draft_target_configs():
        """Liste ordonnée des cibles et des features disponibles à chaque étape."""
        return [
            # Ban phase 1
            ("rb1", ["bb1"]),
            ("bb2", ["bb1", "rb1"]),
            ("rb2", ["bb1", "rb1", "bb2"]),
            ("bb3", ["bb1", "rb1", "bb2", "rb2"]),
            ("rb3", ["bb1", "rb1", "bb2", "rb2", "bb3"]),
            # Pick phase 1
            ("bp1", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3"]),
            ("rp1", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1"]),
            ("rp2", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1"]),
            ("bp2", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2"]),
            ("bp3", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2"]),
            ("rp3", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3"]),
            # Ban phase 2
            ("rb4", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3"]),
            ("bb4", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3", "rb4"]),
            ("rb5", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3", "rb4", "bb4"]),
            ("bb5", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3", "rb4", "bb4", "rb5"]),
            # Pick phase 2
            ("rp4", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3", "rb4", "bb4", "rb5", "bb5"]),
            ("bp4", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3", "rb4", "bb4", "rb5", "bb5", "rp4"]),
            ("bp5", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3", "rb4", "bb4", "rb5", "bb5", "rp4", "bp4"]),
            ("rp5", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3", "rb4", "bb4", "rb5", "bb5", "rp4", "bp4", "bp5"]),
        ]
    
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
            models['LightGBM'] = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                random_state=42,
                verbose=-1,
            )
        
        if HAS_XGBOOST and self.use_xgboost:
            models['XGBoost'] = XGBClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=2,
                verbosity=0,
                eval_metric="mlogloss",
            )
        
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
            best_model_name = max(
                results_for_target.keys(),
                key=lambda x: results_for_target[x]['f1_macro'],
            )
            best_model = results_for_target[best_model_name]['model']
            
            if verbose:
                print(f"\n🔄 Cross-Validation (5-fold) pour: {best_model_name}")
            
            cv_results = self.cross_validate_model(best_model_name, best_model, X_encoded, y_encoded, cv=5)
            
            if cv_results and verbose:
                print(f"  Accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
                print(f"  F1-Macro: {cv_results['f1_macro_mean']:.3f} ± {cv_results['f1_macro_std']:.3f}")
            
            results_for_target[best_model_name]['cv_results'] = cv_results

        # Stocke les artefacts une seule fois (évite de dupliquer X/y dans chaque entrée)
        results_for_target['_artifacts'] = {
            'encoder': encoder,
            'label_encoder': label_encoder,
            'feature_cols': list(feature_cols),
            'target_col': target_col,
            'n_samples': int(X_encoded.shape[0]),
            'n_classes': int(len(label_encoder.classes_)),
            'valid_indices': valid_idx,
            'X_encoded': X_encoded,
            'y_encoded': y_encoded,
        }
        
        return results_for_target
    
    @staticmethod
    def _flatten_dict(d: dict | None, prefix: str) -> dict:
        """Aplati un dict (1 niveau) en colonnes préfixées (utile pour cv_results)."""
        if not isinstance(d, dict):
            return {}
        return {f"{prefix}{k}": v for k, v in d.items()}

    def export_results_to_csv(self, csv_path: str | Path):
        """
        Exporte TOUS les résultats (par target x modèle) dans un CSV.

        Colonnes typiques:
          - target, model_name, is_best
          - accuracy, f1_macro, precision, recall
          - cv_* (si dispo)
          - n_samples, n_classes, n_features, feature_cols
        """
        rows = []

        for target, models_results in (self.results or {}).items():
            if not models_results:
                continue

            artifacts = models_results.get("_artifacts", {})
            model_keys = self._model_keys(models_results)
            if not model_keys:
                continue

            # Meilleur modèle selon la même règle que le bundle (F1 macro)
            best_model_name = max(model_keys, key=lambda x: models_results[x].get("f1_macro", -1))

            for model_name in model_keys:
                entry = models_results.get(model_name, {}) or {}
                cv = entry.get("cv_results")

                rows.append({
                    "target": target,
                    "model_name": model_name,
                    "is_best": model_name == best_model_name,
                    "accuracy": entry.get("accuracy"),
                    "f1_macro": entry.get("f1_macro"),
                    "precision": entry.get("precision"),
                    "recall": entry.get("recall"),
                    **self._flatten_dict(cv, "cv_"),
                    "n_samples": artifacts.get("n_samples"),
                    "n_classes": artifacts.get("n_classes"),
                    "n_features": int(artifacts.get("X_encoded").shape[1]) if artifacts.get("X_encoded") is not None else None,
                    "feature_cols": "|".join(artifacts.get("feature_cols", []) or []),
                })

        df_out = pd.DataFrame(rows)
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"\n🧾 Résultats complets exportés en CSV: {csv_path}")

    def run_full_evaluation(
        self,
        num_targets=None,
        only_missing: bool = False,
        output_dir: str | None = None,
        save_progressively: bool = True,
        skip_if_legacy_model_exists: bool = True,
        results_csv_path: str | None = None,
    ):
        """
        Évalue les modèles sur les cibles de draft.
        """
        print("\n" + "="*70)
        print("🚀 COMPARAISON COMPLÈTE DES MODÈLES")
        print("="*70)

        configs = self.get_draft_target_configs()
        if num_targets is not None:
            configs = configs[:num_targets]

        out_dir_path = Path(output_dir) if output_dir else None
        if only_missing and out_dir_path is not None:
            out_dir_path.mkdir(parents=True, exist_ok=True)

        for target, features in configs:
            if only_missing and out_dir_path is not None:
                bundle_path = out_dir_path / f"{target}_bundle.pkl"
                legacy_path = out_dir_path / f"{target}_model.pkl"
                if bundle_path.exists():
                    print(f"\n⏭️  Skip {target}: bundle déjà présent ({bundle_path})")
                    continue
                if skip_if_legacy_model_exists and legacy_path.exists():
                    print(f"\n⏭️  Skip {target}: legacy model présent ({legacy_path})")
                    continue

            result = self.compare_models_for_target(target, features, verbose=True)
            self.results[target] = result

            if save_progressively and out_dir_path is not None and result:
                # Sauvegarde le meilleur bundle immédiatement pour ne pas perdre le progrès
                self._save_best_bundle_for_target(target, result, str(out_dir_path))

        # Résumé final
        self.print_summary()

        # --- NOUVEAU: export CSV de tous les résultats ---
        if results_csv_path:
            self.export_results_to_csv(results_csv_path)
        else:
            # Par défaut: dans output_dir si fourni, sinon à la racine/models/improved_models
            default_dir = out_dir_path if out_dir_path is not None else (self._project_root() / "models" / "improved_models")
            self.export_results_to_csv(default_dir / "model_comparison_results.csv")

    def print_summary(self):
        """
        Affiche un résumé des résultats.
        """
        print("\n" + "="*70)
        print("📋 RÉSUMÉ FINAL")
        print("="*70)
        
        for target, models_results in self.results.items():
            if models_results:
                model_keys = self._model_keys(models_results)
                if not model_keys:
                    continue
                best_model = max(model_keys, key=lambda x: models_results[x]['f1_macro'])
                best_f1 = models_results[best_model]['f1_macro']
                print(f"{target}: {best_model} (F1: {best_f1:.3f})")
    
    def save_best_models(self, output_dir=None):
        """
        Sauvegarde les meilleurs modèles sous forme de bundles réutilisables.

        Bundle sauvé (pickle):
          - target_col
          - feature_cols
          - model_name
          - model (ré-entraîné sur toutes les données valides)
          - encoder (OneHotEncoder fitted)
          - label_encoder (LabelEncoder fitted)
          - metrics (sur split test)
          - cv_results (si dispo)
        """

        if output_dir is None:
            output_dir = str(self._project_root() / "models" / "improved_models")

        os.makedirs(output_dir, exist_ok=True)

        saved = 0
        
        for target, models_results in self.results.items():
            if models_results:
                if self._save_best_bundle_for_target(target, models_results, output_dir):
                    saved += 1
        
        print(f"\n✅ Bundles sauvegardés: {saved} | Dossier: {output_dir}")


def main():
    """
    Fonction principale.
    """
    parser = argparse.ArgumentParser(description="Entraînement et amélioration des modèles LoL")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Chemin vers le fichier csv_games_fusionnes.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Dossier de sortie pour les bundles (par défaut: models/improved_models à la racine)",
    )
    # --- NOUVEAU ---
    parser.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help="Chemin de sortie du CSV récapitulatif (par défaut: <output-dir>/model_comparison_results.csv).",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Ne réentraîne que les cibles dont le bundle n'existe pas encore dans --output-dir (et, par défaut, skip si legacy *_model.pkl existe).",
    )
    parser.add_argument(
        "--no-skip-legacy",
        action="store_true",
        help="Avec --only-missing, ne considère PAS les fichiers legacy *_model.pkl comme existants.",
    )
    parser.add_argument(
        "--no-progress-save",
        action="store_true",
        help="Désactive la sauvegarde progressive (par cible).",
    )
    parser.add_argument(
        "--no-xgboost",
        action="store_true",
        help="Désactive XGBoost même s'il est installé (accélère l'entraînement).",
    )
    parser.add_argument(
        "--num-targets",
        type=int,
        default=None,
        help="Limiter le nombre de cibles évaluées (par défaut: toutes)",
    )
    args = parser.parse_args()

    project_root = ModelComparator._project_root()
    default_csv = project_root / "processed_data" / "csv_games_fusionnes.csv"
    csv_path = Path(args.csv) if args.csv else default_csv

    effective_output_dir = args.output_dir or str(project_root / "models" / "improved_models")
    
    # Charge les données
    print("📥 Chargement des données...")
    print(f"📁 Fichier: {csv_path}")

    if not csv_path.exists():
        print(f"❌ Erreur: Le fichier {csv_path} n'existe pas")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Lance la comparaison
    comparator = ModelComparator(df, use_xgboost=not args.no_xgboost)
    comparator.run_full_evaluation(
        num_targets=args.num_targets,
        only_missing=args.only_missing,
        output_dir=effective_output_dir,
        save_progressively=not args.no_progress_save,
        skip_if_legacy_model_exists=not args.no_skip_legacy,
        results_csv_path=args.results_csv,
    )
    
    # Sauvegarde les meilleurs modèles
    comparator.save_best_models(output_dir=effective_output_dir)
    
    print("\n" + "="*70)
    print("✨ Script d'amélioration terminé!")
    print("="*70)


if __name__ == "__main__":
    main()
