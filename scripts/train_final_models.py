"""scripts/train_final_models.py

Entraînement "final" par étape de draft en REUTILISANT les modèles déjà sélectionnés
et présents dans models/improved_models.

Principe:
- On charge, pour chaque target, le bundle source: models/improved_models/<target>_bundle.pkl
- On recrée le même type d'estimateur (mêmes params) puis on le ré-entraîne.
- On évalue sur un split:
    - BANS: F1-macro
    - PICKS: "récompense intelligente" (winrate vs picks adverses) + métriques classiques
- On ré-entraîne ensuite sur toutes les données et on sauvegarde un nouveau bundle.

Sortie: bundles par cible compatibles avec scripts/predictor.py
    models/<output-dir>/<target>_bundle.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from utils_roles import load_winrate_lookup


ALL_ROLES = {"top", "jng", "mid", "bot", "sup"}


@dataclass(frozen=True)
class TargetConfig:
    target: str
    features: list[str]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_draft_target_configs() -> list[TargetConfig]:
    # Même ordre/features que Roles/DraftPredictor
    return [
        TargetConfig("rb1", ["bb1"]),
        TargetConfig("bb2", ["bb1", "rb1"]),
        TargetConfig("rb2", ["bb1", "rb1", "bb2"]),
        TargetConfig("bb3", ["bb1", "rb1", "bb2", "rb2"]),
        TargetConfig("rb3", ["bb1", "rb1", "bb2", "rb2", "bb3"]),
        TargetConfig("bp1", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3"]),
        TargetConfig("rp1", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1"]),
        TargetConfig("rp2", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1"]),
        TargetConfig("bp2", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2"]),
        TargetConfig("bp3", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2"]),
        TargetConfig("rp3", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3"]),
        TargetConfig(
            "rb4",
            [
                "bb1",
                "rb1",
                "bb2",
                "rb2",
                "bb3",
                "rb3",
                "bp1",
                "rp1",
                "rp2",
                "bp2",
                "bp3",
                "rp3",
            ],
        ),
        TargetConfig(
            "bb4",
            [
                "bb1",
                "rb1",
                "bb2",
                "rb2",
                "bb3",
                "rb3",
                "bp1",
                "rp1",
                "rp2",
                "bp2",
                "bp3",
                "rp3",
                "rb4",
            ],
        ),
        TargetConfig(
            "rb5",
            [
                "bb1",
                "rb1",
                "bb2",
                "rb2",
                "bb3",
                "rb3",
                "bp1",
                "rp1",
                "rp2",
                "bp2",
                "bp3",
                "rp3",
                "rb4",
                "bb4",
            ],
        ),
        TargetConfig(
            "bb5",
            [
                "bb1",
                "rb1",
                "bb2",
                "rb2",
                "bb3",
                "rb3",
                "bp1",
                "rp1",
                "rp2",
                "bp2",
                "bp3",
                "rp3",
                "rb4",
                "bb4",
                "rb5",
            ],
        ),
        TargetConfig(
            "rp4",
            [
                "bb1",
                "rb1",
                "bb2",
                "rb2",
                "bb3",
                "rb3",
                "bp1",
                "rp1",
                "rp2",
                "bp2",
                "bp3",
                "rp3",
                "rb4",
                "bb4",
                "rb5",
                "bb5",
            ],
        ),
        TargetConfig(
            "bp4",
            [
                "bb1",
                "rb1",
                "bb2",
                "rb2",
                "bb3",
                "rb3",
                "bp1",
                "rp1",
                "rp2",
                "bp2",
                "bp3",
                "rp3",
                "rb4",
                "bb4",
                "rb5",
                "bb5",
                "rp4",
            ],
        ),
        TargetConfig(
            "bp5",
            [
                "bb1",
                "rb1",
                "bb2",
                "rb2",
                "bb3",
                "rb3",
                "bp1",
                "rp1",
                "rp2",
                "bp2",
                "bp3",
                "rp3",
                "rb4",
                "bb4",
                "rb5",
                "bb5",
                "rp4",
                "bp4",
            ],
        ),
        TargetConfig(
            "rp5",
            [
                "bb1",
                "rb1",
                "bb2",
                "rb2",
                "bb3",
                "rb3",
                "bp1",
                "rp1",
                "rp2",
                "bp2",
                "bp3",
                "rp3",
                "rb4",
                "bb4",
                "rb5",
                "bb5",
                "rp4",
                "bp4",
                "bp5",
            ],
        ),
    ]


def is_pick_target(target: str) -> bool:
    return target.startswith("bp") or target.startswith("rp")


def opponent_pick_cols_for(target: str, feature_cols: list[str]) -> list[str]:
    # Dans les features disponibles au moment t, les picks adverses déjà connus sont dans feature_cols
    if target.startswith("bp"):
        return [c for c in feature_cols if c.startswith("rp")]
    if target.startswith("rp"):
        return [c for c in feature_cols if c.startswith("bp")]
    return []


def build_pair_winrate_lookup(winrate_lookup: dict) -> dict[tuple[str, str], float]:
    """Pré-calcule (champion, opponent) -> winrate moyenne sur toutes positions."""
    pairs: dict[tuple[str, str], list[float]] = {}
    for (champ, _pos, opp, _opp_pos), wr in winrate_lookup.items():
        key = (champ, opp)
        pairs.setdefault(key, []).append(float(wr))

    return {k: float(np.mean(v)) for k, v in pairs.items() if v}


def intelligent_reward(
    predicted_champ: str,
    actual_champ: str,
    opponent_champs: list[str],
    pair_wr: dict[tuple[str, str], float],
) -> float:
    if predicted_champ == actual_champ:
        return 1.0
    if not opponent_champs:
        return 0.0

    wrs = []
    for opp in opponent_champs:
        wr = pair_wr.get((predicted_champ, opp))
        if wr is not None:
            wrs.append(wr)

    return float(np.mean(wrs)) if wrs else 0.0


def prepare_data(df: pd.DataFrame, target_col: str, feature_cols: list[str]):
    df_f = df.dropna(subset=feature_cols + [target_col])
    if df_f.empty:
        return None

    X = df_f[feature_cols]
    y = df_f[target_col]

    # Filtrage classes rares
    freq = y.value_counts()
    valid = freq[freq >= 2].index
    mask = y.isin(valid)
    X = X[mask]
    y = y[mask]
    df_ctx = df_f.loc[X.index]

    if X.shape[0] < 10 or y.nunique() < 2:
        return None

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_encoded = encoder.fit_transform(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_encoded, y_encoded, encoder, label_encoder, df_ctx


def recreate_estimator(estimator):
    try:
        return clone(estimator)
    except Exception:
        try:
            return estimator.__class__(**estimator.get_params())
        except Exception:
            return None


def load_source_bundle(source_dir: Path, target: str) -> dict | None:
    bundle_path = source_dir / f"{target}_bundle.pkl"
    if not bundle_path.exists():
        return None
    with open(bundle_path, "rb") as f:
        return pickle.load(f)


def evaluate_model_on_split(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    label_encoder: LabelEncoder,
    df_test_ctx: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    pair_wr: dict[tuple[str, str], float],
):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics classification
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    # Reward (picks only)
    reward_mean = None
    if is_pick_target(target_col):
        pred_champs = label_encoder.inverse_transform(np.array(y_pred, dtype=int))
        true_champs = label_encoder.inverse_transform(np.array(y_test, dtype=int))
        opp_cols = opponent_pick_cols_for(target_col, feature_cols)

        rewards = []
        # df_test_ctx index aligns with X_test rows via split on indices
        for i in range(len(pred_champs)):
            row = df_test_ctx.iloc[i]
            opponent_champs = [str(row[c]) for c in opp_cols if pd.notna(row[c])]
            rewards.append(
                intelligent_reward(
                    predicted_champ=str(pred_champs[i]),
                    actual_champ=str(true_champs[i]),
                    opponent_champs=opponent_champs,
                    pair_wr=pair_wr,
                )
            )
        reward_mean = float(np.mean(rewards)) if rewards else 0.0

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision": prec,
        "recall": rec,
        "reward_mean": reward_mean,
        "model": model,
    }


def save_bundle(
    output_dir: Path,
    target: str,
    feature_cols: list[str],
    model_name: str,
    model,
    encoder: OneHotEncoder,
    label_encoder: LabelEncoder,
    metrics: dict,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "target_col": target,
        "feature_cols": list(feature_cols),
        "model_name": model_name,
        "model": model,
        "encoder": encoder,
        "label_encoder": label_encoder,
        "metrics": metrics,
    }
    out_path = output_dir / f"{target}_bundle.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  💾 Bundle sauvegardé: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Entraînement final des modèles (reward pour picks)")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Chemin vers csv_games_fusionnes.csv (défaut: processed_data/csv_games_fusionnes.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Dossier de sortie (défaut: models/improved_models_intelligent)",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Dossier source des bundles existants (défaut: models/improved_models)",
    )
    parser.add_argument(
        "--num-targets",
        type=int,
        default=None,
        help="Limiter le nombre de cibles traitées (défaut: toutes)",
    )
    parser.add_argument(
        "--only-picks",
        action="store_true",
        help="N'entraîne/sauve que les picks (bp*/rp*).",
    )
    args = parser.parse_args()

    root = project_root()
    csv_path = Path(args.csv) if args.csv else (root / "processed_data" / "csv_games_fusionnes.csv")
    output_dir = Path(args.output_dir) if args.output_dir else (root / "models" / "improved_models_intelligent")
    source_dir = Path(args.source_dir) if args.source_dir else (root / "models" / "improved_models")

    print("📥 Chargement des données...")
    print(f"📁 Fichier: {csv_path}")
    if not csv_path.exists():
        raise SystemExit(f"❌ Fichier introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Load winrates (optional) and precompute pair lookup
    winrate_lookup = load_winrate_lookup()
    pair_wr = build_pair_winrate_lookup(winrate_lookup)
    if not pair_wr:
        print("⚠️  Winrates non disponibles: reward = 1 si exact match, sinon 0")

    configs = get_draft_target_configs()
    if args.only_picks:
        configs = [c for c in configs if is_pick_target(c.target)]
    if args.num_targets is not None:
        configs = configs[: args.num_targets]

    print("\n" + "=" * 70)
    print("🚀 ENTRAINEMENT FINAL (retrain des modèles de models/improved_models)")
    print("=" * 70)

    for cfg in configs:
        target = cfg.target
        print("\n" + "-" * 70)
        print(f"🎯 Target: {target}")

        src_bundle = load_source_bundle(source_dir, target)
        if src_bundle is None:
            print(f"  ⚠️  Bundle source introuvable: {source_dir / (target + '_bundle.pkl')}")
            continue

        src_model = src_bundle.get("model")
        if src_model is None:
            print("  ⚠️  Bundle source invalide (clé 'model' manquante)")
            continue

        feature_cols = list(src_bundle.get("feature_cols") or cfg.features)
        model_name = str(src_bundle.get("model_name") or src_model.__class__.__name__)

        base_model = recreate_estimator(src_model)
        if base_model is None:
            print(f"  ⚠️  Impossible de recréer l'estimateur ({model_name}); utilisation du modèle tel quel")
            base_model = src_model

        prepared = prepare_data(df, target, feature_cols)
        if prepared is None:
            print("  ⚠️  Données insuffisantes")
            continue

        X, y, encoder, label_encoder, df_ctx = prepared

        # Split indices to keep context rows aligned
        idx = np.arange(len(y))
        try:
            idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        df_test_ctx = df_ctx.iloc[idx_test].reset_index(drop=True)

        try:
            metrics = evaluate_model_on_split(
                model=base_model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                label_encoder=label_encoder,
                df_test_ctx=df_test_ctx,
                target_col=target,
                feature_cols=feature_cols,
                pair_wr=pair_wr,
            )
        except Exception as e:
            print(f"  ❌ Erreur entraînement/évaluation ({model_name}): {e}")
            continue

        if is_pick_target(target):
            print(
                f"  {model_name:18} | reward={metrics['reward_mean']:.3f} | "
                f"acc={metrics['accuracy']:.3f} | f1={metrics['f1_macro']:.3f}"
            )
        else:
            print(f"  {model_name:18} | acc={metrics['accuracy']:.3f} | f1={metrics['f1_macro']:.3f}")

        # Retrain on all data
        final_model = recreate_estimator(metrics["model"]) or metrics["model"]
        final_model.fit(X, y)

        metrics_out = {
            "selection_metric": "reward_mean" if is_pick_target(target) else "f1_macro",
            "reward_mean": metrics["reward_mean"],
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "n_samples": int(X.shape[0]),
            "n_classes": int(len(label_encoder.classes_)),
        }

        print(f"  ✅ Modèle source utilisé: {model_name}")
        save_bundle(
            output_dir=output_dir,
            target=target,
            feature_cols=feature_cols,
            model_name=model_name,
            model=final_model,
            encoder=encoder,
            label_encoder=label_encoder,
            metrics=metrics_out,
        )

    print("\n" + "=" * 70)
    print(f"✨ Terminé. Bundles dans: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
