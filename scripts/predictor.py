"""
DraftPredictor class for LoL esports draft prediction.
Handles model training, caching, and champion predictions.
"""
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.extmath import softmax
from sklearn.model_selection import train_test_split
from utils_roles import load_winrate_lookup


class DraftPredictor:
    """
    Main predictor class for LoL draft stages (bans and picks).
    Supports role-aware predictions with intelligent reward computation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        model_dir="../models/saved_models",
        reward_type="standard",
        retrain_mode=False,
        champion_roles_map=None,
        prefer_improved_models: bool = True,
        improved_models_dir: str | None = None,
    ):
        """
        Initialize DraftPredictor.
        
        Args:
            df: DataFrame with game data
            model_dir: Base directory for model storage
            reward_type: "standard" or "intelligent_reward"
            retrain_mode: Force model retraining from scratch
            champion_roles_map: Dict mapping champion -> list of positions
        """
        self.df = df
        self._model_cache = {}
        self.reward_type = reward_type
        self.retrain_mode = retrain_mode
        self.champion_roles_map = champion_roles_map or {}
        self.winrate_lookup = load_winrate_lookup()

        self.prefer_improved_models = prefer_improved_models
        self._bundle_cache = {}

        project_root = Path(__file__).resolve().parents[1]
        self.improved_models_dir = Path(improved_models_dir) if improved_models_dir else (project_root / "models" / "improved_models")
        
        # Build model directory structure
        base_model_dir = os.path.dirname(model_dir)
        self.model_dir_base = os.path.join(base_model_dir, reward_type)
        self.model_dir = self.model_dir_base
        
        # Create subdirectories
        for subdir in ["bans", "picks"]:
            path = os.path.join(self.model_dir_base, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
        
        mode_str = "(RETRAIN MODE)" if self.retrain_mode else ""
        print(f"Using model directory: {self.model_dir_base} {mode_str}")

        if self.prefer_improved_models:
            print(f"Improved models enabled: {self.improved_models_dir}")

    def _load_improved_bundle(self, target_col: str, feature_cols: list):
        """Load a per-target bundle saved by scripts/train_and_improve.py.

        Bundle format:
          - target_col, feature_cols
          - model, encoder, label_encoder (optional)
        """
        bundle_path = self.improved_models_dir / f"{target_col}_bundle.pkl"
        if not bundle_path.exists():
            return None

        try:
            with open(bundle_path, "rb") as f:
                bundle = pickle.load(f)
        except Exception as e:
            print(f"⚠️  Impossible de charger le bundle: {bundle_path} ({e})")
            return None

        if not isinstance(bundle, dict):
            return None

        bundle_features = bundle.get("feature_cols")
        if bundle.get("target_col") != target_col or list(bundle_features or []) != list(feature_cols):
            # Safety: don't use a bundle trained for a different feature set
            return None

        if "model" not in bundle or "encoder" not in bundle:
            return None

        return bundle

    def compute_intelligent_reward(self, predicted_champ, actual_champ, my_position, opponent_picks):
        """
        Compute intelligent reward based on win rates vs opponent picks.
        Averages win rates across all position matchups (position-agnostic).
        
        - Exact match: 1.0
        - Different champion: average win rate vs opponents (all positions)
        - No opponent data: 0.0
        
        Args:
            predicted_champ: Predicted champion
            actual_champ: Actual champion
            my_position: Position of current pick (not used for calculation)
            opponent_picks: List of (champion, position) tuples
            
        Returns:
            float: Reward score (0.0-1.0)
        """
        if predicted_champ == actual_champ:
            return 1.0
        
        if not opponent_picks:
            return 0.0
        
        # Collect all win rates for predicted_champ vs each opponent across all positions
        all_win_rates = []
        for opp_champ, opp_pos in opponent_picks:
            # Find all matchup win rates regardless of position
            matchup_rates = []
            for key, wr in self.winrate_lookup.items():
                champ, pos, opponent, opp_position = key
                if champ == predicted_champ and opponent == opp_champ:
                    matchup_rates.append(wr)
            
            # Average win rates for this opponent across all positions
            if matchup_rates:
                all_win_rates.append(np.mean(matchup_rates))
        
        # Return average across all opponents
        return np.mean(all_win_rates) if all_win_rates else 0.0

    def _predict_ml(self, target_col, feature_cols, result_filter,
                     feature_values, team_filled_roles=None, global_used_champs=None):
        """
        Predict using a trained LinearSVC model.
        Handles model caching, training, and role-aware filtering.
        
        Returns:
            str: Predicted champion
        """
        model_encoder_cache_key = f"model_encoder_{target_col}_{result_filter}_{'_'.join(feature_cols)}"

        # 1) Try improved bundle first (if enabled)
        if self.prefer_improved_models:
            bundle_cache_key = f"bundle_{target_col}_{'_'.join(feature_cols)}"
            bundle = self._bundle_cache.get(bundle_cache_key)
            if bundle is None:
                bundle = self._load_improved_bundle(target_col, feature_cols)
                if bundle is not None:
                    self._bundle_cache[bundle_cache_key] = bundle

            if bundle is not None:
                model = bundle.get("model")
                encoder = bundle.get("encoder")
                label_encoder = bundle.get("label_encoder")

                X_input_df = pd.DataFrame([feature_values], columns=feature_cols)
                X_input_encoded = encoder.transform(X_input_df)

                # Compute probabilities using available API
                classes_raw = getattr(model, "classes_", None)
                probas = None

                if hasattr(model, "decision_function"):
                    scores = model.decision_function(X_input_encoded)
                    scores = scores[0] if hasattr(scores, "__len__") else scores
                    if isinstance(scores, float) or np.ndim(scores) == 0:
                        scores = np.array([-scores, scores])
                    probas = softmax([scores])[0]
                elif hasattr(model, "predict_proba"):
                    probas = model.predict_proba(X_input_encoded)[0]
                else:
                    # Hard fallback: no scores available
                    pred_raw = model.predict(X_input_encoded)[0]
                    if label_encoder is not None:
                        try:
                            return label_encoder.inverse_transform([int(pred_raw)])[0]
                        except Exception:
                            return str(pred_raw)
                    return pred_raw

                # Determine class labels aligned with probas
                if classes_raw is None:
                    # Try to infer from label encoder if sizes match
                    if label_encoder is not None and len(probas) == len(label_encoder.classes_):
                        champion_classes = np.array(label_encoder.classes_, dtype=object)
                    else:
                        # Cannot align probas to champions safely
                        pred_raw = model.predict(X_input_encoded)[0]
                        if label_encoder is not None:
                            try:
                                return label_encoder.inverse_transform([int(pred_raw)])[0]
                            except Exception:
                                return str(pred_raw)
                        return pred_raw
                else:
                    if label_encoder is not None:
                        try:
                            champion_classes = label_encoder.inverse_transform(np.array(classes_raw, dtype=int))
                        except Exception:
                            champion_classes = np.array([str(c) for c in classes_raw], dtype=object)
                    else:
                        champion_classes = np.array(classes_raw, dtype=object)

                # Remove already-picked champs (global mask)
                used = set(global_used_champs) if global_used_champs else set()
                mask = np.array([c not in used for c in champion_classes])
                probas = probas * mask

                # Role-aware penalization for picks
                if team_filled_roles is not None:
                    all_roles = {'top', 'jng', 'mid', 'bot', 'sup'}
                    unfilled_roles = all_roles - team_filled_roles

                    if len(unfilled_roles) > 0:
                        for i, champion in enumerate(champion_classes):
                            if probas[i] > 0:
                                champion_possible_roles = self.champion_roles_map.get(champion, [])
                                can_fill_unfilled = any(role in unfilled_roles for role in champion_possible_roles)

                                if not can_fill_unfilled:
                                    if len(champion_possible_roles) > 0:
                                        probas[i] *= 0.1  # Moderate penalty
                                    else:
                                        probas[i] *= 0.05  # Stronger penalty

                # Fallback if everything is masked
                if probas.sum() == 0:
                    remaining = [c for c in champion_classes if c not in used]
                    return remaining[0] if remaining else champion_classes[0]

                return champion_classes[int(np.argmax(probas))]
        
        # Determine subdirectory: bans or picks
        subdir = "bans" if target_col[1:] in ["b1", "b2", "b3", "b4", "b5"] else "picks"
        model_file = os.path.join(self.model_dir_base, subdir, f"{model_encoder_cache_key}.pkl")

        # Load cached model if available - unless RETRAIN_MODE is on
        if not self.retrain_mode and model_encoder_cache_key in self._model_cache:
            model, encoder = self._model_cache[model_encoder_cache_key]
        elif not self.retrain_mode and os.path.exists(model_file):
            print(f"Chargement du modèle depuis : {model_file}")
            with open(model_file, 'rb') as f:
                model, encoder = pickle.load(f)
            self._model_cache[model_encoder_cache_key] = (model, encoder)
        else:
            # Train a new model (or retrain if RETRAIN_MODE)
            retrain_str = "(RETRAIN)" if self.retrain_mode else ""
            print(f"Entraînement d'un nouveau modèle pour : {target_col} {retrain_str}")
            
            df_f = self.df.dropna(subset=feature_cols + [target_col])
            X = df_f[feature_cols]
            y = df_f[target_col]

            # Keep classes appearing at least twice
            freq = y.value_counts()
            valid = freq[freq >= 2].index
            mask = y.isin(valid)
            X = X[mask]
            y = y[mask]

            # One-hot encode
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            X_encoded = encoder.fit_transform(X)

            # Split data: 80% train, 20% test (random split, no fixed seed for variance)
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y, test_size=0.2, random_state=None, stratify=y
                )
            except ValueError:
                # If stratification fails, split without stratification
                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y, test_size=0.2, random_state=None
                )

            # Train LinearSVC on 80% training set
            model = LinearSVC(C=1.0, max_iter=5000, dual=True)
            model.fit(X_train, y_train)

            # Cache in memory
            self._model_cache[model_encoder_cache_key] = (model, encoder)
            
            # Save to disk
            with open(model_file, 'wb') as f:
                pickle.dump((model, encoder), f)
            print(f"Modèle sauvegardé dans : {model_file}")

        # Retrieve cached model + encoder
        model, encoder = self._model_cache[model_encoder_cache_key]

        # Encode input features
        X_input_df = pd.DataFrame([feature_values], columns=feature_cols)
        X_input_encoded = encoder.transform(X_input_df)

        # Get decision scores and convert to pseudo-probabilities
        scores = model.decision_function(X_input_encoded)[0]
        if isinstance(scores, float) or np.ndim(scores) == 0:
            scores = np.array([-scores, scores])

        probas = softmax([scores])[0]
        classes = model.classes_

        # Remove already-picked champs
        used = set(global_used_champs) if global_used_champs else set()
        mask = np.array([c not in used for c in classes])
        probas = probas * mask

        # Role-aware penalization for picks
        if team_filled_roles is not None:
            all_roles = {'top', 'jng', 'mid', 'bot', 'sup'}
            unfilled_roles = all_roles - team_filled_roles

            if len(unfilled_roles) > 0:
                for i, champion in enumerate(classes):
                    if probas[i] > 0:
                        champion_possible_roles = self.champion_roles_map.get(champion, [])
                        can_fill_unfilled = any(role in unfilled_roles for role in champion_possible_roles)

                        if not can_fill_unfilled:
                            if len(champion_possible_roles) > 0:
                                probas[i] *= 0.1  # Moderate penalty
                            else:
                                probas[i] *= 0.05  # Stronger penalty

        # Fallback if everything is masked
        if probas.sum() == 0:
            remaining = [c for c in classes if c not in used]
            return remaining[0] if remaining else classes[0]

        return classes[np.argmax(probas)]

    # ===== BAN PHASE 1 =====
    def predict_bb1(self):
        """Predict Blue Team Ban 1"""
        df_b = self.df[self.df["result"] == "b"]
        return df_b["bb1"].value_counts().idxmax()

    def predict_rb1(self, bb1, global_used_champs):
        """Predict Red Team Ban 1"""
        return self._predict_ml("rb1", ["bb1"], "r", [bb1], global_used_champs=global_used_champs)

    def predict_bb2(self, bb1, rb1, global_used_champs):
        """Predict Blue Team Ban 2"""
        return self._predict_ml("bb2", ["bb1", "rb1"], "b", [bb1, rb1], global_used_champs=global_used_champs)

    def predict_rb2(self, bb1, rb1, bb2, global_used_champs):
        """Predict Red Team Ban 2"""
        return self._predict_ml("rb2", ["bb1", "rb1", "bb2"], "r", [bb1, rb1, bb2], global_used_champs=global_used_champs)

    def predict_bb3(self, bb1, rb1, bb2, rb2, global_used_champs):
        """Predict Blue Team Ban 3"""
        cols = ["bb1", "rb1", "bb2", "rb2"]
        feature_values = [bb1, rb1, bb2, rb2]
        return self._predict_ml("bb3", cols, "b", feature_values, global_used_champs=global_used_champs)

    def predict_rb3(self, bb1, rb1, bb2, rb2, bb3, global_used_champs):
        """Predict Red Team Ban 3"""
        cols = ["bb1", "rb1", "bb2", "rb2", "bb3"]
        feature_values = [bb1, rb1, bb2, rb2, bb3]
        return self._predict_ml("rb3", cols, "r", feature_values, global_used_champs=global_used_champs)

    # ===== PICK PHASE 1 =====
    def predict_bp1(self, bb1, rb1, bb2, rb2, bb3, rb3, blue_team_filled_roles, global_used_champs):
        """Predict Blue Team Pick 1"""
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3]
        return self._predict_ml("bp1", cols, "b", feature_values, team_filled_roles=blue_team_filled_roles, global_used_champs=global_used_champs)

    def predict_rp1(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, red_team_filled_roles, global_used_champs):
        """Predict Red Team Pick 1"""
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1]
        return self._predict_ml("rp1", cols, "r", feature_values, team_filled_roles=red_team_filled_roles, global_used_champs=global_used_champs)

    def predict_rp2(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, red_team_filled_roles, global_used_champs):
        """Predict Red Team Pick 2"""
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1]
        return self._predict_ml("rp2", cols, "r", feature_values, team_filled_roles=red_team_filled_roles, global_used_champs=global_used_champs)

    def predict_bp2(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, blue_team_filled_roles, global_used_champs):
        """Predict Blue Team Pick 2"""
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2]
        return self._predict_ml("bp2", cols, "b", feature_values, team_filled_roles=blue_team_filled_roles, global_used_champs=global_used_champs)

    def predict_bp3(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, blue_team_filled_roles, global_used_champs):
        """Predict Blue Team Pick 3"""
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2]
        return self._predict_ml("bp3", cols, "b", feature_values, team_filled_roles=blue_team_filled_roles, global_used_champs=global_used_champs)

    def predict_rp3(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, red_team_filled_roles, global_used_champs):
        """Predict Red Team Pick 3"""
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2,bp3]
        return self._predict_ml("rp3", cols, "r", feature_values, team_filled_roles=red_team_filled_roles, global_used_champs=global_used_champs)

    # ===== BAN PHASE 2 =====
    def predict_rb4(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, global_used_champs):
        """Predict Red Team Ban 4"""
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2,bp3,rp3]
        return self._predict_ml("rb4", cols, "r", feature_values, global_used_champs=global_used_champs)

    def predict_bb4(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, global_used_champs):
        """Predict Blue Team Ban 4"""
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2,bp3,rp3,rb4]
        return self._predict_ml("bb4", cols, "b", feature_values, global_used_champs=global_used_champs)

    def predict_rb5(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, global_used_champs):
        """Predict Red Team Ban 5"""
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2,bp3,rp3,rb4,bb4]
        return self._predict_ml("rb5", cols, "r", feature_values, global_used_champs=global_used_champs)

    def predict_bb5(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, rb5, global_used_champs):
        """Predict Blue Team Ban 5"""
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2,bp3,rp3,rb4,bb4,rb5]
        return self._predict_ml("bb5", cols, "b", feature_values, global_used_champs=global_used_champs)

    # ===== PICK PHASE 2 =====
    def predict_rp4(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, rb5, bb5, red_team_filled_roles, global_used_champs):
        """Predict Red Team Pick 4"""
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5","bb5"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2,bp3,rp3,rb4,bb4,rb5,bb5]
        return self._predict_ml("rp4", cols, "r", feature_values, team_filled_roles=red_team_filled_roles, global_used_champs=global_used_champs)

    def predict_bp4(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, rb5, bb5, rp4, blue_team_filled_roles, global_used_champs):
        """Predict Blue Team Pick 4"""
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5","bb5","rp4"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2,bp3,rp3,rb4,bb4,rb5,bb5,rp4]
        return self._predict_ml("bp4", cols, "b", feature_values, team_filled_roles=blue_team_filled_roles, global_used_champs=global_used_champs)

    def predict_bp5(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, rb5, bb5, rp4, bp4, blue_team_filled_roles, global_used_champs):
        """Predict Blue Team Pick 5"""
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5","bb5","rp4","bp4"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2,bp3,rp3,rb4,bb4,rb5,bb5,rp4,bp4]
        return self._predict_ml("bp5", cols, "b", feature_values, team_filled_roles=blue_team_filled_roles, global_used_champs=global_used_champs)

    def predict_rp5(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, rb5, bb5, rp4, bp4, bp5, red_team_filled_roles, global_used_champs):
        """Predict Red Team Pick 5"""
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5","bb5","rp4","bp4","bp5"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2,bp3,rp3,rb4,bb4,rb5,bb5,rp4,bp4,bp5]
        return self._predict_ml("rp5", cols, "r", feature_values, team_filled_roles=red_team_filled_roles, global_used_champs=global_used_champs)
