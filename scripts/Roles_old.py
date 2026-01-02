"""
Main script for LoL esports draft prediction.
Uses modular components: predictor.py, evaluator.py, utils_roles.py
"""
import pandas as pd
from predictor import DraftPredictor
from evaluator import evaluate_all_targets, evaluate_with_intelligent_rewards
from utils_roles import (
    normalize_champion_name,
    assign_champion_role,
    get_user_champion_input,
    resolve_flexible_assignments,
    load_winrate_lookup
)

# Toggle to run evaluation and exit
EVAL_MODE = True

# Toggle to retrain models from scratch (instead of loading from cache)
RETRAIN_MODE = False


class DraftPredictor:
    def __init__(self, df: pd.DataFrame, model_dir="../models/saved_models", reward_type="standard"):
        self.df = df
        self._model_cache = {}  # Cache for models and encoders
        self.reward_type = reward_type  # "standard" or "intelligent_reward"
        self.retrain_mode = RETRAIN_MODE  # Force retraining from scratch
        
        # Build model directory structure based on reward type
        # models/standard/bans/ and models/standard/picks/
        # models/intelligent_reward/bans/ and models/intelligent_reward/picks/
        base_model_dir = os.path.dirname(model_dir)
        self.model_dir_base = os.path.join(base_model_dir, reward_type)
        self.model_dir = self.model_dir_base  # For compatibility
        
        # Create subdirectories
        for subdir in ["bans", "picks"]:
            path = os.path.join(self.model_dir_base, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
        
        mode_str = "(RETRAIN MODE)" if self.retrain_mode else ""
        print(f"Using model directory: {self.model_dir_base} {mode_str}")

        # --- New code for subtask: Load champion roles ---
        self.champion_roles_map = {}
        self.champion_lookup = {}  # normalized name -> canonical name
        try:
            roles_df = pd.read_csv('../processed_data/champions_by_position.csv')
            for index, row in roles_df.iterrows():
                position = row['Position']
                champions_str = row['Champions']
                champions_list = [c.strip() for c in champions_str.split(',')]
                for champion in champions_list:
                    if champion not in self.champion_roles_map:
                        self.champion_roles_map[champion] = []
                    self.champion_roles_map[champion].append(position)
                    norm = normalize_champion_name(champion)
                    # Keep first seen canonical name for a normalized key
                    if norm not in self.champion_lookup:
                        self.champion_lookup[norm] = champion
            print("Champion roles map loaded successfully.")
            # Print first few entries for verification
            print("First 5 entries of champion_roles_map:")
            for i, (champion, roles) in enumerate(self.champion_roles_map.items()):
                if i >= 5: break
                print(f"  {champion}: {roles}")

        except FileNotFoundError:
            print("Error: 'champions_by_position.csv' not found. Champion roles will not be available.")
        except Exception as e:
            print(f"An error occurred while loading champion roles: {e}")
        # --- End new code ---

    # Optional quick evaluation for a single target
    def evaluate_target(self, target_col: str, feature_cols: list) -> dict:
        df_f = self.df.dropna(subset=feature_cols + [target_col])
        if df_f.empty:
            return {"target": target_col, "samples": 0, "accuracy": None, "f1_macro": None}

        X = df_f[feature_cols]
        y = df_f[target_col]

        # Keep classes appearing at least twice
        freq = y.value_counts()
        valid = freq[freq >= 2].index
        mask = y.isin(valid)
        X = X[mask]
        y = y[mask]

        if X.shape[0] < 5 or y.nunique() < 2:
            return {"target": target_col, "samples": int(X.shape[0]), "accuracy": None, "f1_macro": None}

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_enc = encoder.fit_transform(X)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_enc, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_enc, y, test_size=0.2, random_state=42
            )

        model = LinearSVC(C=1.0, max_iter=5000, dual=True)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        return {"target": target_col, "samples": int(X.shape[0]), "accuracy": acc, "f1_macro": f1}

    def evaluate_all_targets(self) -> list:
        configs = [
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

        results = []
        print("\n===== Evaluation Mode: accuracy / F1-macro =====")
        for target, features in configs:
            res = self.evaluate_target(target, features)
            results.append(res)
            acc_str = "N/A" if res["accuracy"] is None else f"{res['accuracy']:.3f}"
            f1_str = "N/A" if res["f1_macro"] is None else f"{res['f1_macro']:.3f}"
            print(f"- {target}: samples={res['samples']} | acc={acc_str} | f1={f1_str}")
        print("===== End Evaluation =====\n")
        return results

    def compute_intelligent_reward(self, predicted_champ, actual_champ, my_position, opponent_picks):
        """
        Compute intelligent reward:
        - If exact match: 1.0
        - Otherwise: average win rate vs all opponent picks
        
        Args:
            predicted_champ: Champion predicted by model
            actual_champ: Actual champion picked
            my_position: Position of this pick
            opponent_picks: List of (opponent_champ, opponent_pos) already picked
        
        Returns:
            float: Reward score (0.0-1.0)
        """
        # Exact match: perfect reward
        if predicted_champ == actual_champ:
            return 1.0
        
        # Different champion: compute average win rate vs opponents
        if not opponent_picks:
            return 0.0
        
        win_rates = []
        for opp_champ, opp_pos in opponent_picks:
            key = (predicted_champ, my_position, opp_champ, opp_pos)
            if key in winrate_lookup:
                win_rates.append(winrate_lookup[key])
        
        # Return average win rate, or 0 if no data
        if win_rates:
            return np.mean(win_rates)
        return 0.0

    def evaluate_with_intelligent_rewards(self) -> list:
        """
        Evaluate picks using intelligent rewards based on win rates vs opponents.
        """
        # Map pick columns to positions
        pick_mapping = {
            "bp1": ("top", "rp1"),
            "bp2": ("jng", "rp2"),
            "bp3": ("mid", "rp3"),
            "bp4": ("bot", "rp4"),
            "bp5": ("sup", "rp5"),
            "rp1": ("top", "bp1"),
            "rp2": ("jng", "bp2"),
            "rp3": ("mid", "bp3"),
            "rp4": ("bot", "bp4"),
            "rp5": ("sup", "bp5"),
        }
        
        configs = [
            ("bp1", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3"]),
            ("rp1", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1"]),
            ("rp2", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1"]),
            ("bp2", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2"]),
            ("bp3", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2"]),
            ("rp3", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3"]),
            ("rp4", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3", "rb4", "bb4", "rb5", "bb5"]),
            ("bp4", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3", "rb4", "bb4", "rb5", "bb5", "rp4"]),
            ("bp5", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3", "rb4", "bb4", "rb5", "bb5", "rp4", "bp4"]),
            ("rp5", ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3", "bp1", "rp1", "rp2", "bp2", "bp3", "rp3", "rb4", "bb4", "rb5", "bb5", "rp4", "bp4", "bp5"]),
        ]
        
        results = []
        print("\n===== Intelligent Reward Evaluation (Win Rates) =====")
        
        for target_col, feature_cols in tqdm(configs, desc="Pick targets"):
            my_position, _ = pick_mapping[target_col]
            
            # Determine which opponent picks are available at this point
            if target_col.startswith("bp"):
                # Blue team pick: opponents are red picks before this
                opp_pick_cols = [c for c in feature_cols if c.startswith("rp")]
            else:
                # Red team pick: opponents are blue picks before this
                opp_pick_cols = [c for c in feature_cols if c.startswith("bp")]
            
            df_f = self.df.dropna(subset=feature_cols + [target_col] + opp_pick_cols)
            if df_f.empty:
                continue
            
            rewards = []
            exact_matches = 0
            for _, row in tqdm(df_f.iterrows(), total=len(df_f), desc=f"  {target_col}", leave=False):
                feature_values = row[feature_cols].tolist()
                actual_champ = row[target_col]
                
                # Collect opponent picks
                opponent_picks = []
                for opp_col in opp_pick_cols:
                    opp_champ = row[opp_col]
                    if pd.notna(opp_champ):
                        opp_pos, _ = pick_mapping[opp_col]
                        opponent_picks.append((opp_champ, opp_pos))
                
                # Predict using model
                try:
                    predicted = self._predict_ml(target_col, feature_cols, None, feature_values)
                    if predicted == actual_champ:
                        exact_matches += 1
                    reward = self.compute_intelligent_reward(predicted, actual_champ, my_position, opponent_picks)
                    rewards.append(reward)
                except:
                    rewards.append(0.0)
            
            if rewards:
                mean_reward = np.mean(rewards)
                exact_acc = exact_matches / len(rewards)
                results.append({
                    "target": target_col,
                    "samples": len(rewards),
                    "exact_accuracy": exact_acc,
                    "mean_reward": mean_reward
                })
                print(f"- {target_col}: samples={len(rewards)} | exact_acc={exact_acc:.3f} | reward={mean_reward:.3f}")
        
        print("===== End Intelligent Evaluation =====\n")
        return results

    # Modified _predict_ml signature: added global_used_champs for full draft awareness
    def _predict_ml(self, target_col, feature_cols, result_filter,
                     feature_values, team_filled_roles=None, global_used_champs=None):
        # Cache key for model and encoder (independent of roles/global_used_champs)
        model_encoder_cache_key = f"model_encoder_{target_col}_{result_filter}_{'_'.join(feature_cols)}"
        
        # Determine subdirectory: bans or picks
        subdir = "bans" if target_col[1:] in ["b1", "b2", "b3", "b4", "b5"] else "picks"
        model_file = os.path.join(self.model_dir_base, subdir, f"{model_encoder_cache_key}.pkl")

        # Load cached model if available (from memory or disk) - unless RETRAIN_MODE is on
        if not self.retrain_mode and model_encoder_cache_key in self._model_cache:
            model, encoder = self._model_cache[model_encoder_cache_key]
        elif not self.retrain_mode and os.path.exists(model_file):
            # Charger depuis le fichier pickle
            print(f"Chargement du modèle depuis : {model_file}")
            with open(model_file, 'rb') as f:
                model, encoder = pickle.load(f)
            # Mettre en cache mémoire
            self._model_cache[model_encoder_cache_key] = (model, encoder)
        else:
            # Entraîner un nouveau modèle (ou ré-entraîner si RETRAIN_MODE)
            retrain_str = "(RETRAIN)" if self.retrain_mode else ""
            print(f"Entraînement d'un nouveau modèle pour : {target_col} {retrain_str}")
            # Filter by result
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

            # ===== LinearSVC classifier =====
            model = LinearSVC(
                C=1.0,
                max_iter=5000,
                dual=True
            )
            model.fit(X_encoded, y)

            # Cache model and encoder in memory
            self._model_cache[model_encoder_cache_key] = (model, encoder)
            
            # Sauvegarder sur disque avec pickle
            with open(model_file, 'wb') as f:
                pickle.dump((model, encoder), f)
            print(f"Modèle sauvegardé dans : {model_file}")


        # Retrieve cached model + encoder
        model, encoder = self._model_cache[model_encoder_cache_key]

        # Encode input features
        # Convert feature_values to a DataFrame with appropriate column names to avoid UserWarning
        X_input_df = pd.DataFrame([feature_values], columns=feature_cols)
        X_input_encoded = encoder.transform(X_input_df)


        # ===== LinearSVC → decision_function → pseudo-probabilities =====
        scores = model.decision_function(X_input_encoded)[0]

        # Handle binary classification: convert scalar score to 2-class scores
        if isinstance(scores, float) or np.ndim(scores) == 0:
            scores = np.array([-scores, scores])

        probas = softmax([scores])[0]
        classes = model.classes_

        # Remove already-picked champs (bans/picks from both teams) - this is global used list
        # Use global_used_champs here, which includes ALL bans and picks so far.
        used = set(global_used_champs) if global_used_champs else set()
        mask = np.array([c not in used for c in classes])
        probas = probas * mask

        # --- New code for role-aware penalization ---
        # Only apply role penalization if team_filled_roles was explicitly passed (i.e., for picks)
        if team_filled_roles is not None:
            all_roles = {'top', 'jng', 'mid', 'bot', 'sup'}
            unfilled_roles_for_current_team = all_roles - team_filled_roles

            # Only apply role penalization if the team still needs to fill roles
            if len(unfilled_roles_for_current_team) > 0:
                for i, champion in enumerate(classes):
                    if probas[i] > 0: # Only consider champions that are not already removed by 'used' mask
                        champion_possible_roles = self.champion_roles_map.get(champion, [])

                        # Check if this champion can fill any of the currently unfilled roles
                        can_fill_an_unfilled_role = any(role in unfilled_roles_for_current_team for role in champion_possible_roles)

                        if not can_fill_an_unfilled_role:
                            # This champion cannot fill any of the currently unfilled roles.
                            # This means picking it would either double up on an already filled role,
                            # or it has no known roles and we still need roles filled.
                            if len(champion_possible_roles) > 0: # Champion has known roles, but none are needed
                                probas[i] *= 0.1 # Moderate penalty for filling an already taken role while others are open
                            else: # Champion has no known roles, and we still need roles filled
                                probas[i] *= 0.05 # Stronger penalty for picking an unknown-role champ when roles are missing
        # --- End new code for role-aware penalization ---

        # Fallback if everything is masked (or penalized to zero)
        if probas.sum() == 0:
            remaining = [c for c in classes if c not in used]
            return remaining[0] if remaining else classes[0]

        return classes[np.argmax(probas)]

    # --- Draft prediction functions ---
    def predict_bb1(self):
        df_b = self.df[self.df["result"] == "b"]
        return df_b["bb1"].value_counts().idxmax()

    # Ban predictions don't directly fill roles, but need to pass global_used_champs
    def predict_rb1(self, bb1, global_used_champs):
        return self._predict_ml("rb1", ["bb1"], "r", [bb1], global_used_champs=global_used_champs)

    def predict_bb2(self, bb1, rb1, global_used_champs):
        return self._predict_ml("bb2", ["bb1", "rb1"], "b", [bb1, rb1], global_used_champs=global_used_champs)

    def predict_rb2(self, bb1, rb1, bb2, global_used_champs):
        return self._predict_ml("rb2", ["bb1", "rb1", "bb2"], "r", [bb1, rb1, bb2], global_used_champs=global_used_champs)

    def predict_bb3(self, bb1, rb1, bb2, rb2, global_used_champs):
        cols = ["bb1", "rb1", "bb2", "rb2"]
        feature_values = [bb1, rb1, bb2, rb2]
        return self._predict_ml("bb3", cols, "b", feature_values, global_used_champs=global_used_champs)

    def predict_rb3(self, bb1, rb1, bb2, rb2, bb3, global_used_champs):
        cols = ["bb1", "rb1", "bb2", "rb2", "bb3"]
        feature_values = [bb1, rb1, bb2, rb2, bb3]
        return self._predict_ml("rb3", cols, "r", feature_values, global_used_champs=global_used_champs)

    # Pick prediction methods now accept team_filled_roles and global_used_champs
    def predict_bp1(self, bb1, rb1, bb2, rb2, bb3, rb3, blue_team_filled_roles, global_used_champs):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3]
        return self._predict_ml("bp1", cols, "b", feature_values, team_filled_roles=blue_team_filled_roles, global_used_champs=global_used_champs)

    def predict_rp1(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, red_team_filled_roles, global_used_champs):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1]
        return self._predict_ml("rp1", cols, "r", feature_values, team_filled_roles=red_team_filled_roles, global_used_champs=global_used_champs)

    def predict_rp2(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, red_team_filled_roles, global_used_champs):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1]
        return self._predict_ml("rp2", cols, "r", feature_values, team_filled_roles=red_team_filled_roles, global_used_champs=global_used_champs)

    def predict_bp2(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, blue_team_filled_roles, global_used_champs):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2]
        return self._predict_ml("bp2", cols, "b", feature_values, team_filled_roles=blue_team_filled_roles, global_used_champs=global_used_champs)

    def predict_bp3(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, blue_team_filled_roles, global_used_champs):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2]
        return self._predict_ml("bp3", cols, "b", feature_values, team_filled_roles=blue_team_filled_roles, global_used_champs=global_used_champs)

    def predict_rp3(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, red_team_filled_roles, global_used_champs):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2,bp3]
        return self._predict_ml("rp3", cols, "r", feature_values, team_filled_roles=red_team_filled_roles, global_used_champs=global_used_champs)

    # Ban phase 2 (rb4, bb4, rb5, bb5) also don't fill roles, but they need global_used_champs for the `used` set.
    def predict_rb4(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, global_used_champs):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2,bp3,rp3]
        return self._predict_ml("rb4", cols, "r", feature_values, global_used_champs=global_used_champs)

    def predict_bb4(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, global_used_champs):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2,bp3,rp3,rb4]
        return self._predict_ml("bb4", cols, "b", feature_values, global_used_champs=global_used_champs)

    def predict_rb5(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, global_used_champs):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2,bp3,rp3,rb4,bb4]
        return self._predict_ml("rb5", cols, "r", feature_values, global_used_champs=global_used_champs)

    def predict_bb5(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, rb5, global_used_champs):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2,bp3,rp3,rb4,bb4,rb5]
        return self._predict_ml("bb5", cols, "b", feature_values, global_used_champs=global_used_champs)


    # Final pick phase
    def predict_rp4(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, rb5, bb5, red_team_filled_roles, global_used_champs):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5","bb5"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2,bp3,rp3,rb4,bb4,rb5,bb5]
        return self._predict_ml("rp4", cols, "r", feature_values, team_filled_roles=red_team_filled_roles, global_used_champs=global_used_champs)

    def predict_bp4(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, rb5, bb5, rp4, blue_team_filled_roles, global_used_champs):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5","bb5","rp4"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2,bp3,rp3,rb4,bb4,rb5,bb5,rp4]
        return self._predict_ml("bp4", cols, "b", feature_values, team_filled_roles=blue_team_filled_roles, global_used_champs=global_used_champs)

    def predict_bp5(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, rb5, bb5, rp4, bp4, blue_team_filled_roles, global_used_champs):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5","bb5","rp4","bp4"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2,bp3,rp3,rb4,bb4,rb5,bb5,rp4,bp4]
        return self._predict_ml("bp5", cols, "b", feature_values, team_filled_roles=blue_team_filled_roles, global_used_champs=global_used_champs)

    def predict_rp5(self, bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, rb5, bb5, rp4, bp4, bp5, red_team_filled_roles, global_used_champs):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5","bb5","rp4","bp4","bp5"]
        feature_values = [bb1,rb1,bb2,rb2,bb3,rb3,bp1,rp1,rp2,bp2,bp3,rp3,rb4,bb4,rb5,bb5,rp4,bp4,bp5]
        return self._predict_ml("rp5", cols, "r", feature_values, team_filled_roles=red_team_filled_roles, global_used_champs=global_used_champs)


# Helper function to assign a role to a picked champion
def assign_champion_role(champion, team_roles_dict, team_filled_roles_set, champion_roles_map, flexible_champions_list):
    all_known_roles = {'top', 'jng', 'mid', 'bot', 'sup'}

    possible_roles_for_champion = champion_roles_map.get(champion, [])

    # If champion has no known roles, assign UNKNOWN
    if not possible_roles_for_champion:
        team_roles_dict[champion] = "UNKNOWN"
        return "UNKNOWN"

    # Calculate roles the champion can play that are not yet filled by other champions
    available_roles_for_champ_slot = set(possible_roles_for_champion) - team_filled_roles_set

    if len(available_roles_for_champ_slot) == 1:
        # Directly assign the single available role
        assigned_role = list(available_roles_for_champ_slot)[0]
        team_filled_roles_set.add(assigned_role)
        team_roles_dict[champion] = assigned_role
        return assigned_role
    elif len(available_roles_for_champ_slot) > 1:
        # This is a flexible pick, add to the flexible list and assign a provisional role
        flexible_champions_list.append((champion, available_roles_for_champ_slot))
        provisional_role = "PROV-" + list(available_roles_for_champ_slot)[0] # Assign first as provisional
        team_roles_dict[champion] = provisional_role
        # DO NOT add to team_filled_roles_set yet, as it's provisional
        return provisional_role
    else: # len(available_roles_for_champ_slot) == 0
        # All roles the champion could play are already definitively assigned.
        # Assign the first role from its possible roles and mark the slot as filled (could be a double pick).
        assigned_role = possible_roles_for_champion[0]
        team_filled_roles_set.add(assigned_role) # Acknowledge the pick, even if it's a duplicate role
        team_roles_dict[champion] = assigned_role
        return assigned_role


def get_user_champion_input(prompt_text, current_team_filled_roles, global_used_champs, champion_roles_map):
    while True:
        user_input = input(prompt_text).strip()
        champion_name = user_input.capitalize() # Assume champion names are capitalized

        if champion_name not in champion_roles_map:
            print(f"Champion '{champion_name}' not found in the list of known champions. Please try again.")
            continue

        if champion_name in global_used_champs:
            print(f"Champion '{champion_name}' has already been banned or picked. Please choose another champion.")
            continue

        # Optional: Warn user if picking a champion that cannot fill an open role
        all_roles = {'top', 'jng', 'mid', 'bot', 'sup'}
        unfilled_roles = all_roles - current_team_filled_roles
        possible_roles_for_champion = champion_roles_map.get(champion_name, [])

        can_fill_an_unfilled_role = any(role in unfilled_roles for role in possible_roles_for_champion)

        if len(unfilled_roles) > 0 and not can_fill_an_unfilled_role:
            print(f"Warning: '{champion_name}' cannot fill any of your remaining unfilled roles ({', '.join(unfilled_roles)}). It might be a suboptimal pick. Type 'yes' to confirm or any other key to choose another: ")
            confirm = input().strip().lower()
            if confirm != 'yes':
                continue

        return champion_name


# New function to resolve flexible assignments
def resolve_flexible_assignments(team_assigned_roles, team_flexible_champions, team_filled_roles_set):
    all_roles = {'top', 'jng', 'mid', 'bot', 'sup'}
    resolved_count = 0

    # Continue as long as there are flexible champions and progress is being made
    while True:
        new_assignments_made_in_iteration = False
        champions_to_remove = []

        # Iterate through flexible champions to find unique assignments
        for i, (champion, possible_roles) in enumerate(team_flexible_champions):
            current_unfilled_roles = all_roles - team_filled_roles_set
            # Roles this champion can play that are still unfilled by other DEFINITIVE assignments
            available_roles_for_this_champ = possible_roles.intersection(current_unfilled_roles)

            if len(available_roles_for_this_champ) == 1:
                # Unique assignment found for this flexible champion
                assigned_role = list(available_roles_for_this_champ)[0]
                team_assigned_roles[champion] = assigned_role
                team_filled_roles_set.add(assigned_role)
                champions_to_remove.append(i)
                new_assignments_made_in_iteration = True
                resolved_count += 1

        # Remove champions that were just assigned a definitive role
        # Iterate backwards to avoid index issues
        for index in sorted(champions_to_remove, reverse=True):
            team_flexible_champions.pop(index)

        if not new_assignments_made_in_iteration and team_flexible_champions: # No new unique assignments were made in this iteration, but flexible champions remain
            break # Exit loop, no more unique assignments possible via this iterative method
        elif not team_flexible_champions:
            break # All flexible champions resolved

    # Handle any remaining flexible champions (e.g., multiple possible roles, or all their roles are already filled)
    for champion, possible_roles in team_flexible_champions:
        # Remove the 'PROV-' prefix from the currently assigned role
        current_provisional_role = team_assigned_roles[champion]
        if current_provisional_role.startswith("PROV-"):
            del team_assigned_roles[champion]

        # Find the first possible role that isn't already definitively filled
        assigned_role = None
        for role in possible_roles:
            if role not in team_filled_roles_set:
                assigned_role = role
                break

        if assigned_role: # Found an unfilled role for this champion
            team_assigned_roles[champion] = assigned_role
            team_filled_roles_set.add(assigned_role)
        else: # All possible roles for this champion are already filled (possibly by other flexible champs or definitive assignments)
            # Assign the first of its possible roles and mark it as filled (could be a double role, but best effort)
            assigned_role = list(possible_roles)[0]
            team_assigned_roles[champion] = assigned_role
            team_filled_roles_set.add(assigned_role)

    print(f"Resolved {resolved_count} flexible champion roles.")


# ======================
#     EXECUTION (Modified for interactive role-aware drafting)
# ======================

df = pd.read_csv("../processed_data/csv_games_fusionnes.csv")

# Use "intelligent_reward" for EVAL_MODE, "standard" for normal drafting
reward_type = "intelligent_reward" if EVAL_MODE else "standard"
predictor = DraftPredictor(df, reward_type=reward_type)

# Run evaluation and exit if enabled
if EVAL_MODE:
    print("Running standard evaluation...")
    predictor.evaluate_all_targets()
    print("\nRunning intelligent reward evaluation...")
    predictor.evaluate_with_intelligent_rewards()
    raise SystemExit(0)

# Initialize lists to keep track of picked champions and filled roles for each team
blue_team_bans = []
blue_team_picks = []
blue_team_assigned_roles = {} # Dictionary to store champion -> assigned role
blue_team_filled_roles = set() # Set of roles that have been filled
blue_team_flexible_champions = [] # New: To store flexible picks for blue team

red_team_bans = []
red_team_picks = []
red_team_assigned_roles = {} # Dictionary to store champion -> assigned role
red_team_filled_roles = set() # Set of roles that have been filled
red_team_flexible_champions = [] # New: To store flexible picks for red team

all_bans_and_picks = [] # This list will contain ALL champions banned or picked

SELF_PLAY = False

mode_label = "Self-Play (Model vs Model)" if SELF_PLAY else "Draft Interactive"
print(f"\n===== Démarrage de la {mode_label} =====\n")

# Ban Phase 1
if SELF_PLAY:
    bb1 = predictor.predict_bb1()
else:
    bb1 = get_user_champion_input("Blue Team Ban 1 (bb1): ", blue_team_filled_roles, all_bans_and_picks, predictor.champion_roles_map)
blue_team_bans.append(bb1)
all_bans_and_picks.append(bb1)
print(f"bb1 = {bb1}")

rb1 = predictor.predict_rb1(bb1, global_used_champs=all_bans_and_picks)
red_team_bans.append(rb1)
all_bans_and_picks.append(rb1)
print(f"rb1 = {rb1}")

if SELF_PLAY:
    bb2 = predictor.predict_bb2(bb1, rb1, global_used_champs=all_bans_and_picks)
else:
    bb2 = get_user_champion_input("Blue Team Ban 2 (bb2): ", blue_team_filled_roles, all_bans_and_picks, predictor.champion_roles_map)
blue_team_bans.append(bb2)
all_bans_and_picks.append(bb2)
print(f"bb2 = {bb2}")

rb2 = predictor.predict_rb2(bb1, rb1, bb2, global_used_champs=all_bans_and_picks)
red_team_bans.append(rb2)
all_bans_and_picks.append(rb2)
print(f"rb2 = {rb2}")

if SELF_PLAY:
    bb3 = predictor.predict_bb3(bb1, rb1, bb2, rb2, global_used_champs=all_bans_and_picks)
else:
    bb3 = get_user_champion_input("Blue Team Ban 3 (bb3): ", blue_team_filled_roles, all_bans_and_picks, predictor.champion_roles_map)
blue_team_bans.append(bb3)
all_bans_and_picks.append(bb3)
print(f"bb3 = {bb3}")

rb3 = predictor.predict_rb3(bb1, rb1, bb2, rb2, bb3, global_used_champs=all_bans_and_picks)
red_team_bans.append(rb3)
all_bans_and_picks.append(rb3)
print(f"rb3 = {rb3}")

# Pick Phase 1
if SELF_PLAY:
    bp1 = predictor.predict_bp1(bb1, rb1, bb2, rb2, bb3, rb3,
                                blue_team_filled_roles=blue_team_filled_roles,
                                global_used_champs=all_bans_and_picks)
else:
    bp1 = get_user_champion_input("Blue Team Pick 1 (bp1): ", blue_team_filled_roles, all_bans_and_picks, predictor.champion_roles_map)
blue_team_picks.append(bp1)
assigned_role_bp1 = assign_champion_role(bp1, blue_team_assigned_roles, blue_team_filled_roles, predictor.champion_roles_map, blue_team_flexible_champions)
all_bans_and_picks.append(bp1)
print(f"bp1 = {bp1} ({assigned_role_bp1})")

rp1 = predictor.predict_rp1(bb1, rb1, bb2, rb2, bb3, rb3, bp1,
                            red_team_filled_roles=red_team_filled_roles,
                            global_used_champs=all_bans_and_picks)
red_team_picks.append(rp1)
assigned_role_rp1 = assign_champion_role(rp1, red_team_assigned_roles, red_team_filled_roles, predictor.champion_roles_map, red_team_flexible_champions)
all_bans_and_picks.append(rp1)
print(f"rp1 = {rp1} ({assigned_role_rp1})")


rp2 = predictor.predict_rp2(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1,
                            red_team_filled_roles=red_team_filled_roles,
                            global_used_champs=all_bans_and_picks)
red_team_picks.append(rp2)
assigned_role_rp2 = assign_champion_role(rp2, red_team_assigned_roles, red_team_filled_roles, predictor.champion_roles_map, red_team_flexible_champions)
all_bans_and_picks.append(rp2)
print(f"rp2 = {rp2} ({assigned_role_rp2})")

if SELF_PLAY:
    bp2 = predictor.predict_bp2(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2,
                                blue_team_filled_roles=blue_team_filled_roles,
                                global_used_champs=all_bans_and_picks)
else:
    bp2 = get_user_champion_input("Blue Team Pick 2 (bp2): ", blue_team_filled_roles, all_bans_and_picks, predictor.champion_roles_map)
blue_team_picks.append(bp2)
assigned_role_bp2 = assign_champion_role(bp2, blue_team_assigned_roles, blue_team_filled_roles, predictor.champion_roles_map, blue_team_flexible_champions)
all_bans_and_picks.append(bp2)
print(f"bp2 = {bp2} ({assigned_role_bp2})")


if SELF_PLAY:
    bp3 = predictor.predict_bp3(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2,
                                blue_team_filled_roles=blue_team_filled_roles,
                                global_used_champs=all_bans_and_picks)
else:
    bp3 = get_user_champion_input("Blue Team Pick 3 (bp3): ", blue_team_filled_roles, all_bans_and_picks, predictor.champion_roles_map)
blue_team_picks.append(bp3)
assigned_role_bp3 = assign_champion_role(bp3, blue_team_assigned_roles, blue_team_filled_roles, predictor.champion_roles_map, blue_team_flexible_champions)
all_bans_and_picks.append(bp3)
print(f"bp3 = {bp3} ({assigned_role_bp3})")

rp3 = predictor.predict_rp3(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3,
                            red_team_filled_roles=red_team_filled_roles,
                            global_used_champs=all_bans_and_picks)
red_team_picks.append(rp3)
assigned_role_rp3 = assign_champion_role(rp3, red_team_assigned_roles, red_team_filled_roles, predictor.champion_roles_map, red_team_flexible_champions)
all_bans_and_picks.append(rp3)
print(f"rp3 = {rp3} ({assigned_role_rp3})")

# Ban Phase 2
rb4 = predictor.predict_rb4(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3,
                            global_used_champs=all_bans_and_picks)
red_team_bans.append(rb4)
all_bans_and_picks.append(rb4)
print(f"rb4 = {rb4}")

if SELF_PLAY:
    bb4 = predictor.predict_bb4(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3,
                                rb4, global_used_champs=all_bans_and_picks)
else:
    bb4 = get_user_champion_input("Blue Team Ban 4 (bb4): ", blue_team_filled_roles, all_bans_and_picks, predictor.champion_roles_map)
blue_team_bans.append(bb4)
all_bans_and_picks.append(bb4)
print(f"bb4 = {bb4}")

rb5 = predictor.predict_rb5(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4,
                            global_used_champs=all_bans_and_picks)
red_team_bans.append(rb5)
all_bans_and_picks.append(rb5)
print(f"rb5 = {rb5}")

if SELF_PLAY:
    bb5 = predictor.predict_bb5(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3,
                                rb4, bb4, rb5, global_used_champs=all_bans_and_picks)
else:
    bb5 = get_user_champion_input("Blue Team Ban 5 (bb5): ", blue_team_filled_roles, all_bans_and_picks, predictor.champion_roles_map)
blue_team_bans.append(bb5)
all_bans_and_picks.append(bb5)
print(f"bb5 = {bb5}")


# Pick Phase 2
rp4 = predictor.predict_rp4(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, rb5, bb5,
                            red_team_filled_roles=red_team_filled_roles,
                            global_used_champs=all_bans_and_picks)
red_team_picks.append(rp4)
assigned_role_rp4 = assign_champion_role(rp4, red_team_assigned_roles, red_team_filled_roles, predictor.champion_roles_map, red_team_flexible_champions)
all_bans_and_picks.append(rp4)
print(f"rp4 = {rp4} ({assigned_role_rp4})")

if SELF_PLAY:
    bp4 = predictor.predict_bp4(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3,
                                rb4, bb4, rb5, bb5, rp4,
                                blue_team_filled_roles=blue_team_filled_roles,
                                global_used_champs=all_bans_and_picks)
else:
    bp4 = get_user_champion_input("Blue Team Pick 4 (bp4): ", blue_team_filled_roles, all_bans_and_picks, predictor.champion_roles_map)
blue_team_picks.append(bp4)
assigned_role_bp4 = assign_champion_role(bp4, blue_team_assigned_roles, blue_team_filled_roles, predictor.champion_roles_map, blue_team_flexible_champions)
all_bans_and_picks.append(bp4)
print(f"bp4 = {bp4} ({assigned_role_bp4})")

if SELF_PLAY:
    bp5 = predictor.predict_bp5(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3,
                                rb4, bb4, rb5, bb5, rp4, bp4,
                                blue_team_filled_roles=blue_team_filled_roles,
                                global_used_champs=all_bans_and_picks)
else:
    bp5 = get_user_champion_input("Blue Team Pick 5 (bp5): ", blue_team_filled_roles, all_bans_and_picks, predictor.champion_roles_map)
blue_team_picks.append(bp5)
assigned_role_bp5 = assign_champion_role(bp5, blue_team_assigned_roles, blue_team_filled_roles, predictor.champion_roles_map, blue_team_flexible_champions)
all_bans_and_picks.append(bp5)
print(f"bp5 = {bp5} ({assigned_role_bp5})")

rp5 = predictor.predict_rp5(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, rb5, bb5, rp4, bp4, bp5,
                            red_team_filled_roles=red_team_filled_roles,
                            global_used_champs=all_bans_and_picks)
red_team_picks.append(rp5)
assigned_role_rp5 = assign_champion_role(rp5, red_team_assigned_roles, red_team_filled_roles, predictor.champion_roles_map, red_team_flexible_champions)
all_bans_and_picks.append(rp5)
print(f"rp5 = {rp5} ({assigned_role_rp5})")


# Resolve flexible assignments after all picks are made for each team
print("\nResolving flexible champion roles for Blue Team...")
resolve_flexible_assignments(blue_team_assigned_roles, blue_team_flexible_champions, blue_team_filled_roles)
print("Resolving flexible champion roles for Red Team...")
resolve_flexible_assignments(red_team_assigned_roles, red_team_flexible_champions, red_team_filled_roles)


# ======================
#     FINAL OUTPUT
# ======================

print("\n===== DRAFT PRÉDITE =====\n")

print(f"Blue Team Bans: {tuple(blue_team_bans)}")
print(f"Blue Team Picks: {tuple(blue_team_picks)}")
print(f"Blue Team Assigned Roles: {blue_team_assigned_roles}")
print(f"Blue Team Flexible Champions (Champion, Potential Roles): {blue_team_flexible_champions}") # Should be empty after resolution
print(f"Red Team Bans: {tuple(red_team_bans)}")
print(f"Red Team Picks: {tuple(red_team_picks)}")
print(f"Red Team Assigned Roles: {red_team_assigned_roles}")
print(f"Red Team Flexible Champions (Champion, Potential Roles): {red_team_flexible_champions}") # Should be empty after resolution

print("\n===== FIN ====")
