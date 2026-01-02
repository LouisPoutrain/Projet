"""
Evaluator module for DraftPredictor.
Handles evaluation of model predictions.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder


def evaluate_target(predictor, target_col: str, feature_cols: list) -> dict:
    """
    Optional quick evaluation for a single target.
    
    Returns:
        dict: {target, samples, accuracy, f1_macro}
    """
    df_f = predictor.df.dropna(subset=feature_cols + [target_col])
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
            X_enc, y, test_size=0.2, random_state=None, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_enc, y, test_size=0.2, random_state=None
        )

    model = LinearSVC(C=1.0, max_iter=5000, dual=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    return {"target": target_col, "samples": int(X.shape[0]), "accuracy": acc, "f1_macro": f1}


def evaluate_all_targets(predictor) -> list:
    """
    Evaluate all ban and pick targets using standard metrics (accuracy / F1-macro).
    
    Returns:
        list: List of evaluation results per target
    """
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
        res = evaluate_target(predictor, target, features)
        results.append(res)
        acc_str = "N/A" if res["accuracy"] is None else f"{res['accuracy']:.3f}"
        f1_str = "N/A" if res["f1_macro"] is None else f"{res['f1_macro']:.3f}"
        print(f"- {target}: samples={res['samples']} | acc={acc_str} | f1={f1_str}")
    print("===== End Evaluation =====\n")
    return results


def evaluate_with_intelligent_rewards(predictor, save_details=True, output_path="../processed_data/evaluation_details.csv") -> list:
    """
    Evaluate picks using intelligent rewards based on win rates vs opponents.
    Uses 20% test set (random split) for fair evaluation.
    Saves detailed predictions to CSV if save_details=True.
    
    Returns:
        list: List of evaluation results per pick target
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
    detailed_records = []  # Store all predictions for CSV export
    
    print("\n===== Intelligent Reward Evaluation (Win Rates) - 20% Test Set =====")
    
    for target_col, feature_cols in tqdm(configs, desc="Pick targets"):
        my_position, _ = pick_mapping[target_col]
        
        # Determine which opponent picks are available at this point
        if target_col.startswith("bp"):
            opp_pick_cols = [c for c in feature_cols if c.startswith("rp")]
        else:
            opp_pick_cols = [c for c in feature_cols if c.startswith("bp")]
        
        df_f = predictor.df.dropna(subset=feature_cols + [target_col] + opp_pick_cols)
        if df_f.empty:
            continue
        
        # Split into 80% train, 20% test (random split)
        try:
            df_train, df_test = train_test_split(
                df_f, test_size=0.2, random_state=None, stratify=df_f[target_col]
            )
        except ValueError:
            df_train, df_test = train_test_split(
                df_f, test_size=0.2, random_state=None
            )
        
        # Evaluate only on 20% test set
        rewards = []
        exact_matches = 0
        for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc=f"  {target_col}", leave=False):
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
                predicted = predictor._predict_ml(target_col, feature_cols, None, feature_values)
                is_exact = predicted == actual_champ
                if is_exact:
                    exact_matches += 1
                
                # Calculate reward with detailed breakdown
                if is_exact:
                    reward = 1.0
                    win_rates_detail = "exact_match"
                elif not opponent_picks:
                    reward = 0.0
                    win_rates_detail = "no_opponents"
                else:
                    # Calculate average win rate across all positions for each opponent
                    all_win_rates = []
                    win_rate_breakdown = []
                    
                    for opp_champ, opp_pos in opponent_picks:
                        # Find all matchup win rates regardless of position
                        matchup_rates = []
                        for key, wr in predictor.winrate_lookup.items():
                            champ, pos, opponent, opp_position = key
                            if champ == predicted and opponent == opp_champ:
                                matchup_rates.append(wr)
                        
                        # Average for this opponent across all positions
                        if matchup_rates:
                            avg_wr = np.mean(matchup_rates)
                            all_win_rates.append(avg_wr)
                            win_rate_breakdown.append(f"{opp_champ}(all_pos):{avg_wr:.3f}[n={len(matchup_rates)}]")
                        else:
                            win_rate_breakdown.append(f"{opp_champ}:N/A")
                    
                    if all_win_rates:
                        reward = np.mean(all_win_rates)
                        win_rates_detail = " | ".join(win_rate_breakdown) + f" | final_avg={reward:.3f}"
                    else:
                        reward = 0.0
                        win_rates_detail = "no_winrate_data"
                
                rewards.append(reward)
                
                # Store detailed record
                record = {
                    "game_index": idx,
                    "target": target_col,
                    "position": my_position,
                    "context": ";".join([f"{col}={val}" for col, val in zip(feature_cols, feature_values)]),
                    "predicted_champion": predicted,
                    "actual_champion": actual_champ,
                    "is_exact_match": is_exact,
                    "opponents": ";".join([f"{c}({p})" for c, p in opponent_picks]),
                    "reward_calculation": win_rates_detail,
                    "final_reward": reward
                }
                detailed_records.append(record)
                
            except Exception as e:
                rewards.append(0.0)
                # Store error record
                record = {
                    "game_index": idx,
                    "target": target_col,
                    "position": my_position,
                    "context": ";".join([f"{col}={val}" for col, val in zip(feature_cols, feature_values)]),
                    "predicted_champion": "ERROR",
                    "actual_champion": actual_champ,
                    "is_exact_match": False,
                    "opponents": ";".join([f"{c}({p})" for c, p in opponent_picks]),
                    "reward_calculation": f"error:{str(e)}",
                    "final_reward": 0.0
                }
                detailed_records.append(record)
        
        if rewards:
            mean_reward = np.mean(rewards)
            exact_acc = exact_matches / len(rewards)
            results.append({
                "target": target_col,
                "samples": len(rewards),
                "exact_accuracy": exact_acc,
                "mean_reward": mean_reward
            })
            print(f"- {target_col}: samples={len(rewards)} (20% test) | exact_acc={exact_acc:.3f} | reward={mean_reward:.3f}")
    
    # Save detailed records to CSV
    if save_details and detailed_records:
        df_details = pd.DataFrame(detailed_records)
        df_details.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n✓ Detailed evaluation saved to: {output_path}")
        print(f"  Total predictions: {len(detailed_records)}")
    
    print("===== End Intelligent Evaluation =====\n")
    return results
