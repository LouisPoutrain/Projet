"""
Main script for LoL esports draft prediction.
Uses modular components: predictor.py, evaluator.py, utils_roles.py
"""

import warnings

# Masque uniquement les warnings de version incohérente lors du chargement de modèles sklearn
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    # Fallback si la classe n'existe pas / import impossible
    warnings.filterwarnings(
        "ignore",
        message=r"Trying to unpickle estimator .* from version .* when using version .*",
    )
from pathlib import Path
import pandas as pd
from predictor import DraftPredictor
from evaluator import evaluate_all_targets, evaluate_with_intelligent_rewards
from utils_roles import (
    normalize_champion_name,
    assign_champion_role,
    get_user_champion_input,
    resolve_flexible_assignments,
)

# Toggle to run evaluation and exit
EVAL_MODE = False

# Toggle evaluation types (only active if EVAL_MODE = True)
EVAL_STANDARD = True     # Standard accuracy / F1-macro metrics
EVAL_INTELLIGENT = True   # Win rate-based intelligent rewards

# Toggle self-play mode (model vs model) for the interactive draft section
SELF_PLAY = False

# Directory containing improved bundles (intelligent-reward retrain)
IMPROVED_MODELS_DIR = "models/improved_models_intelligent"

# Toggle to retrain models from scratch (instead of loading from cache)
RETRAIN_MODE = False


# ======================
#     EXECUTION
# ======================

def main():
    """Main execution flow for draft prediction."""

    project_root = Path(__file__).resolve().parents[1]

    # Load data
    df = pd.read_csv(project_root / "processed_data" / "csv_games_fusionnes.csv")
    
    # Load champion roles map
    champion_roles_map = {}
    try:
        roles_df = pd.read_csv(project_root / "processed_data" / "champions_by_position.csv")
        for index, row in roles_df.iterrows():
            position = row['Position']
            champions_str = row['Champions']
            champions_list = [c.strip() for c in champions_str.split(',')]
            for champion in champions_list:
                if champion not in champion_roles_map:
                    champion_roles_map[champion] = []
                champion_roles_map[champion].append(position)
        print(f"Loaded {len(champion_roles_map)} champions with role information")
    except Exception as e:
        print(f"Warning: Could not load champion roles: {e}")

    def _apply_ban(team_bans, ban_value, all_used, label):
        team_bans.append(ban_value)
        all_used.append(ban_value)
        print(f"{label} = {ban_value}")

    def _apply_pick(team_picks, pick_value, team_assigned_roles, team_filled_roles, flexible_champions, all_used, label):
        team_picks.append(pick_value)
        assigned = assign_champion_role(
            pick_value,
            team_assigned_roles,
            team_filled_roles,
            champion_roles_map,
            flexible_champions,
        )
        all_used.append(pick_value)
        print(f"{label} = {pick_value} ({assigned})")

    # Initialize predictor with selected reward type
    reward_type = "intelligent_reward" if EVAL_INTELLIGENT else "standard"
    predictor = DraftPredictor(
        df,
        reward_type=reward_type,
        retrain_mode=RETRAIN_MODE,
        champion_roles_map=champion_roles_map,
        improved_models_dir=project_root / IMPROVED_MODELS_DIR,
    )

    # Run evaluation and exit if EVAL_MODE is enabled
    if EVAL_MODE:
        if EVAL_STANDARD:
            print("Running standard evaluation...")
            evaluate_all_targets(predictor)
        
        if EVAL_INTELLIGENT:
            print("Running intelligent reward evaluation...")
            evaluate_with_intelligent_rewards(predictor)
        
        return

    # ===== INTERACTIVE DRAFT MODE =====
    
    # Initialize team states
    blue_team_bans = []
    blue_team_picks = []
    blue_team_assigned_roles = {}
    blue_team_filled_roles = set()
    blue_team_flexible_champions = []

    red_team_bans = []
    red_team_picks = []
    red_team_assigned_roles = {}
    red_team_filled_roles = set()
    red_team_flexible_champions = []

    all_bans_and_picks = []

    mode_label = "Self-Play (Model vs Model)" if SELF_PLAY else "Draft Interactive"
    print(f"\n===== Démarrage de la {mode_label} =====\n")

    # Ban Phase 1
    if SELF_PLAY:
        bb1 = predictor.predict_bb1()
    else:
        bb1 = get_user_champion_input("Blue Team Ban 1 (bb1): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
    _apply_ban(blue_team_bans, bb1, all_bans_and_picks, "bb1")

    rb1 = predictor.predict_rb1(bb1, global_used_champs=all_bans_and_picks)
    _apply_ban(red_team_bans, rb1, all_bans_and_picks, "rb1")

    if SELF_PLAY:
        bb2 = predictor.predict_bb2(bb1, rb1, global_used_champs=all_bans_and_picks)
    else:
        bb2 = get_user_champion_input("Blue Team Ban 2 (bb2): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
    _apply_ban(blue_team_bans, bb2, all_bans_and_picks, "bb2")

    rb2 = predictor.predict_rb2(bb1, rb1, bb2, global_used_champs=all_bans_and_picks)
    _apply_ban(red_team_bans, rb2, all_bans_and_picks, "rb2")

    if SELF_PLAY:
        bb3 = predictor.predict_bb3(bb1, rb1, bb2, rb2, global_used_champs=all_bans_and_picks)
    else:
        bb3 = get_user_champion_input("Blue Team Ban 3 (bb3): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
    _apply_ban(blue_team_bans, bb3, all_bans_and_picks, "bb3")

    rb3 = predictor.predict_rb3(bb1, rb1, bb2, rb2, bb3, global_used_champs=all_bans_and_picks)
    _apply_ban(red_team_bans, rb3, all_bans_and_picks, "rb3")

    # Pick Phase 1
    if SELF_PLAY:
        bp1 = predictor.predict_bp1(bb1, rb1, bb2, rb2, bb3, rb3,
                                    blue_team_filled_roles=blue_team_filled_roles,
                                    global_used_champs=all_bans_and_picks)
    else:
        bp1 = get_user_champion_input("Blue Team Pick 1 (bp1): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
    _apply_pick(
        blue_team_picks,
        bp1,
        blue_team_assigned_roles,
        blue_team_filled_roles,
        blue_team_flexible_champions,
        all_bans_and_picks,
        "bp1",
    )

    rp1 = predictor.predict_rp1(bb1, rb1, bb2, rb2, bb3, rb3, bp1,
                                red_team_filled_roles=red_team_filled_roles,
                                global_used_champs=all_bans_and_picks)
    _apply_pick(
        red_team_picks,
        rp1,
        red_team_assigned_roles,
        red_team_filled_roles,
        red_team_flexible_champions,
        all_bans_and_picks,
        "rp1",
    )

    rp2 = predictor.predict_rp2(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1,
                                red_team_filled_roles=red_team_filled_roles,
                                global_used_champs=all_bans_and_picks)
    _apply_pick(
        red_team_picks,
        rp2,
        red_team_assigned_roles,
        red_team_filled_roles,
        red_team_flexible_champions,
        all_bans_and_picks,
        "rp2",
    )

    if SELF_PLAY:
        bp2 = predictor.predict_bp2(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2,
                                    blue_team_filled_roles=blue_team_filled_roles,
                                    global_used_champs=all_bans_and_picks)
    else:
        bp2 = get_user_champion_input("Blue Team Pick 2 (bp2): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
    _apply_pick(
        blue_team_picks,
        bp2,
        blue_team_assigned_roles,
        blue_team_filled_roles,
        blue_team_flexible_champions,
        all_bans_and_picks,
        "bp2",
    )

    if SELF_PLAY:
        bp3 = predictor.predict_bp3(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2,
                                    blue_team_filled_roles=blue_team_filled_roles,
                                    global_used_champs=all_bans_and_picks)
    else:
        bp3 = get_user_champion_input("Blue Team Pick 3 (bp3): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
    _apply_pick(
        blue_team_picks,
        bp3,
        blue_team_assigned_roles,
        blue_team_filled_roles,
        blue_team_flexible_champions,
        all_bans_and_picks,
        "bp3",
    )

    rp3 = predictor.predict_rp3(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3,
                                red_team_filled_roles=red_team_filled_roles,
                                global_used_champs=all_bans_and_picks)
    _apply_pick(
        red_team_picks,
        rp3,
        red_team_assigned_roles,
        red_team_filled_roles,
        red_team_flexible_champions,
        all_bans_and_picks,
        "rp3",
    )

    # Ban Phase 2
    rb4 = predictor.predict_rb4(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3,
                                global_used_champs=all_bans_and_picks)
    _apply_ban(red_team_bans, rb4, all_bans_and_picks, "rb4")

    if SELF_PLAY:
        bb4 = predictor.predict_bb4(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3,
                                    rb4, global_used_champs=all_bans_and_picks)
    else:
        bb4 = get_user_champion_input("Blue Team Ban 4 (bb4): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
    _apply_ban(blue_team_bans, bb4, all_bans_and_picks, "bb4")

    rb5 = predictor.predict_rb5(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4,
                                global_used_champs=all_bans_and_picks)
    _apply_ban(red_team_bans, rb5, all_bans_and_picks, "rb5")

    if SELF_PLAY:
        bb5 = predictor.predict_bb5(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3,
                                    rb4, bb4, rb5, global_used_champs=all_bans_and_picks)
    else:
        bb5 = get_user_champion_input("Blue Team Ban 5 (bb5): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
    _apply_ban(blue_team_bans, bb5, all_bans_and_picks, "bb5")

    # Pick Phase 2
    rp4 = predictor.predict_rp4(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, rb5, bb5,
                                red_team_filled_roles=red_team_filled_roles,
                                global_used_champs=all_bans_and_picks)
    _apply_pick(
        red_team_picks,
        rp4,
        red_team_assigned_roles,
        red_team_filled_roles,
        red_team_flexible_champions,
        all_bans_and_picks,
        "rp4",
    )

    if SELF_PLAY:
        bp4 = predictor.predict_bp4(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3,
                                    rb4, bb4, rb5, bb5, rp4,
                                    blue_team_filled_roles=blue_team_filled_roles,
                                    global_used_champs=all_bans_and_picks)
    else:
        bp4 = get_user_champion_input("Blue Team Pick 4 (bp4): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
    _apply_pick(
        blue_team_picks,
        bp4,
        blue_team_assigned_roles,
        blue_team_filled_roles,
        blue_team_flexible_champions,
        all_bans_and_picks,
        "bp4",
    )

    if SELF_PLAY:
        bp5 = predictor.predict_bp5(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3,
                                    rb4, bb4, rb5, bb5, rp4, bp4,
                                    blue_team_filled_roles=blue_team_filled_roles,
                                    global_used_champs=all_bans_and_picks)
    else:
        bp5 = get_user_champion_input("Blue Team Pick 5 (bp5): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
    _apply_pick(
        blue_team_picks,
        bp5,
        blue_team_assigned_roles,
        blue_team_filled_roles,
        blue_team_flexible_champions,
        all_bans_and_picks,
        "bp5",
    )

    rp5 = predictor.predict_rp5(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, rb5, bb5, rp4, bp4, bp5,
                                red_team_filled_roles=red_team_filled_roles,
                                global_used_champs=all_bans_and_picks)
    _apply_pick(
        red_team_picks,
        rp5,
        red_team_assigned_roles,
        red_team_filled_roles,
        red_team_flexible_champions,
        all_bans_and_picks,
        "rp5",
    )

    # Resolve flexible assignments
    print("\nResolving flexible champion roles for Blue Team...")
    resolve_flexible_assignments(blue_team_assigned_roles, blue_team_flexible_champions, blue_team_filled_roles)
    print("Resolving flexible champion roles for Red Team...")
    resolve_flexible_assignments(red_team_assigned_roles, red_team_flexible_champions, red_team_filled_roles)

    # Final output
    print("\n===== DRAFT PRÉDITE =====\n")
    print(f"Blue Team Bans: {tuple(blue_team_bans)}")
    print(f"Blue Team Picks: {tuple(blue_team_picks)}")
    print(f"Blue Team Assigned Roles: {blue_team_assigned_roles}")
    print(f"Red Team Bans: {tuple(red_team_bans)}")
    print(f"Red Team Picks: {tuple(red_team_picks)}")
    print(f"Red Team Assigned Roles: {red_team_assigned_roles}")
    print("\n===== FIN ====")


if __name__ == "__main__":
    main()
