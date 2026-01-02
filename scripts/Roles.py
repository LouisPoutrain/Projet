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

# Toggle evaluation types (only active if EVAL_MODE = True)
EVAL_STANDARD = False     # Standard accuracy / F1-macro metrics
EVAL_INTELLIGENT = True   # Win rate-based intelligent rewards

# Toggle to retrain models from scratch (instead of loading from cache)
RETRAIN_MODE = False


# ======================
#     EXECUTION
# ======================

def main():
    """Main execution flow for draft prediction."""
    
    # Load data
    df = pd.read_csv("../processed_data/csv_games_fusionnes.csv")
    
    # Load champion roles map
    champion_roles_map = {}
    champion_lookup = {}
    try:
        roles_df = pd.read_csv('../processed_data/champions_by_position.csv')
        for index, row in roles_df.iterrows():
            position = row['Position']
            champions_str = row['Champions']
            champions_list = [c.strip() for c in champions_str.split(',')]
            for champion in champions_list:
                if champion not in champion_roles_map:
                    champion_roles_map[champion] = []
                champion_roles_map[champion].append(position)
                norm = normalize_champion_name(champion)
                if norm not in champion_lookup:
                    champion_lookup[norm] = champion
        print(f"Loaded {len(champion_roles_map)} champions with role information")
    except Exception as e:
        print(f"Warning: Could not load champion roles: {e}")

    # Initialize predictor with selected reward type
    reward_type = "intelligent_reward" if EVAL_INTELLIGENT else "standard"
    predictor = DraftPredictor(df, reward_type=reward_type, retrain_mode=RETRAIN_MODE, 
                              champion_roles_map=champion_roles_map)

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

    SELF_PLAY = False
    mode_label = "Self-Play (Model vs Model)" if SELF_PLAY else "Draft Interactive"
    print(f"\n===== Démarrage de la {mode_label} =====\n")

    # Ban Phase 1
    if SELF_PLAY:
        bb1 = predictor.predict_bb1()
    else:
        bb1 = get_user_champion_input("Blue Team Ban 1 (bb1): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
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
        bb2 = get_user_champion_input("Blue Team Ban 2 (bb2): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
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
        bb3 = get_user_champion_input("Blue Team Ban 3 (bb3): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
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
        bp1 = get_user_champion_input("Blue Team Pick 1 (bp1): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
    blue_team_picks.append(bp1)
    assigned_role_bp1 = assign_champion_role(bp1, blue_team_assigned_roles, blue_team_filled_roles, 
                                             champion_roles_map, blue_team_flexible_champions)
    all_bans_and_picks.append(bp1)
    print(f"bp1 = {bp1} ({assigned_role_bp1})")

    rp1 = predictor.predict_rp1(bb1, rb1, bb2, rb2, bb3, rb3, bp1,
                                red_team_filled_roles=red_team_filled_roles,
                                global_used_champs=all_bans_and_picks)
    red_team_picks.append(rp1)
    assigned_role_rp1 = assign_champion_role(rp1, red_team_assigned_roles, red_team_filled_roles, 
                                             champion_roles_map, red_team_flexible_champions)
    all_bans_and_picks.append(rp1)
    print(f"rp1 = {rp1} ({assigned_role_rp1})")

    rp2 = predictor.predict_rp2(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1,
                                red_team_filled_roles=red_team_filled_roles,
                                global_used_champs=all_bans_and_picks)
    red_team_picks.append(rp2)
    assigned_role_rp2 = assign_champion_role(rp2, red_team_assigned_roles, red_team_filled_roles, 
                                             champion_roles_map, red_team_flexible_champions)
    all_bans_and_picks.append(rp2)
    print(f"rp2 = {rp2} ({assigned_role_rp2})")

    if SELF_PLAY:
        bp2 = predictor.predict_bp2(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2,
                                    blue_team_filled_roles=blue_team_filled_roles,
                                    global_used_champs=all_bans_and_picks)
    else:
        bp2 = get_user_champion_input("Blue Team Pick 2 (bp2): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
    blue_team_picks.append(bp2)
    assigned_role_bp2 = assign_champion_role(bp2, blue_team_assigned_roles, blue_team_filled_roles, 
                                             champion_roles_map, blue_team_flexible_champions)
    all_bans_and_picks.append(bp2)
    print(f"bp2 = {bp2} ({assigned_role_bp2})")

    if SELF_PLAY:
        bp3 = predictor.predict_bp3(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2,
                                    blue_team_filled_roles=blue_team_filled_roles,
                                    global_used_champs=all_bans_and_picks)
    else:
        bp3 = get_user_champion_input("Blue Team Pick 3 (bp3): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
    blue_team_picks.append(bp3)
    assigned_role_bp3 = assign_champion_role(bp3, blue_team_assigned_roles, blue_team_filled_roles, 
                                             champion_roles_map, blue_team_flexible_champions)
    all_bans_and_picks.append(bp3)
    print(f"bp3 = {bp3} ({assigned_role_bp3})")

    rp3 = predictor.predict_rp3(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3,
                                red_team_filled_roles=red_team_filled_roles,
                                global_used_champs=all_bans_and_picks)
    red_team_picks.append(rp3)
    assigned_role_rp3 = assign_champion_role(rp3, red_team_assigned_roles, red_team_filled_roles, 
                                             champion_roles_map, red_team_flexible_champions)
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
        bb4 = get_user_champion_input("Blue Team Ban 4 (bb4): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
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
        bb5 = get_user_champion_input("Blue Team Ban 5 (bb5): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
    blue_team_bans.append(bb5)
    all_bans_and_picks.append(bb5)
    print(f"bb5 = {bb5}")

    # Pick Phase 2
    rp4 = predictor.predict_rp4(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, rb5, bb5,
                                red_team_filled_roles=red_team_filled_roles,
                                global_used_champs=all_bans_and_picks)
    red_team_picks.append(rp4)
    assigned_role_rp4 = assign_champion_role(rp4, red_team_assigned_roles, red_team_filled_roles, 
                                             champion_roles_map, red_team_flexible_champions)
    all_bans_and_picks.append(rp4)
    print(f"rp4 = {rp4} ({assigned_role_rp4})")

    if SELF_PLAY:
        bp4 = predictor.predict_bp4(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3,
                                    rb4, bb4, rb5, bb5, rp4,
                                    blue_team_filled_roles=blue_team_filled_roles,
                                    global_used_champs=all_bans_and_picks)
    else:
        bp4 = get_user_champion_input("Blue Team Pick 4 (bp4): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
    blue_team_picks.append(bp4)
    assigned_role_bp4 = assign_champion_role(bp4, blue_team_assigned_roles, blue_team_filled_roles, 
                                             champion_roles_map, blue_team_flexible_champions)
    all_bans_and_picks.append(bp4)
    print(f"bp4 = {bp4} ({assigned_role_bp4})")

    if SELF_PLAY:
        bp5 = predictor.predict_bp5(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3,
                                    rb4, bb4, rb5, bb5, rp4, bp4,
                                    blue_team_filled_roles=blue_team_filled_roles,
                                    global_used_champs=all_bans_and_picks)
    else:
        bp5 = get_user_champion_input("Blue Team Pick 5 (bp5): ", blue_team_filled_roles, 
                                       all_bans_and_picks, champion_roles_map)
    blue_team_picks.append(bp5)
    assigned_role_bp5 = assign_champion_role(bp5, blue_team_assigned_roles, blue_team_filled_roles, 
                                             champion_roles_map, blue_team_flexible_champions)
    all_bans_and_picks.append(bp5)
    print(f"bp5 = {bp5} ({assigned_role_bp5})")

    rp5 = predictor.predict_rp5(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, rb5, bb5, rp4, bp4, bp5,
                                red_team_filled_roles=red_team_filled_roles,
                                global_used_champs=all_bans_and_picks)
    red_team_picks.append(rp5)
    assigned_role_rp5 = assign_champion_role(rp5, red_team_assigned_roles, red_team_filled_roles, 
                                             champion_roles_map, red_team_flexible_champions)
    all_bans_and_picks.append(rp5)
    print(f"rp5 = {rp5} ({assigned_role_rp5})")

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
