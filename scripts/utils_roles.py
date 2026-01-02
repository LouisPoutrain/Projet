"""
Utility functions for role assignment and champion input handling.
"""
import re
import pandas as pd


def normalize_champion_name(name: str) -> str:
    """Normalize champion names to a lowercase alphanumerics-only key."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load_winrate_lookup(winrate_csv_path='../processed_data/all_matchup_winrates.csv'):
    """
    Load win rate data from CSV and build a lookup dictionary.
    
    Returns:
        dict: {(champion, position, opponent, opponent_pos) -> win_rate}
    """
    try:
        df_winrates = pd.read_csv(winrate_csv_path)
        winrate_lookup = {}
        for _, row in df_winrates.iterrows():
            key = (row['champion'], row['position'], row['opponent'], row['opponent_pos'])
            winrate_lookup[key] = row['win_rate']
        print(f"Loaded {len(winrate_lookup)} matchup win rates")
        return winrate_lookup
    except Exception as e:
        print(f"Warning: Could not load win rate data: {e}")
        return {}


def assign_champion_role(champion, team_roles_dict, team_filled_roles_set, champion_roles_map, flexible_champions_list):
    """
    Assign a role to a picked champion based on available roles and flexibility.
    
    Returns:
        str: The assigned role
    """
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
        provisional_role = "PROV-" + list(available_roles_for_champ_slot)[0]
        team_roles_dict[champion] = provisional_role
        return provisional_role
    else:  # len(available_roles_for_champ_slot) == 0
        # All roles the champion could play are already definitively assigned.
        assigned_role = possible_roles_for_champion[0]
        team_filled_roles_set.add(assigned_role)
        team_roles_dict[champion] = assigned_role
        return assigned_role


def get_user_champion_input(prompt_text, current_team_filled_roles, global_used_champs, champion_roles_map):
    """
    Get champion input from user with validation and role-aware warnings.
    
    Returns:
        str: The normalized champion name
    """
    while True:
        user_input = input(prompt_text).strip()
        
        # Try to find the champion (case-insensitive lookup)
        norm_input = normalize_champion_name(user_input)
        champion_name = None
        
        for champ in champion_roles_map.keys():
            if normalize_champion_name(champ) == norm_input:
                champion_name = champ
                break
        
        if champion_name is None:
            print(f"Champion '{user_input}' not found in the list of known champions. Please try again.")
            continue

        if champion_name in global_used_champs:
            print(f"Champion '{champion_name}' has already been banned or picked. Please choose another champion.")
            continue

        # Warn user if picking a champion that cannot fill an open role
        all_roles = {'top', 'jng', 'mid', 'bot', 'sup'}
        unfilled_roles = all_roles - current_team_filled_roles
        possible_roles_for_champion = champion_roles_map.get(champion_name, [])

        can_fill_an_unfilled_role = any(role in unfilled_roles for role in possible_roles_for_champion)

        if len(unfilled_roles) > 0 and not can_fill_an_unfilled_role:
            print(f"Warning: '{champion_name}' cannot fill any of your remaining unfilled roles ({', '.join(unfilled_roles)}). Type 'yes' to confirm or any other key to choose another: ")
            confirm = input().strip().lower()
            if confirm != 'yes':
                continue

        return champion_name


def resolve_flexible_assignments(team_assigned_roles, team_flexible_champions, team_filled_roles_set):
    """
    Resolve flexible champion role assignments iteratively.
    """
    all_roles = {'top', 'jng', 'mid', 'bot', 'sup'}
    resolved_count = 0

    # Continue as long as there are flexible champions and progress is being made
    while True:
        new_assignments_made_in_iteration = False
        champions_to_remove = []

        # Iterate through flexible champions to find unique assignments
        for i, (champion, possible_roles) in enumerate(team_flexible_champions):
            current_unfilled_roles = all_roles - team_filled_roles_set
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
        for index in sorted(champions_to_remove, reverse=True):
            team_flexible_champions.pop(index)

        if not new_assignments_made_in_iteration and team_flexible_champions:
            break
        elif not team_flexible_champions:
            break

    # Handle any remaining flexible champions
    for champion, possible_roles in team_flexible_champions:
        current_provisional_role = team_assigned_roles[champion]
        if current_provisional_role.startswith("PROV-"):
            del team_assigned_roles[champion]

        # Find the first possible role that isn't already definitively filled
        assigned_role = None
        for role in possible_roles:
            if role not in team_filled_roles_set:
                assigned_role = role
                break

        if assigned_role:
            team_assigned_roles[champion] = assigned_role
            team_filled_roles_set.add(assigned_role)
        else:
            assigned_role = list(possible_roles)[0]
            team_assigned_roles[champion] = assigned_role
            team_filled_roles_set.add(assigned_role)

    print(f"Resolved {resolved_count} flexible champion roles.")
