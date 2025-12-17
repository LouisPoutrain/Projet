import pandas as pd

# Path to the CSV file
file_path = 'CSV/2025_LoL_esports_match_data_from_OraclesElixir.csv'

try:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    print("DataFrame loaded successfully. Shape:", df.shape)
    print("DataFrame head:\n", df.head())
    print("DataFrame columns:", df.columns.tolist())

    # Select the 'champion' and 'position' columns
    # Drop rows where either 'champion' or 'position' is missing
    champions_positions = df[['champion', 'position']].dropna()

    print("Champions and positions after dropping NaNs, shape:", champions_positions.shape)

    if champions_positions.empty:
        print("Warning: champions_positions DataFrame is empty after dropping NaN values. No champions to list.")
    else:
        # Count champion appearances per position
        champion_position_counts = champions_positions.groupby(['position', 'champion']).size().reset_index(name='count')

        # Filter to keep only champions that appear more than 5 times in that position
        filtered_champion_position_counts = champion_position_counts[champion_position_counts['count'] > 5]

        # Group by position and get the list of champions from the filtered data
        # This will be used to generate the champions_by_position.csv in the next step
        global champions_by_position
        champions_by_position = filtered_champion_position_counts.groupby('position')['champion'].apply(list)

        print(f"Grouped champions by position after filtering (count > 5). Number of unique positions found: {len(champions_by_position)}")

        # Print the result (limiting to 10 for brevity)
        print("Liste des champions par position (plus de 5 apparitions) :\n")
        for position, champions in champions_by_position.items():
            print(f"Position: {position}")
            print(f"  Champions: {', '.join(sorted(list(champions))[:10])}...")
            print("  (" + str(len(champions)) + " champions au total)\n")

except FileNotFoundError:
    print(f"Erreur: Le fichier '{file_path}' n'a pas été trouvé.")
except KeyError as e:
    print(f"Erreur: Une des colonnes requises est manquante dans le fichier CSV. Détails: {e}")
except Exception as e:
    print(f"Une erreur inattendue est survenue: {e}")
# Assuming champions_by_position is already available from the previous execution
# If not, the previous cell needs to be run first

# Convert the Series to a DataFrame for easier saving
champions_df = champions_by_position.reset_index()

# Rename the columns for clarity
champions_df.columns = ['Position', 'Champions']

# Convert the list of champions (numpy array) into a comma-separated string for each position
champions_df['Champions'] = champions_df['Champions'].apply(lambda x: ', '.join(sorted(list(x))))

# Define the output file path
output_file_path = 'champions_by_position.csv'

# Save the DataFrame to a CSV file
champions_df.to_csv(output_file_path, index=False)

print(f"La liste des champions par position a été enregistrée dans '{output_file_path}'")
print("Aperçu du fichier CSV :")
print(champions_df.head())