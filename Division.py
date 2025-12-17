import pandas as pd
import os

# Charger le CSV final
df = pd.read_csv("csv_games_fusionnes.csv")

# Liste des colonnes pour chaque CSV (ordre croissant)
column_sets = [
    ["result","bb1"],
    ["result","bb1","rb1"],
    ["result","bb1","rb1","bb2"],
    ["result","bb1","rb1","bb2","rb2"],
    ["result","bb1","rb1","bb2","rb2","bb3"],
    ["result","bb1","rb1","bb2","rb2","bb3","rb3"],
    ["result","bb1","rb1","bb2","rb2","bb3","rb3","bp1"],
    ["result","bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1"],
    ["result","bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2"],
    ["result","bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2"],
    ["result","bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3"],
    ["result","bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3"],
    ["result","bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4"],
    ["result","bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4"],
    ["result","bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5"],
    ["result","bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5","bb5"],
    ["result","bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5","bb5","rp4"],
    ["result","bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5","bb5","rp4","bp4"],
    ["result","bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5","bb5","rp4","bp4","bp5"],
    ["result","bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5","bb5","rp4","bp4","bp5","rp5"]
]

# Dossier de sortie
output_folder = "csv_sousensembles"
os.makedirs(output_folder, exist_ok=True)

# Boucle pour créer chaque CSV
for i, cols in enumerate(column_sets, start=1):
    subset_df = df[cols]
    output_path = os.path.join(output_folder, f"csv_subset_{i}.csv")
    subset_df.to_csv(output_path, index=False)
    print(f"CSV {i} créé : {output_path} ({len(subset_df.columns)} colonnes)")
