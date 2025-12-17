import pandas as pd
import os

# Dossier contenant les CSV
folder_path = "CSV"

# Colonnes obligatoires pour le filtrage
required_cols = [
    "ban1","ban2","ban3","ban4","ban5",
    "pick1","pick2","pick3","pick4","pick5"
]

# DataFrames filtrés stockés ici
filtered_dfs = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        print("Traitement de :", file_path)

        # Charge le CSV
        df = pd.read_csv(file_path, low_memory=False)

        # Filtrage NaN
        df_filtered = df.dropna(subset=required_cols)

        # Filtrage chaînes vides
        df_filtered = df_filtered[
            (df_filtered[required_cols] != "").all(axis=1)
        ]

        print(" → lignes gardées :", len(df_filtered))

        filtered_dfs.append(df_filtered)

# Concaténation de tous les DataFrames filtrés
df_final = pd.concat(filtered_dfs, ignore_index=True)

# Enregistrement du CSV final
output_path = "csv_filtre_concatenes.csv"
df_final.to_csv(output_path, index=False)

print("\n====================================")
print("FUSION TERMINÉE !")
print("Nombre total de lignes :", len(df_final))
print("Nouveau CSV enregistré :", output_path)
print("====================================")
