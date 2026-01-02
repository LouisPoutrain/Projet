import pandas as pd

# Charger le CSV fusionné et filtré
df = pd.read_csv("csv_filtre_concatenes.csv", low_memory=False)

# Colonnes à conserver
cols_to_keep = [
    "gameid",
    "side",
    "ban1","ban2","ban3","ban4","ban5",
    "pick1","pick2","pick3","pick4","pick5",
    "result"
]

# On ne garde que ces colonnes
df_reduced = df[cols_to_keep]

# Sauvegarde du CSV final
output_path = "draft_dataset_bans_picks.csv"
df_reduced.to_csv(output_path, index=False)

print("Nouveau fichier sauvegardé sous :", output_path)
print("Nombre de lignes :", len(df_reduced))
print("Colonnes conservées :", df_reduced.columns.tolist())
