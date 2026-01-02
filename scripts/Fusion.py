import pandas as pd

# On charge le CSV précédent
df = pd.read_csv("../processed_data/draft_dataset_bans_picks.csv", low_memory=False)

# Séparer Blue et Red
blue = df[df["side"] == "Blue"].copy()
red  = df[df["side"] == "Red"].copy()

# Renommer les colonnes pour Blue
blue = blue.rename(columns={
    "ban1": "bb1", "ban2": "bb2", "ban3": "bb3", "ban4": "bb4", "ban5": "bb5",
    "pick1": "bp1", "pick2": "bp2", "pick3": "bp3", "pick4": "bp4", "pick5": "bp5",
})

# Renommer les colonnes pour Red
red = red.rename(columns={
    "ban1": "rb1", "ban2": "rb2", "ban3": "rb3", "ban4": "rb4", "ban5": "rb5",
    "pick1": "rp1", "pick2": "rp2", "pick3": "rp3", "pick4": "rp4", "pick5": "rp5",
})

# Sélectionner uniquement les colonnes utiles
blue = blue[["gameid","bb1","bb2","bb3","bb4","bb5","bp1","bp2","bp3","bp4","bp5","result"]]
red  = red[ ["gameid","rb1","rb2","rb3","rb4","rb5","rp1","rp2","rp3","rp4","rp5","result"]]

# Fusionner sur gameid
merged = pd.merge(blue, red, on="gameid", suffixes=("_blue","_red"))

# Calcul du résultat final : 'b' ou 'r'
def compute_result(row):
    # si le Blue side a result == 1 → victoire blue
    if row["result_blue"] == 1:
        return "b"
    # sinon → victoire red
    return "r"

merged["result"] = merged.apply(compute_result, axis=1)

# On supprime les anciennes colonnes result_blue/result_red
merged = merged.drop(columns=["result_blue", "result_red"])

# Sauvegarde du fichier final
output_path = "../processed_data/csv_games_fusionnes.csv"
merged.to_csv(output_path, index=False)

print("Fusion terminée !")
print("Nouveau fichier :", output_path)
print("Nombre total de games :", len(merged))

