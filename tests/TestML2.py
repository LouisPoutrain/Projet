
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder



class DraftPredictor:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _predict_ml(self, target_col, feature_cols, result_filter, feature_values):
        df_f = self.df[self.df["result"] == result_filter].copy()
        df_f = df_f.dropna(subset=feature_cols + [target_col])

        X = df_f[feature_cols]
        y = df_f[target_col]

        # Garder classes fréquentes
        freq = y.value_counts()
        valid = freq[freq >= 2].index
        mask = y.isin(valid)
        X = X[mask]
        y = y[mask]

        # OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_encoded = encoder.fit_transform(X)

        # LightGBM
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=len(y.unique()),
            n_estimators=200,
            max_depth=7,
            random_state=42,
            n_jobs=-1,
            verbose=-1 
        )
        model.fit(X_encoded, y)

        # Encodage de l'entrée
        X_input = encoder.transform([feature_values])
        probas = model.predict_proba(X_input)[0]
        classes = model.classes_

        # Exclure champions déjà pickés
        used = set(feature_values)
        probas = model.predict_proba(X_input)[0]
        classes = model.classes_

        # On met à zéro les probabilités des champions déjà pickés
        mask = [c not in used for c in classes]
        probas = probas * mask

        # Si tout est éliminé → fallback sécurisé
        if probas.sum() == 0:
            # choisir le champion le plus fréquent parmi ceux non pickés
            remaining_classes = [c for c in classes if c not in used]
            return remaining_classes[0]  # ou un random.choice(remaining_classes)

        return classes[probas.argmax()]



    # --- Fonctions de prédiction du draft ---
    def predict_bb1(self):
        # bb1 use only itself distribution
        df_b = self.df[self.df["result"] == "b"]
        return df_b["bb1"].value_counts().idxmax()

    def predict_rb1(self, bb1):
        return self._predict_ml("rb1", ["bb1"], "r", [bb1])

    def predict_bb2(self, bb1, rb1):
        return self._predict_ml("bb2", ["bb1", "rb1"], "b", [bb1, rb1])

    def predict_rb2(self, bb1, rb1, bb2):
        return self._predict_ml("rb2", ["bb1", "rb1", "bb2"], "r",
                                [bb1, rb1, bb2])

    def predict_bb3(self, bb1, rb1, bb2, rb2):
        return self._predict_ml("bb3", ["bb1", "rb1", "bb2", "rb2"],
                                "b", [bb1, rb1, bb2, rb2])

    def predict_rb3(self, bb1, rb1, bb2, rb2, bb3):
        return self._predict_ml("rb3", ["bb1", "rb1", "bb2", "rb2", "bb3"],
                                "r", [bb1, rb1, bb2, rb2, bb3])

    def predict_bp1(self, *picks):
        cols = ["bb1", "rb1", "bb2", "rb2", "bb3", "rb3"]
        return self._predict_ml("bp1", cols, "b", list(picks))

    def predict_rp1(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1"]
        return self._predict_ml("rp1", cols, "r", list(picks))

    def predict_rp2(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1"]
        return self._predict_ml("rp2", cols, "r", list(picks))

    def predict_bp2(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2"]
        return self._predict_ml("bp2", cols, "b", list(picks))

    def predict_bp3(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2"]
        return self._predict_ml("bp3", cols, "b", list(picks))

    def predict_rp3(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3"]
        return self._predict_ml("rp3", cols, "r", list(picks))

    def predict_rb4(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1",
                "rp2","bp2","bp3","rp3"]
        return self._predict_ml("rb4", cols, "r", list(picks))

    def predict_bb4(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1",
                "rp2","bp2","bp3","rp3","rb4"]
        return self._predict_ml("bb4", cols, "b", list(picks))

    def predict_rb5(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1",
                "rp2","bp2","bp3","rp3","rb4","bb4"]
        return self._predict_ml("rb5", cols, "r", list(picks))

    def predict_bb5(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1",
                "rp2","bp2","bp3","rp3","rb4","bb4","rb5"]
        return self._predict_ml("bb5", cols, "b", list(picks))

    def predict_rp4(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2",
                "bp2","bp3","rp3","rb4","bb4","rb5","bb5"]
        return self._predict_ml("rp4", cols, "r", list(picks))

    def predict_bp4(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2",
                "bp2","bp3","rp3","rb4","bb4","rb5","bb5","rp4"]
        return self._predict_ml("bp4", cols, "b", list(picks))

    def predict_bp5(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3",
                "rp3","rb4","bb4","rb5","bb5","rp4","bp4"]
        return self._predict_ml("bp5", cols, "b", list(picks))

    def predict_rp5(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3",
                "rp3","rb4","bb4","rb5","bb5","rp4","bp4","bp5"]
        return self._predict_ml("rp5", cols, "r", list(picks))


df = pd.read_csv("csv_games_fusionnes.csv")

predictor = DraftPredictor(df)

# ======================
#   PRÉDICTIONS DRAFT
# ======================

# Pick 1
bb1 = predictor.predict_bb1()
rb1 = predictor.predict_rb1(bb1)

print(f"bb1 = {bb1}")
print(f"rb1 = {rb1}")


# Pick 2
bb2 = predictor.predict_bb2(bb1, rb1)
rb2 = predictor.predict_rb2(bb1, rb1, bb2)

print(f"bb2 = {bb2}")
print(f"rb2 = {rb2}")

# Pick 3
bb3 = predictor.predict_bb3(bb1, rb1, bb2, rb2)
rb3 = predictor.predict_rb3(bb1, rb1, bb2, rb2, bb3)

print(f"bb3 = {bb3}")
print(f"rb3 = {rb3}")

# Ban phase (bp1 / rp1 / rp2 / bp2 / bp3 / rp3)
bp1 = predictor.predict_bp1(bb1, rb1, bb2, rb2, bb3, rb3)
rp1 = predictor.predict_rp1(bb1, rb1, bb2, rb2, bb3, rb3, bp1)

print(f"bp1 = {bp1}")
print(f"rp1 = {rp1}")


rp2 = predictor.predict_rp2(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1)
bp2 = predictor.predict_bp2(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2)

print(f"rp2 = {rp2}")
print(f"bp2 = {bp2}")


bp3 = predictor.predict_bp3(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2)
rp3 = predictor.predict_rp3(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3)

print(f"bp3 = {bp3}")
print(f"rp3 = {rp3}")

# Pick 4
rb4 = predictor.predict_rb4(bb1, rb1, bb2, rb2, bb3, rb3,
                            bp1, rp1, rp2, bp2, bp3, rp3)
bb4 = predictor.predict_bb4(bb1, rb1, bb2, rb2, bb3, rb3,
                            bp1, rp1, rp2, bp2, bp3, rp3, rb4)

print(f"rb4 = {rb4}")
print(f"bb4 = {bb4}")

# Pick 5
rb5 = predictor.predict_rb5(bb1, rb1, bb2, rb2, bb3, rb3,
                            bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4)
bb5 = predictor.predict_bb5(bb1, rb1, bb2, rb2, bb3, rb3,
                            bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4, rb5)
print(f"rb5 = {rb5}")
print(f"bb5 = {bb5}")

# Final ban phase (rp4 / bp4 / bp5 / rp5)
rp4 = predictor.predict_rp4(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2,
                            bp2, bp3, rp3, rb4, bb4, rb5, bb5)
bp4 = predictor.predict_bp4(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2,
                            bp2, bp3, rp3, rb4, bb4, rb5, bb5, rp4)

print(f"rp4 = {rp4}")
print(f"bp4 = {bp4}")

bp5 = predictor.predict_bp5(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2,
                            bp2, bp3, rp3, rb4, bb4, rb5, bb5, rp4, bp4)
rp5 = predictor.predict_rp5(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2,
                            bp2, bp3, rp3, rb4, bb4, rb5, bb5, rp4, bp4, bp5)
print(f"bp5 = {bp5}")
print(f"rp5 = {rp5}")

# ======================
#     RÉSULTAT FINAL
# ======================

print("\n===== DRAFT PRÉDITE =====\n")

print(f"ban blue = {bb1,bb2,bb3,bb4,bb5}")
print(f"peack blue = {bp1,bp2,bp3,bp4,bp5}")
print(f"ban red  = {rb1,rb2,rb3,rb4,rb5}")
print(f"pick red = {rp1,rp2,rp3,rp4,rp5}")


print("\n===== FIN =====")