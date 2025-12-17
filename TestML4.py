import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.extmath import softmax


class DraftPredictor:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._model_cache = {}  # Cache for models and encoders

    def _predict_ml(self, target_col, feature_cols, result_filter, feature_values):
        # Cache key
        cache_key = f"{target_col}_{result_filter}"

        # Load cached model if available
        if cache_key in self._model_cache:
            model, encoder = self._model_cache[cache_key]
        else:
            # Filter by result
            df_f = self.df[self.df["result"] == result_filter].copy()
            df_f = df_f.dropna(subset=feature_cols + [target_col])

            X = df_f[feature_cols]
            y = df_f[target_col]

            # Keep classes appearing at least twice
            freq = y.value_counts()
            valid = freq[freq >= 2].index
            mask = y.isin(valid)
            X = X[mask]
            y = y[mask]

            # One-hot encode
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            X_encoded = encoder.fit_transform(X)

            # ===== LinearSVC classifier =====
            model = LinearSVC(
                C=1.0,
                max_iter=5000,
                dual=True
            )
            model.fit(X_encoded, y)

            # Cache model
            self._model_cache[cache_key] = (model, encoder)

        # Retrieve cached model + encoder
        model, encoder = self._model_cache[cache_key]

        # Encode input features
        X_input = encoder.transform([feature_values])

        # ===== LinearSVC → decision_function → pseudo-probabilities =====
        scores = model.decision_function(X_input)[0]

        # Handle binary classification: convert scalar score to 2-class scores
        if isinstance(scores, float) or np.ndim(scores) == 0:
            scores = np.array([-scores, scores])

        probas = softmax([scores])[0]
        classes = model.classes_

        # Remove already-picked champs
        used = set(feature_values)
        mask = np.array([c not in used for c in classes])
        probas = probas * mask

        # Fallback if everything is masked
        if probas.sum() == 0:
            remaining = [c for c in classes if c not in used]
            return remaining[0] if remaining else classes[0]

        return classes[np.argmax(probas)]

    # --- Draft prediction functions ---
    def predict_bb1(self):
        df_b = self.df[self.df["result"] == "b"]
        return df_b["bb1"].value_counts().idxmax()

    def predict_rb1(self, bb1):
        return self._predict_ml("rb1", ["bb1"], "r", [bb1])

    def predict_bb2(self, bb1, rb1):
        return self._predict_ml("bb2", ["bb1", "rb1"], "b", [bb1, rb1])

    def predict_rb2(self, bb1, rb1, bb2):
        return self._predict_ml("rb2", ["bb1", "rb1", "bb2"], "r", [bb1, rb1, bb2])

    def predict_bb3(self, bb1, rb1, bb2, rb2):
        return self._predict_ml("bb3", ["bb1", "rb1", "bb2", "rb2"], "b", [bb1, rb1, bb2, rb2])

    def predict_rb3(self, bb1, rb1, bb2, rb2, bb3):
        return self._predict_ml("rb3", ["bb1", "rb1", "bb2", "rb2", "bb3"], "r", [bb1, rb1, bb2, rb2, bb3])

    def predict_bp1(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3"]
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
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3"]
        return self._predict_ml("rb4", cols, "r", list(picks))

    def predict_bb4(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4"]
        return self._predict_ml("bb4", cols, "b", list(picks))

    def predict_rb5(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4"]
        return self._predict_ml("rb5", cols, "r", list(picks))

    def predict_bb5(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5"]
        return self._predict_ml("bb5", cols, "b", list(picks))

    def predict_rp4(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5","bb5"]
        return self._predict_ml("rp4", cols, "r", list(picks))

    def predict_bp4(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5","bb5","rp4"]
        return self._predict_ml("bp4", cols, "b", list(picks))

    def predict_bp5(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5","bb5","rp4","bp4"]
        return self._predict_ml("bp5", cols, "b", list(picks))

    def predict_rp5(self, *picks):
        cols = ["bb1","rb1","bb2","rb2","bb3","rb3","bp1","rp1","rp2","bp2","bp3","rp3","rb4","bb4","rb5","bb5","rp4","bp4","bp5"]
        return self._predict_ml("rp5", cols, "r", list(picks))


# ======================
#     EXECUTION
# ======================

df = pd.read_csv("csv_games_fusionnes.csv")
predictor = DraftPredictor(df)

# ======================
#   PRÉDICTIONS DRAFT
# ======================

# Pick 1
bb1 = input("bb1 : ")
rb1 = predictor.predict_rb1(bb1)


print(f"rb1 = {rb1}")


# Pick 2
bb2 = input("bb2 : ")
rb2 = predictor.predict_rb2(bb1, rb1, bb2)


print(f"rb2 = {rb2}")

# Pick 3
bb3 = input("bb3 : ")
rb3 = predictor.predict_rb3(bb1, rb1, bb2, rb2, bb3)


print(f"rb3 = {rb3}")

# Ban phase (bp1 / rp1 / rp2 / bp2 / bp3 / rp3)
bp1 = input("bp1 : ")
rp1 = predictor.predict_rp1(bb1, rb1, bb2, rb2, bb3, rb3, bp1)


print(f"rp1 = {rp1}")


rp2 = predictor.predict_rp2(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1)

print(f"rp2 = {rp2}")
bp2 = input("bp2 : ")





bp3 = input("bp3 : ")
rp3 = predictor.predict_rp3(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2, bp2, bp3)


print(f"rp3 = {rp3}")

# Pick 4
rb4 = predictor.predict_rb4(bb1, rb1, bb2, rb2, bb3, rb3,
                            bp1, rp1, rp2, bp2, bp3, rp3)

print(f"rb4 = {rb4}")
bb4 = input("bb4 : ")




# Pick 5
rb5 = predictor.predict_rb5(bb1, rb1, bb2, rb2, bb3, rb3,
                            bp1, rp1, rp2, bp2, bp3, rp3, rb4, bb4)

print(f"rb5 = {rb5}")
bb5 = input("bb5 : ")



# Final ban phase (rp4 / bp4 / bp5 / rp5)
rp4 = predictor.predict_rp4(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2,
                            bp2, bp3, rp3, rb4, bb4, rb5, bb5)

print(f"rp4 = {rp4}")
bp4 = input("bp4 : ")




bp5 = input("bp5 : ")
rp5 = predictor.predict_rp5(bb1, rb1, bb2, rb2, bb3, rb3, bp1, rp1, rp2,
                            bp2, bp3, rp3, rb4, bb4, rb5, bb5, rp4, bp4, bp5)

print(f"rp5 = {rp5}")

# ======================
#     FINAL OUTPUT
# ======================

print("\n===== DRAFT PRÉDITE =====\n")

print(f"ban blue = {bb1,bb2,bb3,bb4,bb5}")
print(f"peack blue = {bp1,bp2,bp3,bp4,bp5}")
print(f"ban red  = {rb1,rb2,rb3,rb4,rb5}")
print(f"pick red = {rp1,rp2,rp3,rp4,rp5}")

print("\n===== FIN =====")
