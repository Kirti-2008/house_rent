"""
app.py  –  Flat Recommendation API
────────────────────────────────────
Endpoints
  POST /predict          → predict rent for ONE flat spec
  POST /recommend        → list matching flats from the dataset
  GET  /metadata         → valid cities, localities, furnishing types, ranges
  GET  /health           → liveness check
"""

import os, json, pickle, logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# ─────────────────────────────── setup ────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
DATA_PATH  = os.path.join(BASE_DIR, "data.csv")

# ─────────────────────────── load artefacts ───────────────────────────
def load_artefacts():
    with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "preprocessor.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "rb") as f:
        label_encoders = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
        meta = json.load(f)
    df = pd.read_csv(DATA_PATH)
    return model, scaler, label_encoders, meta, df

try:
    MODEL, SCALER, LABEL_ENC, META, DATASET = load_artefacts()
    FEATURE_COLS = META["feature_cols"]
    log.info("✔ Artefacts loaded. Best model: %s", META["best_model"])
except Exception as e:
    log.error("Could not load artefacts: %s – run train_model.py first.", e)
    MODEL = SCALER = LABEL_ENC = META = DATASET = None
    FEATURE_COLS = []

# ─────────────────────────── helpers ──────────────────────────────────
FURNISHING_ALIASES = {
    "semi": "Semi-Furnished", "semi-furnished": "Semi-Furnished",
    "semi furnished": "Semi-Furnished",
    "unfurnished": "Unfurnished", "un-furnished": "Unfurnished",
    "furnished": "Furnished",
}

def normalise_furnishing(val: str) -> str:
    return FURNISHING_ALIASES.get(val.lower().strip(), val.strip())

def encode_feature(col: str, val: str) -> int:
    le = LABEL_ENC[col]
    classes = list(le.classes_)
    val_strip = val.strip()
    if val_strip not in classes:
        raise ValueError(
            f"Unknown {col} '{val_strip}'. Valid: {classes}"
        )
    return int(le.transform([val_strip])[0])

def build_feature_vector(data: dict) -> np.ndarray:
    """Turn a request dict into a scaled feature vector."""
    furnishing = normalise_furnishing(data.get("furnishing", ""))

    row = [
        float(data["area"]),
        int(data["beds"]),
        int(data["bathrooms"]),
        int(data.get("balconies", 0)),
        float(data.get("area_rate", 0)),
        encode_feature("city",       data["city"]),
        encode_feature("locality",   data["locality"]),
        encode_feature("furnishing", furnishing),
    ]
    return SCALER.transform([row])

def flat_to_dict(row: pd.Series) -> dict:
    return {
        "house_type":  row["house_type"],
        "locality":    row["locality"],
        "city":        row["city"],
        "area":        row["area"],
        "beds":        int(row["beds"]),
        "bathrooms":   int(row["bathrooms"]),
        "balconies":   int(row["balconies"]),
        "furnishing":  row["furnishing"],
        "area_rate":   row["area_rate"],
        "rent":        row["rent"],
    }

# ─────────────────────────── routes ───────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "best_model":   META["best_model"] if META else None,
    })


@app.route("/metadata", methods=["GET"])
def metadata():
    """Return valid values the caller can use to build a form."""
    if not META:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify({
        "cities":        META["cities"],
        "localities":    META["localities"],
        "furnishings":   META["furnishings"],
        "beds_range":    META["beds_range"],
        "area_range":    META["area_range"],
        "rent_range":    META["rent_range"],
        "model_results": META["model_results"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the rent for a single flat specification.

    Body (JSON):
      city, locality, area, beds, bathrooms,
      furnishing, balconies (opt, default 0), area_rate (opt, default 0)
    """
    if not MODEL:
        return jsonify({"error": "Model not loaded – run train_model.py"}), 503

    data = request.get_json(force=True)
    required = ["city", "locality", "area", "beds", "bathrooms", "furnishing"]
    missing  = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        X = build_feature_vector(data)
        predicted_rent = float(MODEL.predict(X)[0])
        return jsonify({
            "predicted_rent_inr": round(predicted_rent, 2),
            "input_summary": {
                "city":       data["city"],
                "locality":   data["locality"],
                "area_sqft":  data["area"],
                "beds":       data["beds"],
                "bathrooms":  data["bathrooms"],
                "furnishing": normalise_furnishing(data.get("furnishing", "")),
            },
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        log.exception("Predict error")
        return jsonify({"error": str(e)}), 500


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Return matching flats from the dataset that satisfy user requirements.

    Required body fields:
      city         – exact city name
      beds         – minimum number of bedrooms
      max_rent     – maximum monthly rent (INR)

    Optional body fields:
      locality          – filter by locality (substring match)
      furnishing        – Furnished / Semi-Furnished / Unfurnished
      min_area          – minimum area (sq ft)
      max_area          – maximum area (sq ft)
      bathrooms         – minimum bathrooms
      balconies         – minimum balconies
      top_n             – number of results (default 10, max 50)
      sort_by           – "rent" | "area" | "area_rate"  (default "rent")
    """
    if DATASET is None:
        return jsonify({"error": "Model not loaded – run train_model.py"}), 503

    data     = request.get_json(force=True)
    required = ["city", "beds", "max_rent"]
    missing  = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        df = DATASET.copy()

        # ── mandatory filters ──────────────────────────────────────────
        df = df[df["city"].str.lower() == data["city"].lower()]
        df = df[df["beds"] >= int(data["beds"])]
        df = df[df["rent"] <= float(data["max_rent"])]

        # ── optional filters ───────────────────────────────────────────
        if "locality" in data and data["locality"]:
            df = df[df["locality"].str.lower().str.contains(
                data["locality"].lower(), na=False)]

        if "furnishing" in data and data["furnishing"]:
            norm = normalise_furnishing(data["furnishing"])
            df   = df[df["furnishing"].str.lower() == norm.lower()]

        if "min_area" in data:
            df = df[df["area"] >= float(data["min_area"])]
        if "max_area" in data:
            df = df[df["area"] <= float(data["max_area"])]
        if "bathrooms" in data:
            df = df[df["bathrooms"] >= int(data["bathrooms"])]
        if "balconies" in data:
            df = df[df["balconies"] >= int(data["balconies"])]

        # ── sort & limit ───────────────────────────────────────────────
        sort_col = data.get("sort_by", "rent")
        if sort_col not in ("rent", "area", "area_rate"):
            sort_col = "rent"
        top_n = min(int(data.get("top_n", 10)), 50)

        df = df.sort_values(sort_col).head(top_n)

        flats = [flat_to_dict(row) for _, row in df.iterrows()]

        return jsonify({
            "total_found": len(flats),
            "filters_applied": {
                "city":       data["city"],
                "min_beds":   int(data["beds"]),
                "max_rent":   float(data["max_rent"]),
                "locality":   data.get("locality"),
                "furnishing": data.get("furnishing"),
                "min_area":   data.get("min_area"),
                "bathrooms":  data.get("bathrooms"),
                "balconies":  data.get("balconies"),
            },
            "sorted_by":  sort_col,
            "flats":      flats,
        })

    except Exception as e:
        log.exception("Recommend error")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────── main ─────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)