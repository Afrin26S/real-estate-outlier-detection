"""
app.py
------
Flask backend for the Real Estate Price Outlier Detection project.

Endpoints:
  GET  /api/health          -> simple health check
  GET  /api/sample-data     -> returns the bundled sample dataset (raw rows)
  POST /api/analyze         -> body: {"properties": [...]} OR uses sample
                                data if no body is sent. Runs Grubbs' test +
                                EVT and returns per-property + per-zone results.
  POST /api/upload          -> multipart/form-data CSV upload, analyzed
                                the same way as /api/analyze.

Run with:  python app.py
Then open ../frontend/index.html in a browser (it calls this API on
http://localhost:5000).
"""

import io
import os

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from outlier_detection import run_full_analysis

app = Flask(__name__)
CORS(app)  # allow the static frontend (opened as a file:// or separate port) to call this API
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB cap on uploaded files

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_real_estate.csv")


@app.errorhandler(413)
def handle_file_too_large(e):
    return jsonify({"error": "File too large (max 5 MB)."}), 413


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Real Estate Outlier Detection API is running"})


@app.route("/api/sample-data", methods=["GET"])
def sample_data():
    df = pd.read_csv(DATA_PATH)
    return jsonify(df.to_dict(orient="records"))


def _analyze_dataframe(df):
    required_cols = {"property_id", "zone", "area_sqft", "price"}
    missing = required_cols - set(df.columns)
    if missing:
        return None, f"Dataset missing required columns: {sorted(missing)}"

    df = df.copy()

    # property_id must be unique: run_full_analysis merges results back
    # onto rows keyed by property_id, which silently breaks on duplicates.
    dupes = df["property_id"][df["property_id"].duplicated()].unique().tolist()
    if dupes:
        shown = dupes[:10]
        suffix = "..." if len(dupes) > 10 else ""
        return None, f"Duplicate property_id value(s) found: {shown}{suffix}"

    # Coerce numeric columns; anything that can't be parsed becomes NaN
    # instead of silently breaking arithmetic later or crashing int().
    numeric_cols = [c for c in ("area_sqft", "price", "bedrooms", "bathrooms", "age_years") if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_rows = df[df["area_sqft"].isna() | df["price"].isna()]
    if len(bad_rows):
        bad_ids = bad_rows["property_id"].astype(str).tolist()[:10]
        suffix = "..." if len(bad_rows) > 10 else ""
        return None, (
            f"{len(bad_rows)} row(s) have a non-numeric or missing area_sqft/price "
            f"and were rejected: {bad_ids}{suffix}"
        )

    non_positive = df[(df["area_sqft"] <= 0) | (df["price"] <= 0)]
    if len(non_positive):
        bad_ids = non_positive["property_id"].astype(str).tolist()[:10]
        suffix = "..." if len(non_positive) > 10 else ""
        return None, (
            f"{len(non_positive)} row(s) have area_sqft or price <= 0, which breaks "
            f"price-per-sqft math: {bad_ids}{suffix}"
        )

    # Optional numeric columns: fill any unparseable values with 0 rather
    # than crashing int(NaN) later when results are assembled.
    for col in ("bedrooms", "bathrooms", "age_years"):
        if col in df.columns:
            df[col] = df[col].fillna(0)

    try:
        results, zone_summaries = run_full_analysis(df)
    except ValueError as e:
        return None, str(e)

    outlier_count = sum(1 for r in results if r["classification"] != "Normal")
    overpriced = sum(1 for r in results if r["classification"] == "Overpriced Outlier")
    underpriced = sum(1 for r in results if r["classification"] == "Underpriced Outlier")

    response = {
        "properties": results,
        "zone_summaries": zone_summaries,
        "overview": {
            "total_properties": len(results),
            "total_outliers": outlier_count,
            "overpriced": overpriced,
            "underpriced": underpriced,
            "zones_analyzed": len(zone_summaries),
        },
    }
    return response, None


@app.route("/api/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(silent=True)

    if payload and "properties" in payload:
        df = pd.DataFrame(payload["properties"])
    else:
        # no body provided -> fall back to the bundled sample dataset
        df = pd.read_csv(DATA_PATH)

    response, error = _analyze_dataframe(df)
    if error:
        return jsonify({"error": error}), 400
    return jsonify(response)


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part named 'file' in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        content = file.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400

    response, error = _analyze_dataframe(df)
    if error:
        return jsonify({"error": error}), 400
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, port=5000)