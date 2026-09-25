"""
outlier_detection.py
---------------------
The statistical core of the project. Implements two complementary
anomaly-detection techniques and combines them into one verdict per
property:

1. GRUBBS' TEST (iterative / generalized ESD-style)
   A classical hypothesis test for detecting outliers in an
   approximately normal univariate sample. We run it per urban zone
   on price-per-sqft, removing the most extreme point each round and
   re-testing, until the test no longer finds a significant outlier.
   This gives a clear statistical (p-value backed) verdict.

2. EXTREME VALUE THEORY (Peaks-Over-Threshold + Generalized Pareto Dist.)
   Grubbs' assumes the *bulk* of the data is normal, which is a
   simplification. EVT instead models the *tail* of the distribution
   directly: we take the exceedances over a high threshold (90th
   percentile of |z|) and fit a Generalized Pareto Distribution (GPD)
   to them. This lets us assign a tail probability to any value
   without assuming the whole dataset is Gaussian, and is the
   standard tool for modelling rare/extreme events (floods, insurance
   claims, and here: freak property prices).

Combining both gives the project two independent lines of evidence,
which is exactly what a pattern-recognition/anomaly-detection course
wants to see: a classical test + a distribution-modelling approach.
"""

import numpy as np
import pandas as pd
from scipy import stats

GRUBBS_ALPHA = 0.05      # significance level for Grubbs' test
EVT_TAIL_QUANTILE = 0.90  # threshold quantile (on |z|) used for POT
EVT_PROB_CUTOFF = 0.01    # flag as EVT-outlier if tail exceedance prob < this


def grubbs_critical_value(n, alpha=GRUBBS_ALPHA):
    """Two-sided Grubbs' critical value G_crit for sample size n."""
    if n < 3:
        return np.inf
    t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    numerator = (n - 1) * np.sqrt(t_dist ** 2)
    denominator = np.sqrt(n) * np.sqrt(n - 2 + t_dist ** 2)
    return numerator / denominator


def iterative_grubbs(values, ids, alpha=GRUBBS_ALPHA, max_outliers_frac=0.25):
    """
    Repeatedly apply Grubbs' test, removing the most extreme remaining
    point each time it is significant, until:
      - the test is no longer significant, or
      - we've flagged more than `max_outliers_frac` of the sample
        (safety cap so one bad zone can't flag everything), or
      - fewer than 3 points remain.

    Returns a set of `ids` flagged as Grubbs outliers.
    """
    values = np.asarray(values, dtype=float)
    ids = np.asarray(ids)
    remaining_mask = np.ones(len(values), dtype=bool)
    flagged = set()
    max_flags = max(1, int(len(values) * max_outliers_frac))

    while remaining_mask.sum() >= 3 and len(flagged) < max_flags:
        sub_vals = values[remaining_mask]
        sub_ids = ids[remaining_mask]
        mean, std = sub_vals.mean(), sub_vals.std(ddof=1)
        if std == 0:
            break
        abs_dev = np.abs(sub_vals - mean)
        worst_local_idx = np.argmax(abs_dev)
        G = abs_dev[worst_local_idx] / std
        G_crit = grubbs_critical_value(len(sub_vals), alpha)

        if G > G_crit:
            flagged.add(sub_ids[worst_local_idx])
            # remove this point from the "remaining" pool and re-test
            global_idx = np.where(ids == sub_ids[worst_local_idx])[0][0]
            remaining_mask[global_idx] = False
        else:
            break

    return flagged


def evt_tail_probabilities(values):
    """
    Peaks-Over-Threshold EVT:
      1. Work on |z-score| so both very high and very low prices count
         as "extreme" in the same tail.
      2. Set threshold u = 90th percentile of |z|.
      3. Fit a Generalized Pareto Distribution to the exceedances (|z| - u).
      4. For every point, compute P(Z > z | Z > u) via the fitted GPD,
         combined with the empirical P(Z > u), to get an overall tail
         probability for each observation.

    Returns an array of tail probabilities (small = more extreme).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 8:
        # too small a sample for a stable GPD fit; fall back to a
        # simple empirical rank-based tail probability
        z = np.abs(stats.zscore(values)) if values.std() > 0 else np.zeros(n)
        ranks = stats.rankdata(-z)
        return ranks / (n + 1)

    mean, std = values.mean(), values.std(ddof=1)
    z = np.abs((values - mean) / std) if std > 0 else np.zeros(n)

    u = np.quantile(z, EVT_TAIL_QUANTILE)
    exceedances = z[z > u] - u
    p_exceed_u = len(exceedances) / n

    if len(exceedances) < 4 or exceedances.std() == 0:
        # not enough tail data to fit GPD reliably -> empirical fallback
        ranks = stats.rankdata(-z)
        return ranks / (n + 1)

    # Fit Generalized Pareto Distribution (shape c, loc fixed at 0, scale)
    c, loc, scale = stats.genpareto.fit(exceedances, floc=0)

    tail_prob = np.empty(n)
    for i, zi in enumerate(z):
        if zi <= u:
            # below threshold: use empirical proportion beyond this point
            tail_prob[i] = np.mean(z >= zi)
        else:
            exceed = zi - u
            # survival function of the fitted GPD at this exceedance,
            # scaled by the probability of exceeding u in the first place
            sf = stats.genpareto.sf(exceed, c, loc=0, scale=scale)
            tail_prob[i] = p_exceed_u * sf

    return tail_prob


def analyze_zone(df_zone):
    """
    Runs both detectors on one zone's data (using price-per-sqft as
    the normalizing variable, so a small flat and a big villa in the
    same zone are compared fairly) and returns a per-property result.
    """
    df_zone = df_zone.copy()
    # Guard against area_sqft <= 0, which would produce inf/NaN and poison
    # the mean/std for the whole zone. app.py validates this on the way in,
    # but this check makes the function safe even if called directly.
    if (df_zone["area_sqft"] <= 0).any():
        bad_ids = df_zone.loc[df_zone["area_sqft"] <= 0, "property_id"].tolist()
        raise ValueError(f"area_sqft must be positive; offending property_id(s): {bad_ids}")
    df_zone["price_per_sqft"] = df_zone["price"] / df_zone["area_sqft"]

    values = df_zone["price_per_sqft"].values
    ids = df_zone["property_id"].values

    mean, std = values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0
    z_scores = (values - mean) / std if std > 0 else np.zeros(len(values))

    grubbs_flagged = iterative_grubbs(values, ids)
    evt_probs = evt_tail_probabilities(values)

    results = []
    for i, pid in enumerate(ids):
        is_grubbs = pid in grubbs_flagged
        is_evt = evt_probs[i] < EVT_PROB_CUTOFF
        if is_grubbs or is_evt:
            direction = "Overpriced" if z_scores[i] > 0 else "Underpriced"
            if is_grubbs and is_evt:
                confidence = "High"
            else:
                confidence = "Moderate"
            classification = f"{direction} Outlier"
        else:
            classification = "Normal"
            confidence = "-"

        results.append({
            "property_id": pid,
            "price_per_sqft": round(float(values[i]), 2),
            "zone_mean_ppsf": round(float(mean), 2),
            "zone_std_ppsf": round(float(std), 2),
            "z_score": round(float(z_scores[i]), 3),
            "grubbs_outlier": bool(is_grubbs),
            "evt_tail_probability": round(float(evt_probs[i]), 5),
            "evt_outlier": bool(is_evt),
            "classification": classification,
            "confidence": confidence,
        })

    return results, {
        "zone_mean_ppsf": round(float(mean), 2),
        "zone_std_ppsf": round(float(std), 2),
        "grubbs_critical_value": round(float(grubbs_critical_value(len(values))), 3),
        "n_properties": len(values),
        "n_outliers": sum(1 for r in results if r["classification"] != "Normal"),
    }


def run_full_analysis(df):
    """
    Entry point used by the API. Takes the full properties DataFrame,
    runs analyze_zone() per urban zone, and returns:
      - a flat list of per-property results (merged with original fields)
      - a per-zone summary dict

    Assumes property_id values are unique (enforced by app.py's
    _analyze_dataframe before this is called) -- the merge step below
    looks rows up by property_id and will behave unpredictably on
    duplicates.
    """
    required_cols = {"property_id", "zone", "area_sqft", "price"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    all_results = []
    zone_summaries = {}

    for zone, df_zone in df.groupby("zone"):
        results, summary = analyze_zone(df_zone)
        zone_summaries[zone] = summary
        for r in results:
            r["zone"] = zone
        all_results.extend(results)

    # merge stats back onto the original row data for a complete record
    df_indexed = df.set_index("property_id")
    for r in all_results:
        row = df_indexed.loc[r["property_id"]]
        r["area_sqft"] = float(row["area_sqft"])
        r["price"] = float(row["price"])
        if "bedrooms" in row:
            r["bedrooms"] = int(row["bedrooms"])
        if "bathrooms" in row:
            r["bathrooms"] = int(row["bathrooms"])
        if "age_years" in row:
            r["age_years"] = int(row["age_years"])

    return all_results, zone_summaries