"""
generate_data.py
-----------------
Creates a synthetic but realistic real-estate dataset with multiple
"urban zones", each having its own typical price-per-sqft range.
A handful of properties in each zone are deliberately injected as
outliers (over- or under-priced) so the detection pipeline has
something real to find. Run once to (re)build data/sample_real_estate.csv.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# Each zone has a different "typical" price-per-sqft (in INR) and
# a different typical property profile. This mimics real cities
# where a downtown core is pricier per sqft than rural outskirts.
ZONES = {
    "Downtown Core":        {"n": 60, "ppsf_mean": 9500,  "ppsf_std": 900,  "area_mean": 950,  "area_std": 200},
    "Suburban Residential": {"n": 70, "ppsf_mean": 5200,  "ppsf_std": 500,  "area_mean": 1400, "area_std": 300},
    "Waterfront":           {"n": 40, "ppsf_mean": 11800, "ppsf_std": 1100, "area_mean": 1600, "area_std": 350},
    "Industrial Fringe":    {"n": 50, "ppsf_mean": 3100,  "ppsf_std": 350,  "area_mean": 1100, "area_std": 250},
    "Rural Outskirts":      {"n": 55, "ppsf_mean": 1800,  "ppsf_std": 250,  "area_mean": 1800, "area_std": 400},
}

rows = []
pid = 1000

for zone, cfg in ZONES.items():
    n = cfg["n"]
    areas = np.random.normal(cfg["area_mean"], cfg["area_std"], n).clip(300, None)
    ppsf = np.random.normal(cfg["ppsf_mean"], cfg["ppsf_std"], n).clip(200, None)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.3, 0.35, 0.18, 0.07])
    bathrooms = np.clip(bedrooms - np.random.choice([0, 1], n, p=[0.6, 0.4]), 1, None)
    age = np.random.randint(0, 40, n)

    for i in range(n):
        price = round(areas[i] * ppsf[i], -3)
        rows.append({
            "property_id": f"P{pid}",
            "zone": zone,
            "area_sqft": round(areas[i], 1),
            "bedrooms": int(bedrooms[i]),
            "bathrooms": int(bathrooms[i]),
            "age_years": int(age[i]),
            "price": int(price),
        })
        pid += 1

    # --- Deliberately inject a few real outliers into this zone ---
    n_outliers = max(2, n // 20)
    outlier_idx = np.random.choice(n, n_outliers, replace=False)
    for idx in outlier_idx:
        direction = np.random.choice(["over", "under"])
        base_ppsf = cfg["ppsf_mean"]
        if direction == "over":
            # a property priced 2.2x-3x the zone's normal rate (e.g. bad data entry, unique luxury feature)
            factor = np.random.uniform(2.2, 3.0)
        else:
            # a property priced 0.3x-0.5x the zone's normal rate (e.g. distress sale, undervaluation)
            factor = np.random.uniform(0.3, 0.5)
        # overwrite the price of one already-appended row for this zone
        target = rows[len(rows) - n + idx]
        target["price"] = int(round(target["area_sqft"] * base_ppsf * factor, -3))

df = pd.DataFrame(rows)
df.to_csv("data/sample_real_estate.csv", index=False)
print(f"Wrote {len(df)} rows to data/sample_real_estate.csv")
print(df.groupby("zone")["price"].describe()[["mean", "std", "min", "max"]])