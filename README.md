# Real Estate Price Outlier Detection

**Course:** Pattern Recognition and Anomaly Detection
**Project topic:** Real Estate Price Outliers
**Method:** Grubbs' Test + Extreme Value Theory (Peaks-Over-Threshold / Generalized Pareto Distribution)

A full-stack project that flags properties priced far outside the norm for
their urban zone. A Python/Flask backend runs the statistical detection;
a browser dashboard visualizes the results.

```
real-estate-outlier-detection/
├── backend/
│   ├── app.py                  Flask REST API
│   ├── outlier_detection.py    Grubbs' test + EVT implementation (the core logic)
│   ├── generate_data.py        Script that built the sample dataset (optional to re-run)
│   ├── requirements.txt
│   └── data/
│       └── sample_real_estate.csv   275 synthetic properties across 5 zones
├── frontend/
│   └── index.html              Dashboard (vanilla HTML/CSS/JS + Chart.js)
└── README.md
```

## How to run it

1. **Backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   python app.py
   ```
   This starts the API at `http://localhost:5000`. Leave this terminal running.

2. **Frontend**
   Just open `frontend/index.html` in a browser (double-click it, or
   right-click → Open With → your browser). It automatically calls the
   API on page load and shows the analysis.

   You can also upload your own CSV from the dashboard — it needs at
   least the columns `property_id, zone, area_sqft, price` (bedrooms,
   bathrooms, age_years are optional extras).

## What each part does, for your report / viva

**The dataset** (`generate_data.py` → `sample_real_estate.csv`): 275
synthetic properties spread across 5 urban zones (Downtown Core,
Suburban Residential, Waterfront, Industrial Fringe, Rural Outskirts),
each zone with its own realistic price-per-sqft range. A handful of
properties per zone are deliberately overwritten with unrealistic
prices (2.2–3× or 0.3–0.5× the zone's normal rate) so the detectors
have real anomalies to find — this is a standard way to validate an
anomaly detector when you don't have a labelled real-world dataset.

**Why price-per-sqft, and why per zone?** Comparing raw prices across
a studio flat and a villa is meaningless, and comparing a Downtown
property to a Rural one is meaningless too — "expensive" only makes
sense relative to similar properties. So the pipeline normalizes to
price/sqft and analyzes each urban zone as its own group.

**Grubbs' test** (`outlier_detection.py::iterative_grubbs`) is a
classical hypothesis test for one outlier in an approximately normal
sample: it compares the most extreme point's distance from the mean
(in standard deviations) against a critical value derived from the
t-distribution. Here it's applied *iteratively* — flag the worst
point, remove it, re-test the rest — which is the standard way to
find more than one outlier per group (a fixed cap prevents it from
over-flagging a genuinely skewed zone).

**Extreme Value Theory** (`outlier_detection.py::evt_tail_probabilities`)
takes a different, distribution-free approach to the tail: instead of
assuming the whole zone is normally distributed, it only models the
*extremes*. It sets a threshold at the 90th percentile of |z-score|,
fits a Generalized Pareto Distribution to the exceedances over that
threshold (the standard Peaks-Over-Threshold method used in fields
like hydrology and insurance for rare-event modelling), and uses that
fit to compute how probable each property's price really is. This is
the "extreme value analysis" the project topic explicitly asks for,
and it's a genuinely different technique from Grubbs' — not just a
second run of the same idea — which is worth pointing out to your
examiner.

**Combining them:** a property is an outlier if *either* test flags
it; if *both* agree, the dashboard marks it "High confidence" rather
than just "Moderate". This two-method design is deliberate: Grubbs'
gives you a clean, well-known statistical test with a p-value story;
EVT gives you a probability-based severity score without the
normality assumption. Together they make a stronger case than either
alone — and it directly demonstrates two different anomaly-detection
paradigms from the course.

**The API** (`app.py`) is a thin REST layer: it loads a dataset (bundled
sample or an uploaded CSV), calls `run_full_analysis()`, and returns
per-property verdicts plus per-zone summary statistics as JSON.

**The dashboard** (`frontend/index.html`) is a single self-contained
HTML file — no build tooling — that fetches from the API and renders:
a scatter plot of price vs. area colored by verdict, a per-zone
summary table (mean, std dev, Grubbs' critical value, outlier count),
and a sortable ledger of every property with its z-score, EVT tail
probability, and final classification.

## Ideas if you want to extend it further

- Add a map view (zones plotted geographically) if your dataset has
  real coordinates.
- Swap the synthetic dataset for a real one (e.g. a city's public
  property records) — the pipeline needs no code changes as long as
  the four required columns are present.
- Add a second EVT variant (Block Maxima + GEV fit) alongside the
  current POT + GPD approach, and compare which flags more sensibly
  on your data — good material for a "comparison of methods" section.
