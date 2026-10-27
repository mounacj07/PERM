# ST-AGBDT (expectile boosted trees + conformal bound) Final Model Documentation & Format Specification

## 1. Overview
This directory contains the production-ready **ST-AGBDT (expectile boosted trees + conformal bound)** models trained on Locust canonical runs (`data/runs_locust`).
All models use **arrival-only features**, ensuring predictions remain valid regardless of online worker autoscaling pool adjustments.

- **Git Commit**: `908aca68d8b3095a429e0ee1349955ee540215d6` (tagged `final-model-v1`)
- **Loss Function**: Asymmetric Least Squares (Expectile) with $\alpha = 0.85$
- **Ensemble**: 50 trees, max depth 4, learning rate 0.1, 64 histogram bins
- **Target**: Maximum of 3-second rolling smoothed arrivals over future horizon $h \in \{10, 20, 30\}$ seconds
- **Capacity**: Calibrated at 7.89 req/s per worker (`testbed_calibration.json:results[0].saturation_rps`), targeted at 70% utilization (5.523 req/s per worker).

---

## 2. File Manifest
- `models/stagbdt_h10_lean.json`: Compiled flat-tree ensemble for $h=10\text{s}$ (zero external ML dependencies).
- `models/stagbdt_h20_lean.json`: Compiled flat-tree ensemble for $h=20\text{s}$.
- `models/stagbdt_h30_lean.json`: Compiled flat-tree ensemble for $h=30\text{s}$.
- `models/stagbdt_final_config.json`: Master metadata file containing features, run splits, conformal margins, and hyperparameters.
- `models/README_STAGBDT.md`: This file.

---

## 3. Lean Model JSON Schema
Each lean JSON file stores the ensemble compiled into continuous flat NumPy arrays for single-sample inference in <100 µs:
```json
{
  "model_type": "STAGBDT",
  "full_name": "Self-Throttling Asymmetric Gradient Boosted Decision Trees",
  "n_estimators": 50,
  "learning_rate": 0.1,
  "max_depth": 4,
  "alpha": 0.85,
  "l2_reg": 1.0,
  "min_child_hessian": 0.001,
  "n_bins": 64,
  "random_state": 42,
  "base_pred_": 4.1234,
  "feature_names_": [...],
  "bin_edges_": [[...], ...],
  "flat_trees_": [
    {
      "max_depth": 4,
      "feature_indices": [...],
      "thresholds": [...],
      "values": [...]
    },
    ...
  ]
}
```

### Loading & Inference (Pure NumPy)
```python
from stagbdt.model import STAGBDT

# Load model without any external ML dependencies
model = STAGBDT.load_json("models/stagbdt_h20_lean.json")

# Predict raw arrival rate
raw_forecast = model.predict_one(feature_vector)

# Apply 90% split-conformal upper bound
conformal_margin = 5.0763
upper_bound = raw_forecast + conformal_margin

# Compute recommended workers (clipped [2, 8])
eff_cap = 5.523
recommended_workers = int(min(8, max(2, math.ceil(upper_bound / eff_cap))))
```

---

## 4. Arrival-Only Feature Specification (25 Features)
Features are computed over 1-second arrival counts $arr$:
- `arr_curr`: Instantaneous count $arr[t]$
- For window durations $w \in [5, 10, 30, 60]$ seconds:
  - `arr_mean_{w}`: Rolling mean $\frac{1}{w} \sum arr$
  - `arr_max_{w}`: Rolling maximum $\max arr$
  - `arr_std_{w}`: Rolling population standard deviation
  - `arr_slope_{w}`: Linear slope $(arr[t] - arr[t-w+1]) / (w - 1)$
  - `arr_accel_{w}`: Slope change between recent and older half-window
  - `arr_ewma_{w}`: Exponential moving average with $\alpha_{ewma} = \frac{2}{w + 1}$

Both offline batch extraction and online streaming buffer use the identical implementation in `stagbdt/features.py`.

---

## 5. Calibrated Conformal Margins (90% Coverage)
Calibrated on the 7 designated calibration runs (`random_mix` seeds 15–19, `step_seed44`, `ramp_seed44`):
- **h = 10s**: `+3.4595` req/s
- **h = 20s**: `+5.0763` req/s
- **h = 30s**: `+6.7019` req/s
