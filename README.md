# House Price Prediction

A regression project for predicting residential sale prices from mixed tabular features. The emphasis is on leakage-safe validation, robust preprocessing, and evaluation in both currency and relative-error terms.

## Results

| Metric | Held-out validation |
|---|---:|
| MAE | 15,830.01 |
| RMSE | 27,974.15 |
| R² | 0.8980 |
| RMSLE | 0.1332 |

The final solution uses `HistGradientBoostingRegressor` with a log-transformed target. The transformation reduces the influence of the most expensive homes and aligns training more closely with relative price error.

## Modeling approach

- Audited missingness, skewed distributions, and categorical cardinality.
- Kept preprocessing inside a reproducible pipeline.
- Compared linear and tree-based baselines.
- Used a held-out split for final evaluation and reported metrics on the original price scale.
- Inspected residuals to identify where the model is least reliable.

## Repository guide

- `notebooks/` — EDA, model comparison, and final evaluation.
- `src/` — reusable code, where available.
- `reports/` — figures and outputs.
- `requirements.txt` — recorded environment.

## Reproduce

Create a Python environment, install `requirements.txt`, and run the notebooks in order. Data paths are relative to the project root.

## Limitations and next steps

A random holdout does not reproduce every real-estate market shift. A production version would add time- or geography-based validation, prediction intervals, richer location features, and drift monitoring.

## Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn
