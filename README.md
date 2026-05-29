# 03 House Prices

## Project goal

This project solves a supervised tabular regression task using the Kaggle House Prices dataset.

The goal is to predict house sale prices from numerical and categorical property features while practicing a correct machine learning workflow:
- dataset audit;
- EDA;
- local train/test split;
- preprocessing with Pipeline / ColumnTransformer;
- cross-validation;
- log-target transformation;
- final holdout evaluation;
- model artifact saving.

## Dataset

Dataset: House Prices - Advanced Regression Techniques

Main files:
- `train.csv`: contains features and target `SalePrice`;
- `test.csv`: official Kaggle test file without `SalePrice`;
- `sample_submission.csv`: Kaggle submission format;
- `data_description.txt`: feature dictionary.

The official Kaggle `test.csv` was not used for local validation or final local evaluation.

## Task framing

Task type: regression  
Target: `SalePrice`

Feature matrix:
- `SalePrice` removed from `X`;
- `Id` excluded from `X` by default because it is an identifier, not a meaningful predictive feature.

## Validation design

A local holdout split was created from `train.csv` only:

```python
train_test_split(test_size=0.2, random_state=42)

The local holdout test set was kept untouched until the final evaluation stage.

Model comparison and tuning were performed using cross-validation on X_train only.

Preprocessing

Preprocessing was implemented inside Pipeline / ColumnTransformer.

Numeric features:

SimpleImputer(strategy="median")
StandardScaler

Categorical features:

SimpleImputer(strategy="constant", fill_value="None")
OneHotEncoder(handle_unknown="ignore", sparse_output=False)

No imputer, scaler, or encoder was fitted on the full dataset before split.

Model development summary

Baseline models:

DummyRegressor
LinearRegression
Ridge
DecisionTreeRegressor(max_depth=3)

Controlled model comparison:

RandomForestRegressor
ExtraTreesRegressor
GradientBoostingRegressor
HistGradientBoostingRegressor

Best Stage 4 model by RMSE:

HistGradientBoostingRegressor

Stage 5 tested log-target transformation with:

TransformedTargetRegressor(
    func=np.log1p,
    inverse_func=np.expm1
)
Final model

Final frozen candidate:

HistGradientBoostingRegressor
Target transform: log1p(SalePrice)
Inverse transform: expm1

Parameters:

l2_regularization = 0.1
learning_rate = 0.1
max_leaf_nodes = 15
random_state = 42
Final local test metrics

The final model was evaluated once on the held-out local test set.

Metric	Value
MAE	15,830.01
RMSE	27,974.15
R²	0.8980
RMSLE	0.1332
Comparison with Stage 5 CV

Stage 5 tuned HGB + log-target CV:

Metric	CV	Final test
MAE	~16,485	15,830.01
RMSE	~28,110	27,974.15
R²	~0.8663	0.8980
RMSLE	~0.1326	0.1332

The final holdout metrics aligned well with CV expectations.

Train/test diagnostic

The final model fits the training data much more strongly than the holdout data, which is expected for flexible boosting models.

Split	MAE	RMSE	R²	RMSLE
Train	7,506.37	12,380.06	0.9743	0.0619
Test	15,830.01	27,974.15	0.8980	0.1332

This indicates substantial train/test gap, but not a validation failure because the final test metrics match Stage 5 CV expectations.

Improvement over Ridge baseline

Approximate comparison against Stage 3 Ridge CV baseline:

Metric	Ridge CV baseline	Final test	Improvement
MAE	~18,758	15,830	~15.6%
RMSE	~33,858	27,974	~17.4%
R²	~0.803	0.898	+0.095

This comparison is directional because baseline numbers are CV metrics on X_train, while final metrics are from the held-out local test set.

Leakage controls
Official Kaggle test.csv was not used for local validation or final evaluation.
Local X_test/y_test was evaluated exactly once.
No model switching after final test evaluation.
No retuning after final test evaluation.
No outlier removal after final test evaluation.
Id was excluded from X.
Preprocessing stayed inside Pipeline / ColumnTransformer.
Target transformation was handled with TransformedTargetRegressor.
Key limitations
The final model was selected by RMSE priority.
GradientBoostingRegressor was stronger by MAE/RMSLE during CV, but was not evaluated after the HGB candidate was frozen.
Some overfitting capacity is visible from the train/test diagnostic.
No advanced feature engineering was added.
No official Kaggle submission workflow was included in this stage.
Project structure
03_house_prices/
├── data/
│   └── raw/
├── models/
│   └── final_house_price_pipeline.joblib
├── notebooks/
│   ├── 00_dataset_audit.ipynb
│   ├── 01_initial_eda.ipynb
│   ├── 02_baseline_regression.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_log_target_and_tuning.ipynb
│   └── 05_final_evaluation.ipynb
├── src/
├── .gitignore
└── README.md
Reproduction steps

Create and activate virtual environment:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install pandas numpy matplotlib scikit-learn jupyter ipykernel joblib

Run notebooks in order:

00_dataset_audit.ipynb
01_initial_eda.ipynb
02_baseline_regression.ipynb
03_model_comparison.ipynb
04_log_target_and_tuning.ipynb
05_final_evaluation.ipynb

Final model artifact:

models/final_house_price_pipeline.joblib

## 5. Git

После сохранения artifact и README:

```powershell
git status