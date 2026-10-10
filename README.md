# Student Performance Prediction

Predict final grades (`G3`) with regression and classify performance bands using the Kaggle/UCI secondary-school dataset. The Streamlit dashboard runs cross-validated benchmarks and scores a student profile for exploratory early-support analysis. It is not intended for high-stakes decisions about students.

## Dataset

- [Student Alcohol Consumption on Kaggle](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption)
- Math and Portuguese course CSVs; the default app uses the 395-row Math file.
- Kaggle lists the dataset as CC0. Raw CSVs are downloaded locally and excluded from Git.

The model excludes `G1` and `G2` to represent prediction before those term grades are available. Grade bands are fixed: 0-9 at-risk, 10-13 developing, 14-16 proficient, 17-20 advanced.

## Method and Results

Preprocessing is fitted inside scikit-learn pipelines. The fixed-seed 80/20 holdout experiment compared Ridge, Random Forest, and Gradient Boosting for regression and two tree ensembles for classification.

| Task / model | Cross-validation | Test results |
| --- | ---: | --- |
| Regression: Random Forest | MAE 2.91 | MAE 3.01, RMSE 3.76, $R^2$ 0.309 |
| Regression: Gradient Boosting | MAE 3.10 | MAE 3.11, RMSE 3.93, $R^2$ 0.247 |
| Regression: Ridge | MAE 3.56 | MAE 3.39, RMSE 4.19, $R^2$ 0.142 |
| Bands: Random Forest | Macro-F1 0.376 | Accuracy 0.367, Macro-F1 0.322 |
| Bands: Gradient Boosting | Macro-F1 0.340 | Accuracy 0.405, Macro-F1 0.344 |

The advanced class is small, so multi-class performance is unstable. Predictions need educator context and must not be used to determine student outcomes.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/download_dataset.py
```

Launch the dashboard with `streamlit run app.py`. It runs on `http://localhost:8501` by default. Run checks with `python -m pytest` and `python -m ruff check .`.

## Layout

`app.py` is the Streamlit dashboard; `src/student_performance/` contains the validated loader, pipelines, and experiment runner; `tests/` covers behavior; `configs/` stores experiment settings. Raw datasets and model files are excluded from Git.