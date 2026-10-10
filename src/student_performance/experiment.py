"""Train and compare student final-grade regression and band classifiers."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from student_performance.pipelines import (
    classification_pipelines,
    make_grade_band,
    regression_pipelines,
)


def load_student_data(source: str | Path) -> pd.DataFrame:
    """Load a student CSV and validate the final-grade target."""
    try:
        frame = pd.read_csv(source)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"Could not read student CSV: {error}") from error
    if frame.empty or "G3" not in frame.columns:
        raise ValueError("Student data must be non-empty and contain the G3 target.")
    frame["G3"] = pd.to_numeric(frame["G3"], errors="coerce")
    if frame["G3"].isna().any() or not frame["G3"].between(0, 20).all():
        raise ValueError("G3 must contain grades from 0 through 20 without missing values.")
    return frame


def run_student_experiment(
    frame: pd.DataFrame, random_state: int = 42, cv_folds: int = 5
) -> dict[str, Any]:
    """Compare leakage-conscious grade regressors and performance-band classifiers.

    G1 and G2 are excluded to model early support before term grades are
    available. All learned preprocessing remains within each model pipeline.
    """
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if "G3" not in frame.columns or frame.empty:
        raise ValueError("A non-empty frame with a G3 column is required.")

    labels_regression = frame["G3"].astype(float)
    features = frame.drop(columns=[column for column in ("G1", "G2", "G3") if column in frame])
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    if features.empty:
        raise ValueError("No predictors remain after excluding G1, G2, and G3.")

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels_regression,
        test_size=0.2,
        random_state=random_state,
    )
    regression_results: list[dict[str, float | str]] = []
    fitted_regressors: dict[str, Any] = {}
    for name, pipeline in regression_pipelines(numeric, categorical).items():
        cv_scores = cross_val_score(
            pipeline,
            train_x,
            train_y,
            cv=cv_folds,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        pipeline.fit(train_x, train_y)
        predictions = pipeline.predict(test_x)
        regression_results.append(
            {
                "model": name,
                "cv_mae": float(-cv_scores.mean()),
                "test_mae": float(mean_absolute_error(test_y, predictions)),
                "test_rmse": float(np.sqrt(mean_squared_error(test_y, predictions))),
                "test_r2": float(r2_score(test_y, predictions)),
            }
        )
        fitted_regressors[name] = pipeline

    band_labels = make_grade_band(labels_regression).astype(str)
    train_x_cls, test_x_cls, train_y_cls, test_y_cls = train_test_split(
        features,
        band_labels,
        test_size=0.2,
        stratify=band_labels,
        random_state=random_state,
    )
    folds = min(cv_folds, int(train_y_cls.value_counts().min()))
    if folds < 2:
        raise ValueError("Each grade band needs at least two training examples.")
    stratified_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    classification_results: list[dict[str, float | str]] = []
    fitted_classifiers: dict[str, Any] = {}
    for name, pipeline in classification_pipelines(numeric, categorical).items():
        cv_scores = cross_val_score(
            pipeline,
            train_x_cls,
            train_y_cls,
            cv=stratified_cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        pipeline.fit(train_x_cls, train_y_cls)
        predictions = pipeline.predict(test_x_cls)
        classification_results.append(
            {
                "model": name,
                "cv_macro_f1": float(cv_scores.mean()),
                "test_accuracy": float(accuracy_score(test_y_cls, predictions)),
                "test_macro_f1": float(f1_score(test_y_cls, predictions, average="macro")),
            }
        )
        fitted_classifiers[name] = pipeline

    return {
        "regression": pd.DataFrame(regression_results).sort_values("test_mae"),
        "classification": pd.DataFrame(classification_results).sort_values(
            "test_macro_f1", ascending=False
        ),
        "regressors": fitted_regressors,
        "classifiers": fitted_classifiers,
        "features": features,
    }
