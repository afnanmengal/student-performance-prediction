"""Leakage-conscious preprocessing and baseline model factories."""

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_grade_band(grades: pd.Series) -> pd.Series:
    """Map integer grades on a 0-20 scale to fixed performance bands."""
    if grades.isna().any() or not grades.between(0, 20).all():
        raise ValueError("Grades must be non-missing values between 0 and 20.")
    return pd.cut(
        grades,
        bins=[-1, 9, 13, 16, 20],
        labels=["at_risk", "developing", "proficient", "advanced"],
    )


def _preprocessor(
    numeric_features: Sequence[str], categorical_features: Sequence[str]
) -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(numeric_features)),
            ("categorical", categorical, list(categorical_features)),
        ],
        remainder="drop",
    )


def regression_pipelines(
    numeric_features: Sequence[str], categorical_features: Sequence[str]
) -> dict[str, Pipeline]:
    """Return regression baselines with all learned preprocessing in-pipeline."""
    models = {
        "ridge": Ridge(),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }
    return {
        name: Pipeline(
            [
                ("preprocess", _preprocessor(numeric_features, categorical_features)),
                ("model", model),
            ]
        )
        for name, model in models.items()
    }


def classification_pipelines(
    numeric_features: Sequence[str], categorical_features: Sequence[str]
) -> dict[str, Pipeline]:
    """Return multi-class baselines with all learned preprocessing in-pipeline."""
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    return {
        name: Pipeline(
            [
                ("preprocess", _preprocessor(numeric_features, categorical_features)),
                ("model", model),
            ]
        )
        for name, model in models.items()
    }
