"""Interactive student grade regression and performance-band classifier."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from student_performance.experiment import load_student_data, run_student_experiment


st.set_page_config(page_title="Student Performance Lab", layout="wide")
st.title("Student Performance Lab")
st.caption(
    "Explore early-support grade estimates and performance bands. "
    "This educational model is not for high-stakes decisions about students."
)

default_dataset = PROJECT_ROOT / "data" / "raw" / "student-mat.csv"
uploaded_file = st.file_uploader("Student CSV", type=["csv"])
if uploaded_file is not None:
    try:
        data = load_student_data(uploaded_file)
        source_label = uploaded_file.name
    except ValueError as error:
        st.error(str(error))
        st.stop()
elif default_dataset.exists():
    try:
        data = load_student_data(default_dataset)
        source_label = "Kaggle Math-course data"
    except ValueError as error:
        st.error(str(error))
        st.stop()
else:
    st.info("Upload a student CSV containing numeric final-grade column G3 to begin.")
    st.stop()

features = data.drop(columns=[column for column in ("G1", "G2", "G3") if column in data])
overview = st.columns(4)
overview[0].metric("Students", f"{len(data):,}")
overview[1].metric("Predictors", f"{len(features.columns):,}")
overview[2].metric("Mean final grade", f"{data['G3'].mean():.1f} / 20")
overview[3].metric("At-risk band", f"{(data['G3'] <= 9).mean():.1%}")
st.caption(f"Dataset: {source_label}. G1 and G2 are excluded from model inputs.")

with st.expander("Dataset preview", expanded=False):
    st.dataframe(data.head(12), use_container_width=True)

if st.button("Run regression and classification benchmarks", type="primary"):
    with st.spinner("Training cross-validated models…"):
        try:
            st.session_state["student_results"] = run_student_experiment(data)
        except ValueError as error:
            st.error(str(error))
            st.stop()

results = st.session_state.get("student_results")
if results:
    regression_tab, classification_tab, prediction_tab = st.tabs(
        ["Regression", "Performance bands", "Student prediction"]
    )
    with regression_tab:
        st.dataframe(
            results["regression"].style.format(
                {
                    "cv_mae": "{:.2f}",
                    "test_mae": "{:.2f}",
                    "test_rmse": "{:.2f}",
                    "test_r2": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        selected_regressor = st.selectbox(
            "Regression model", list(results["regressors"]), key="student_regressor"
        )
    with classification_tab:
        st.dataframe(
            results["classification"].style.format(
                {"cv_macro_f1": "{:.3f}", "test_accuracy": "{:.3f}", "test_macro_f1": "{:.3f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        selected_classifier = st.selectbox(
            "Classification model", list(results["classifiers"]), key="student_classifier"
        )

    with prediction_tab:
        st.caption("Set a student profile; defaults use training-data medians/modes.")
        numeric_features = features.select_dtypes(include="number").columns.tolist()
        categorical_features = [
            column for column in features.columns if column not in numeric_features
        ]
        numeric_defaults = features[numeric_features].median()
        prediction_values: dict[str, float | int | str] = {}
        with st.form("student_prediction"):
            input_columns = st.columns(3)
            for index, column in enumerate(features.columns):
                with input_columns[index % 3]:
                    if column in categorical_features:
                        choices = sorted(features[column].dropna().unique().tolist())
                        prediction_values[column] = st.selectbox(
                            column, choices, key=f"student_input_{column}"
                        )
                    else:
                        prediction_values[column] = st.number_input(
                            column,
                            value=float(numeric_defaults[column]),
                            key=f"student_input_{column}",
                        )
            predict_clicked = st.form_submit_button("Estimate final grade and band")

        if predict_clicked:
            student = pd.DataFrame([prediction_values], columns=features.columns)
            grade = float(results["regressors"][selected_regressor].predict(student)[0])
            band = str(results["classifiers"][selected_classifier].predict(student)[0])
            prediction_columns = st.columns(2)
            prediction_columns[0].metric("Estimated G3", f"{grade:.1f} / 20")
            prediction_columns[1].metric(
                "Estimated performance band", band.replace("_", " ").title()
            )

st.divider()
st.caption(
    "Bands: 0-9 at-risk, 10-13 developing, 14-16 proficient, 17-20 advanced. "
    "The advanced class is small; predictions need human context and should not "
    "determine student outcomes."
)
