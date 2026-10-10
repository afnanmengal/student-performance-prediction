import pandas as pd

from student_performance.pipelines import make_grade_band, regression_pipelines


def test_grade_bands_follow_fixed_cut_points():
    grades = pd.Series([0, 9, 10, 13, 14, 16, 17, 20])

    assert make_grade_band(grades).astype(str).tolist() == [
        "at_risk",
        "at_risk",
        "developing",
        "developing",
        "proficient",
        "proficient",
        "advanced",
        "advanced",
    ]


def test_regression_pipelines_fit_mixed_data():
    features = pd.DataFrame({"study_hours": [1, 2, 3, 4], "school": ["A", "B", "A", "B"]})
    targets = [8, 10, 14, 17]

    model = regression_pipelines(["study_hours"], ["school"])["ridge"]
    model.fit(features, targets)

    assert len(model.predict(features)) == len(features)
