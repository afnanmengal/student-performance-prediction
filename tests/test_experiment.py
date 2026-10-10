import pandas as pd
import pytest

from student_performance.experiment import load_student_data
from student_performance.pipelines import make_grade_band


def test_grade_bands_reject_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="between 0 and 20"):
        make_grade_band(pd.Series([21]))


def test_student_loader_checks_final_grade(tmp_path) -> None:
    path = tmp_path / "student.csv"
    pd.DataFrame({"studytime": [1, 2], "G3": [10, 15]}).to_csv(path, index=False)

    loaded = load_student_data(path)

    assert loaded["G3"].tolist() == [10, 15]
