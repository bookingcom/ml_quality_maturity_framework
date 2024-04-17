from ml_quality.assessments import QualityAssessment
import pytest
import pandas as pd
import random
import string

TEST_MODEL_NAME = "test_model"
TEST_MODEL_CRITICALITY = "poc"
RANDOM_STR = ''.join(random.choices(string.ascii_lowercase, k=10))


@pytest.fixture()
def dataframe():
    d = {"a": [3, 3, 5], "b": [2, 6, 0], "c": [4, 1, 9]}
    return pd.DataFrame.from_dict(d)


@pytest.fixture()
def dataframe_with_dates():
    d = {
        "name": ["Mary", "Mary", "Mary", "Robert", "Jane"],
        "quality": ["good", "terrible", "bad", "ok", "great"],
        "date": ["2022-06-02", "2022-02-03", "2022-03-01", "2022-01-03", "2022-04-03"]
    }
    return pd.DataFrame.from_dict(d)


@pytest.fixture
def test_instance():
    return QualityAssessment(name=TEST_MODEL_NAME, business_criticality=TEST_MODEL_CRITICALITY)


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)

