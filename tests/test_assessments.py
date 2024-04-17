import os
import random
import shutil
import warnings
from datetime import datetime

import pandas as pd
import pytest
from pathlib import Path
import itertools

from .conftest import RANDOM_STR
from ml_quality.assessments import (
    maturity_levels,
    maturity_requirements,
    read_recommendations_from_csv,
    read_summary_from_csv,
    QualityAssessment,
    get_historical_summary,
    format_team_name,
)

from ml_quality.constants import (
    PLOTS_FOLDER,
    Gap,
    MAX_MATURITY,
    CHAR_ORDER,
    MATURITY_STANDARDS,
)
from ml_quality.utils import load_obj_from_pickle, check_date_format, read_practices_to_maturity_levels_from_csv
import re

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.set_option("display.max_columns", None)


def check_files_are_equal(f1_path: str, f2_path: str):
    f1 = open(f1_path, "r")
    f2 = open(f2_path, "r")
    assert f1.read() == f2.read()
    f1.close()
    f2.close()


def test_set_date(test_instance):
    new_date = "1900-01-01"
    assert test_instance.date == str(datetime.now().date())
    test_instance.date = new_date
    assert test_instance.date == new_date
    assert new_date in test_instance.model_folder


def test_set_date_wrong_format(test_instance):
    new_date = "1900-33-01"
    with pytest.raises(ValueError):
        test_instance.date = new_date


def test_summary_init(test_instance):
    summary_chars = list(test_instance.summary["characteristic"].unique())
    assert sorted(list(CHAR_ORDER.keys())) == sorted(list(summary_chars))


def test_set_date_wrong_format_init():
    with pytest.raises(ValueError, match="%Y-%m-%d"):
        QualityAssessment(name=RANDOM_STR, date="2000-99-01")


def test_get_recommendations_per_maturity_level_correct_results_critical(test_instance):
    test_instance.set_gap("usability", "large")
    test_instance.set_gap("fairness", "large")
    test_instance.set_gap("maintainability", "small")
    test_instance.set_gap("repeatability", "small")
    test_instance.set_gap("cost_effectiveness", "large")
    test_instance.set_gap("monitoring", "small")
    test_instance.set_gap("accuracy", "small")
    test_instance.set_gap("efficiency", "large")

    test_instance.business_criticality = "production_critical"

    practices_per_maturity_level = test_instance.get_recommendations_per_maturity_level()

    assert (
            len(practices_per_maturity_level)
            == MATURITY_STANDARDS[test_instance.business_criticality] - test_instance.maturity
    )
    assert "accuracy", "usability" in list(practices_per_maturity_level[3]["sub_characteristic"].values)
    assert "repeatability", "efficiency" in list(practices_per_maturity_level[4]["sub_characteristic"].values)
    assert all(
        x in ["maintainability", "monitoring", "cost_effectiveness", "efficiency", "fairness"]
        for x in list(practices_per_maturity_level[5]["sub_characteristic"].values)
    )


def test_get_recommendations_per_mat_level_until_expected_correct_results_non_critical(test_instance):
    test_instance.set_gap("usability", "large")
    test_instance.set_gap("resilience", "large")
    test_instance.set_gap("accuracy", "large")

    test_instance.business_criticality = "production_non_critical"

    practices_per_maturity_level = test_instance.get_recommendations_per_maturity_level()

    assert len(practices_per_maturity_level) == MAX_MATURITY
    assert list(practices_per_maturity_level[1]["sub_characteristic"].values) == ["accuracy"]
    assert list(practices_per_maturity_level[2]["sub_characteristic"].values) == []
    assert all(
        x in ["accuracy", "usability", "resilience"]
        for x in list(practices_per_maturity_level[3]["sub_characteristic"].values)
    )


def test_get_recommendations_per_maturity_level_already_in_expected_level(test_instance):
    test_instance.set_gap("fairness", "small")

    test_instance.business_criticality = "production_non_critical"

    practices_per_maturity_level = test_instance.get_recommendations_per_maturity_level()

    assert list(practices_per_maturity_level.keys()) == [5]
    assert practices_per_maturity_level[5]["is_expected_level"].values[0] == False


def test_get_flag_colours_per_mat_level_until_level_3(test_instance):
    test_instance.set_gap("usability", "large")
    test_instance.set_gap("availability", "large")
    test_instance.set_gap("accuracy", "large")
    test_instance.set_gap("explainability", "large")

    test_instance.business_criticality = "production_non_critical"

    practices_per_maturity_level = test_instance.get_recommendations_per_maturity_level()

    practices_w_color = test_instance.get_flag_colours_per_mat_level(
        practices_per_maturity_level=practices_per_maturity_level
    )

    assert practices_w_color[1]["flag_colour"].values[0] == "red"
    assert practices_w_color[3]["flag_colour"].values[0] == "orange"
    assert practices_w_color[4]["flag_colour"].values[0] == "yellow"


def test_get_flag_colours_per_mat_level_until_level_5(test_instance):
    test_instance.set_gap("usability", "large")
    test_instance.set_gap("availability", "large")
    test_instance.set_gap("accuracy", "large")
    test_instance.set_gap("readability", "large")

    test_instance.business_criticality = "production_critical"

    practices_per_maturity_level = test_instance.get_recommendations_per_maturity_level()

    practices_w_color = test_instance.get_flag_colours_per_mat_level(
        practices_per_maturity_level=practices_per_maturity_level
    )

    assert practices_w_color[1]["flag_colour"].values[0] == "red"
    assert practices_w_color[3]["flag_colour"].values[0] == "orange"
    assert practices_w_color[5]["flag_colour"].values[0] == "orange"


def test_get_flag_colours_per_mat_level_already_at_expected(test_instance):
    test_instance.set_gap("readability", "large")

    test_instance.business_criticality = "poc"

    practices_per_maturity_level = test_instance.get_recommendations_per_maturity_level()

    practices_w_color = test_instance.get_flag_colours_per_mat_level(
        practices_per_maturity_level=practices_per_maturity_level
    )

    assert practices_w_color[3]["flag_colour"].values[0] == "yellow"
    assert practices_w_color[5]["flag_colour"].values[0] == "yellow"


def test_prioritize_recommendations_within_maturity_level_2_practices_for_level_1(test_instance):
    test_instance.set_gap("usability", "large")
    test_instance.set_gap("availability", "large")
    test_instance.set_gap("accuracy", "large")

    gap_values_next_mat_level = test_instance.get_flags(maturity=test_instance.maturity)
    gaps_to_act_upon = gap_values_next_mat_level[
        gap_values_next_mat_level["gap_value"] > gap_values_next_mat_level["allowed_gap"]
        ]

    practices_to_fix_gaps = test_instance.get_practices_to_fix_specific_gaps(gaps_to_act_upon=gaps_to_act_upon)

    ranked_practices_for_next_level = test_instance.prioritize_recommendations_within_maturity_level(
        practices_to_fix_gaps=practices_to_fix_gaps
    )

    assert ranked_practices_for_next_level == [
        ("Deploy the model in a highly available & scalable serving system", 24.0),
        ("compare with a baseline", 7.15),
    ]


def test_prioritize_recommendations_within_maturity_level_3_practices_for_level_2(test_instance):
    test_instance.set_gap("usability", "large")
    test_instance.set_gap("modularity", "large")
    test_instance.set_gap("maintainability", "large")
    test_instance.set_gap("testability", "large")

    test_instance.business_criticality = "production_non_critical"
    gap_values_next_mat_level = test_instance.get_flags(maturity=2)
    gaps_to_act_upon = gap_values_next_mat_level[
        gap_values_next_mat_level["gap_value"] > gap_values_next_mat_level["allowed_gap"]
        ]

    practices_to_fix_gaps = test_instance.get_practices_to_fix_specific_gaps(gaps_to_act_upon=gaps_to_act_upon)

    ranked_practices_for_next_level = test_instance.prioritize_recommendations_within_maturity_level(
        practices_to_fix_gaps=practices_to_fix_gaps
    )

    assert ranked_practices_for_next_level == [
        ("code modularity and reusability", 50.54),
        ("automated tests", 34.85),
        ("code versioning", 0.0),
    ]


def test_add_rank_in_practices_2_practices_for_level_1(test_instance):
    data = [
        {
            "characteristic": "utility",
            "sub_characteristic": "accuracy",
            "gap_value": 2,
            "allowed_gap": 1,
            "flag": "red",
            "maturity": "level_1",
            "recommended_practice": "input data validation",
        },
        {
            "characteristic": "robustness",
            "sub_characteristic": "availability",
            "gap_value": 2,
            "allowed_gap": 0,
            "flag": "red",
            "maturity": "level_1",
            "recommended_practice": "Deploy the model in a highly available & scalable serving system",
        },
    ]
    practices_to_fix_gaps = pd.DataFrame(data=data)

    practices_w_rank = test_instance.add_rank_in_practices(practices_to_fix_gaps=practices_to_fix_gaps)

    assert len(practices_to_fix_gaps) == len(practices_w_rank)
    assert "rank" in practices_w_rank.columns
    assert (
            practices_w_rank[
                practices_w_rank["recommended_practice"]
                == "Deploy the model in a highly available & scalable serving system"
                ]["rank"].values[0]
            == 1
    )
    assert practices_w_rank[practices_w_rank["recommended_practice"] == "input data validation"]["rank"].values[0] == 2


def test_add_rank_in_practices_3_practices_for_level_2(test_instance):
    data = [
        {
            "characteristic": "modifiability",
            "sub_characteristic": "maintainability",
            "gap_value": 2,
            "allowed_gap": 1,
            "flag": "red",
            "maturity": "level_2",
            "recommended_practice": "code versioning",
        },
        {
            "characteristic": "modifiability",
            "sub_characteristic": "modularity",
            "gap_value": 2,
            "allowed_gap": 1,
            "flag": "red",
            "maturity": "level_2",
            "recommended_practice": "code modularity and reusability",
        },
        {
            "characteristic": "modifiability",
            "sub_characteristic": "testability",
            "gap_value": 2,
            "allowed_gap": 1,
            "flag": "red",
            "maturity": "level_2",
            "recommended_practice": "automated tests",
        },
    ]

    practices_to_fix_gaps = pd.DataFrame(data=data)

    practices_w_rank = test_instance.add_rank_in_practices(practices_to_fix_gaps=practices_to_fix_gaps)

    # assert that rank column is included
    assert "rank" in practices_w_rank.columns
    assert len(practices_to_fix_gaps) == len(practices_w_rank)

    assert (
            practices_w_rank[practices_w_rank["recommended_practice"] == "code modularity and reusability"][
                "rank"].values[
                0
            ]
            == 1
    )
    assert practices_w_rank[practices_w_rank["recommended_practice"] == "automated tests"]["rank"].values[0] == 2
    assert practices_w_rank[practices_w_rank["recommended_practice"] == "code versioning"]["rank"].values[0] == 3


def test_add_rank_in_practices_baseline_ranked_first(test_instance):
    data = [
        {
            "characteristic": "utility",
            "sub_characteristic": "accuracy",
            "gap_value": 2,
            "allowed_gap": 1,
            "flag": "red",
            "maturity": "level_1",
            "recommended_practice": "compare with a baseline",
        },
        {
            "characteristic": "modifiability",
            "sub_characteristic": "modularity",
            "gap_value": 2,
            "allowed_gap": 1,
            "flag": "red",
            "maturity": "level_2",
            "recommended_practice": "code modularity and reusability",
        },
        {
            "characteristic": "modifiability",
            "sub_characteristic": "testability",
            "gap_value": 2,
            "allowed_gap": 1,
            "flag": "red",
            "maturity": "level_2",
            "recommended_practice": "automated tests",
        },
    ]

    practices_to_fix_gaps = pd.DataFrame(data=data)

    practices_w_rank = test_instance.add_rank_in_practices(practices_to_fix_gaps=practices_to_fix_gaps)

    assert (
            practices_w_rank[practices_w_rank["recommended_practice"] == "compare with a baseline"]["rank"].values[
                0] == 1
    )
    assert (
            practices_w_rank[practices_w_rank["recommended_practice"] == "code modularity and reusability"][
                "rank"].values[
                0
            ]
            == 2
    )
    assert practices_w_rank[practices_w_rank["recommended_practice"] == "automated tests"]["rank"].values[0] == 3


def test_find_already_implemented_practices_all_implemented(test_instance):
    practices_to_mat_levels = read_practices_to_maturity_levels_from_csv()
    number_unique_practices = len(practices_to_mat_levels["practice_name"].unique())
    already_implemented_practices = test_instance.find_already_implemented_practices()
    number_already_implemented_practices = len(already_implemented_practices)
    assert number_already_implemented_practices == number_unique_practices


def test_find_already_implemented_practices_nothing_implemented(test_instance):
    test_instance.set_gap("accuracy", "large")
    test_instance.set_gap("effectiveness", "large")
    test_instance.set_gap("responsiveness", "large")
    test_instance.set_gap("cost_effectiveness", "large")
    test_instance.set_gap("efficiency", "large")
    test_instance.set_gap("availability", "large")
    test_instance.set_gap("resilience", "large")
    test_instance.set_gap("adaptability", "large")
    test_instance.set_gap("scalability", "large")
    test_instance.set_gap("maintainability", "large")
    test_instance.set_gap("modularity", "large")
    test_instance.set_gap("testability", "large")
    test_instance.set_gap("repeatability", "large")
    test_instance.set_gap("operability", "large")
    test_instance.set_gap("monitoring", "large")
    test_instance.set_gap("discoverability", "large")
    test_instance.set_gap("readability", "large")
    test_instance.set_gap("traceability", "large")
    test_instance.set_gap("understandability", "large")
    test_instance.set_gap("usability", "large")
    test_instance.set_gap("explainability", "large")
    test_instance.set_gap("fairness", "large")
    test_instance.set_gap("ownership", "large")
    test_instance.set_gap("standards_compliance", "large")
    test_instance.set_gap("vulnerability", "large")

    already_implemented_practices = test_instance.find_already_implemented_practices()

    assert already_implemented_practices == []


def test_find_already_implemented_practices_only_2_gap_0(test_instance):
    test_instance.set_gap("accuracy", "no")
    test_instance.set_gap("effectiveness", "large")
    test_instance.set_gap("responsiveness", "large")
    test_instance.set_gap("cost_effectiveness", "large")
    test_instance.set_gap("efficiency", "no")
    test_instance.set_gap("availability", "large")
    test_instance.set_gap("resilience", "large")
    test_instance.set_gap("adaptability", "large")
    test_instance.set_gap("scalability", "large")
    test_instance.set_gap("maintainability", "large")
    test_instance.set_gap("modularity", "large")
    test_instance.set_gap("testability", "large")
    test_instance.set_gap("repeatability", "large")
    test_instance.set_gap("operability", "large")
    test_instance.set_gap("monitoring", "large")
    test_instance.set_gap("discoverability", "large")
    test_instance.set_gap("readability", "large")
    test_instance.set_gap("traceability", "large")
    test_instance.set_gap("understandability", "large")
    test_instance.set_gap("usability", "large")
    test_instance.set_gap("explainability", "large")
    test_instance.set_gap("fairness", "large")
    test_instance.set_gap("ownership", "large")
    test_instance.set_gap("standards_compliance", "large")
    test_instance.set_gap("vulnerability", "large")

    already_implemented_practices = test_instance.find_already_implemented_practices()

    assert sorted(already_implemented_practices) == sorted(
        ["compare with a baseline", "optimize technical resources for training and inference", "input data validation"]
    )


def test_find_already_implemented_practices_only_2_gap_1(test_instance):
    test_instance.set_gap("accuracy", "small")
    test_instance.set_gap("effectiveness", "large")
    test_instance.set_gap("responsiveness", "large")
    test_instance.set_gap("cost_effectiveness", "large")
    test_instance.set_gap("efficiency", "large")
    test_instance.set_gap("availability", "large")
    test_instance.set_gap("resilience", "large")
    test_instance.set_gap("adaptability", "large")
    test_instance.set_gap("scalability", "large")
    test_instance.set_gap("maintainability", "large")
    test_instance.set_gap("modularity", "small")
    test_instance.set_gap("testability", "large")
    test_instance.set_gap("repeatability", "large")
    test_instance.set_gap("operability", "large")
    test_instance.set_gap("monitoring", "large")
    test_instance.set_gap("discoverability", "large")
    test_instance.set_gap("readability", "large")
    test_instance.set_gap("traceability", "large")
    test_instance.set_gap("understandability", "large")
    test_instance.set_gap("usability", "large")
    test_instance.set_gap("explainability", "large")
    test_instance.set_gap("fairness", "large")
    test_instance.set_gap("ownership", "large")
    test_instance.set_gap("standards_compliance", "large")
    test_instance.set_gap("vulnerability", "large")

    already_implemented_practices = test_instance.find_already_implemented_practices()

    assert sorted(already_implemented_practices) == sorted(
        ["compare with a baseline", "code modularity and reusability"]
    )


def test_find_already_implemented_practices_1_small_1_no_gap(test_instance):
    test_instance.set_gap("accuracy", "small")
    test_instance.set_gap("effectiveness", "large")
    test_instance.set_gap("responsiveness", "large")
    test_instance.set_gap("cost_effectiveness", "large")
    test_instance.set_gap("efficiency", "large")
    test_instance.set_gap("availability", "large")
    test_instance.set_gap("resilience", "large")
    test_instance.set_gap("adaptability", "large")
    test_instance.set_gap("scalability", "large")
    test_instance.set_gap("maintainability", "large")
    test_instance.set_gap("modularity", "large")
    test_instance.set_gap("testability", "large")
    test_instance.set_gap("repeatability", "large")
    test_instance.set_gap("operability", "large")
    test_instance.set_gap("monitoring", "no")
    test_instance.set_gap("discoverability", "large")
    test_instance.set_gap("readability", "large")
    test_instance.set_gap("traceability", "large")
    test_instance.set_gap("understandability", "large")
    test_instance.set_gap("usability", "large")
    test_instance.set_gap("explainability", "large")
    test_instance.set_gap("fairness", "large")
    test_instance.set_gap("ownership", "large")
    test_instance.set_gap("standards_compliance", "large")
    test_instance.set_gap("vulnerability", "large")

    already_implemented_practices = test_instance.find_already_implemented_practices()

    assert sorted(already_implemented_practices) == sorted(
        ["monitor feature drift", "model performance monitoring", "compare with a baseline"]
    )


def test_find_already_satisfied_subcharacteristics_expected_result(test_instance):
    already_implemented_practices = [
        "monitor feature drift",
        "model performance monitoring",
        "compare with a baseline",
        "input data validation",
    ]

    already_satisfied_subcharacteristics = test_instance.find_already_satisfied_subcharacteristics(
        already_implemented_practices=already_implemented_practices
    )

    expected_already_satisfied_subcharacteristics = {
        "monitor feature drift": "monitoring",
        "model performance monitoring": "monitoring",
        "compare with a baseline": "accuracy",
        "input data validation": "accuracy",
    }

    assert sorted(already_satisfied_subcharacteristics) == sorted(expected_already_satisfied_subcharacteristics)


def test_find_already_satisfied_subcharacteristics_fairness_is_excluded(test_instance):
    already_implemented_practices = [
        "bias assessment",
        "model performance monitoring",
        "compare with a baseline",
        "input data validation",
        "Conduct a responsible AI risk assessment",
    ]

    already_satisfied_subcharacteristics = test_instance.find_already_satisfied_subcharacteristics(
        already_implemented_practices=already_implemented_practices
    )

    expected_already_satisfied_subcharacteristics = {
        "model performance monitoring": "monitoring",
        "compare with a baseline": "accuracy",
        "input data validation": "accuracy",
        "Conduct a responsible AI risk assessment": "standards_compliance",
    }

    assert "fairness" not in already_satisfied_subcharacteristics.values()
    assert sorted(already_satisfied_subcharacteristics) == sorted(expected_already_satisfied_subcharacteristics)


def get_expected_maturity_from_data(sub_char: str, gap_value: int):
    expected_maturity = maturity_requirements[maturity_requirements.sub_characteristic == sub_char]
    levels_cols = [c for c in expected_maturity.columns if "level" in c]
    expected_maturity = expected_maturity[levels_cols].T
    expected_maturity.columns = ["gap"]
    expected_maturity = expected_maturity[expected_maturity["gap"] < gap_value].index[0].split("_")[1]
    return int(expected_maturity) - 1


def test_class_init_valid_summary(test_instance):
    assert isinstance(test_instance.summary, pd.DataFrame)


def test_class_update_summary_valid_columns(test_instance):
    with pytest.raises(ValueError):
        test_instance.summary = pd.DataFrame()


def test_get_characteristics(test_instance):
    assert all(test_instance.get_characteristics() == read_summary_from_csv()["characteristic"].unique())


def test_get_characteristics_with_sub(test_instance):
    col_names = ["characteristic", "sub_characteristic"]
    assert all(test_instance.get_characteristics_with_sub() == read_summary_from_csv()[col_names])


def test_class_update_summary_is_dataframe(test_instance):
    with pytest.raises(ValueError):
        test_instance.summary = 3


def test_class_update_summary_all_subchar(test_instance):
    new_summary = test_instance.summary.head(3)
    with pytest.raises(ValueError):
        test_instance.summary = new_summary


def test_class_initialization_valid_criticality(test_instance):
    with pytest.raises(ValueError, match="not recognized"):
        test_instance.business_criticality = RANDOM_STR


def test_set_gap_wrong_sub_char_input(test_instance):
    with pytest.raises(ValueError):
        test_instance.set_gap("blablah", "large")


def test_set_gap_wrong_gap_input(test_instance):
    with pytest.raises(ValueError):
        test_instance.set_gap("accuracy", "blabablah")


def test_set_gap_all_outputs(test_instance):
    for sub_char in test_instance.summary.sub_characteristic:
        for gap in Gap:
            gap = gap.values[1]
            test_instance.set_gap(sub_char=sub_char, gap=gap)
            df_sum = test_instance.summary
            assert int(df_sum[df_sum.sub_characteristic == sub_char]["gap_value"]) == gap


def test_set_gaps_expected_output(test_instance):
    all_sub_chars = test_instance.get_sub_characteristics()
    n = random.randint(2, len(all_sub_chars))
    sub_chars = random.sample(all_sub_chars, n)
    for gap_str in Gap:
        sub_char_with_gaps = {k: gap_str.value for k in sub_chars}
        test_instance.set_gaps(sub_char_with_gaps=sub_char_with_gaps)
        df_sum = test_instance.summary
        vals = df_sum[df_sum.sub_characteristic.isin(sub_chars)]["gap_value"].tolist()
        assert all([v == Gap(gap_str).values[1] for v in vals])


def test_set_model_folder(test_instance):
    with pytest.raises(ValueError):
        test_instance.model_folder = "my model folder"


def test_set_gaps_from_csv_manual_inputs(test_instance):
    csv_with_gap_path = f"{Path(__file__).parent}/test_data/test_gaps_manual.csv"
    test_instance.set_gaps_from_csv(csv_path=csv_with_gap_path)
    df_flags = test_instance.flags
    gap_val = df_flags[(df_flags["sub_characteristic"] == "accuracy") & (df_flags["maturity"] == "level_1")][
        "gap_value"
    ]
    assert gap_val.values[0] == 2

    gap_val = df_flags[(df_flags["sub_characteristic"] == "monitoring") & (df_flags["maturity"] == "level_1")][
        "gap_value"
    ]
    assert gap_val.values[0] == 1

    test_instance.create_pdf_report()


def test_set_gaps_from_csv_auto_inferred_inputs(test_instance):
    csv_with_gap_path = f"{Path(__file__).parent}/test_data/test_gaps_auto_inferred.csv"
    test_instance.set_gaps_from_csv(csv_path=csv_with_gap_path)

    assert (
            test_instance.flags[
                (test_instance.flags["sub_characteristic"] == "effectiveness")
                & (test_instance.flags["maturity"] == "level_1")
                ]["gap_value"].values[0]
            == 1
    )
    assert (
            test_instance.flags[
                (test_instance.flags["sub_characteristic"] == "responsiveness")
                & (test_instance.flags["maturity"] == "level_1")
                ]["gap_value"].values[0]
            == 2
    )

    assert test_instance.create_pdf_report()


def test_set_gaps_from_csv_works_with_null_gap(test_instance):
    csv_with_gap_path = f"{Path(__file__).parent}/test_data/test_gaps_auto_inferred_w_null.csv"
    test_instance.set_gaps_from_csv(csv_path=csv_with_gap_path)

    assert (
            test_instance.flags[
                (test_instance.flags["sub_characteristic"] == "effectiveness")
                & (test_instance.flags["maturity"] == "level_1")
                ]["gap_value"].values[0]
            == 2
    )
    assert (
            test_instance.flags[
                (test_instance.flags["sub_characteristic"] == "responsiveness")
                & (test_instance.flags["maturity"] == "level_1")
                ]["gap_value"].values[0]
            == 2
    )

    assert test_instance.create_pdf_report()


def test_set_all_levels_flags(test_instance):
    test_instance.set_gap("accuracy", "large")
    test_instance.set_gap("availability", "large")
    test_instance.set_gap("responsiveness", "large")
    test_instance.set_gap("efficiency", "small")
    all_flags = test_instance.flags
    assert len(all_flags.index) == len(test_instance.summary.index) * len(maturity_levels)


def test_find_maturity(test_instance):
    for sub_char in test_instance.get_sub_characteristics():
        for gap in ["small", "large"]:
            test_instance.set_gap(sub_char, gap)
            maturity = test_instance.find_maturity()
            expected_maturity = get_expected_maturity_from_data(sub_char, Gap(gap).values[1])
            assert maturity == expected_maturity
            test_instance.reset()


def test_private_set_maturity(test_instance):
    assert test_instance.maturity == test_instance.find_maturity()
    with pytest.raises(AttributeError):
        test_instance.maturity = 3


def test_private_set_scores(test_instance):
    with pytest.raises(AttributeError):
        test_instance.scores = "Bla"


def test_private_set_flags(test_instance):
    with pytest.raises(AttributeError):
        test_instance.flags = 3


def test_get_flags_filter_by_maturity(test_instance):
    assert len(test_instance.get_flags(maturity=3).index) == len(test_instance.summary.index)


def test_get_flags_filter_by_color(test_instance):
    for sub_char in test_instance.get_sub_characteristics():
        test_instance.set_gap(sub_char, "large")
        assert len(test_instance.get_flags(color="red", maturity=5).index) == 1
        test_instance.reset()


def test_compute_scores(test_instance):
    test_instance.set_gap("accuracy", 1)
    test_instance.set_gap("usability", 1)
    test_instance.set_gap("modularity", 2)
    assert len(test_instance.scores.index) == len(test_instance.get_characteristics())


def test_reset(test_instance):
    test_instance.set_gap("accuracy", "large")
    test_instance.set_gap("modularity", "large")
    red_flags = test_instance.flags[test_instance.flags.flag == "red"]
    assert len(red_flags.index) > 0
    test_instance.reset()
    red_flags = test_instance.flags[test_instance.flags.flag == "red"]
    assert len(red_flags.index) == 0


def test_plot_chart(test_instance):
    plot_path = test_instance.plot_chart(show_figure=False)
    assert os.path.isfile(plot_path)
    os.system(f"rm -r ./{PLOTS_FOLDER}")


def test_create_html_report_all_gaps(test_instance):
    test_instance.set_gap("usability", "large")
    test_instance.set_gap("accuracy", "large")
    test_instance.set_gap("effectiveness", "large")
    test_instance.set_gap("cost_effectiveness", "large")
    test_instance.set_gap("efficiency", "large")
    test_instance.set_gap("availability", "large")
    test_instance.set_gap("resilience", "large")
    test_instance.set_gap("adaptability", "large")
    test_instance.set_gap("scalability", "large")
    test_instance.set_gap("modularity", "large")
    test_instance.set_gap("adaptability", "large")
    test_instance.set_gap("operability", "large")
    test_instance.set_gap("monitoring", "large")
    test_instance.set_gap("discoverability", "large")
    test_instance.set_gap("readability", "large")
    test_instance.set_gap("traceability", "large")
    test_instance.set_gap("readability", "large")
    test_instance.set_gap("understandability", "large")
    test_instance.set_gap("explainability", "large")
    test_instance.set_gap("ownership", "large")
    test_instance.set_gap("standards_compliance", "large")
    test_instance.set_gap("vulnerability", "large")
    test_instance.set_gap("fairness", "large")
    test_instance.set_gap("maintainability", "large")
    test_instance.set_gap("repeatability", "large")

    test_instance.business_criticality = "production_non_critical"
    test_instance.model_family = "Test Model Family"

    report_file_path = test_instance.create_html_report(font_type="Verdana")
    plot_file_path = report_file_path.replace("report.html", "radar_chart.png")
    assert os.path.exists(f"./{report_file_path}")
    assert os.path.exists(f"./{plot_file_path}")
    shutil.rmtree("ml_quality_reports")


def test_create_html_report_report_created(test_instance):
    test_instance.set_gap("accuracy", "large")
    test_instance.set_gap("modularity", "small")
    test_instance.set_gap("monitoring", "large")
    test_instance.set_gap("responsiveness", "large")
    test_instance.set_gap("understandability", "small")

    test_instance.business_criticality = "production_critical"
    test_instance.set_flag_reason("accuracy", "No comparison with baseline and no input data validation.")
    test_instance.set_flag_reason("modularity", "The code is not fully modular.")
    test_instance.set_flag_reason("monitoring", "No ML performance and feature drift monitoring")
    test_instance.set_flag_reason("responsiveness", "Latency requirements are not known.")
    test_instance.set_flag_reason("understandability", "Documentation is only partial.")

    report_file_path = test_instance.create_html_report(font_type="Verdana")
    plot_file_path = report_file_path.replace("report.html", "radar_chart.png")
    assert os.path.exists(f"./{report_file_path}")
    assert os.path.exists(f"./{plot_file_path}")

    shutil.rmtree("ml_quality_reports")


# Check which practices are added as input in the prio framework!
def test_create_html_report_report_created_with_resilience_gap(test_instance):
    test_instance.set_gap("accuracy", "large")
    test_instance.set_gap("modularity", "small")
    test_instance.set_gap("monitoring", "large")
    test_instance.set_gap("responsiveness", "large")
    test_instance.set_gap("resilience", "large")
    test_instance.set_gap("understandability", "small")
    test_instance.set_gap("standards_compliance", "large")

    test_instance.business_criticality = "production_non_critical"
    test_instance.set_flag_reason("accuracy", "No comparison with baseline and no input data validation.")
    test_instance.set_flag_reason("modularity", "The code is not fully modular.")
    test_instance.set_flag_reason("monitoring", "No ML performance and feature drift monitoring")
    test_instance.set_flag_reason("responsiveness", "Latency requirements are not known.")
    test_instance.set_flag_reason("resilience", "More than 5 ML failures per quarter.")
    test_instance.set_flag_reason("understandability", "Documentation is only partial.")
    test_instance.set_flag_reason("standards_compliance", "Standards to be complied are not known.")

    report_file_path = test_instance.create_html_report(font_type="Verdana")
    plot_file_path = report_file_path.replace("report.html", "radar_chart.png")
    assert os.path.exists(f"./{report_file_path}")
    assert os.path.exists(f"./{plot_file_path}")

    shutil.rmtree("ml_quality_reports")


def test_class_property_custom_recommendation(test_instance):
    with pytest.raises(ValueError, match="remove"):
        test_instance.custom_recommendation = "Good"


def test_create_html_report_almost_perfect_model(test_instance):
    test_instance.set_gap("accuracy", "small")
    test_instance.create_html_report()

    shutil.rmtree("ml_quality_reports")


def test_create_html_report_2_gaps_for_level_3(test_instance):
    test_instance.business_criticality = "production_non_critical"

    subchars = {
        "accuracy": "small",
        "effectiveness": "large",  # gap
        "responsiveness": "no",
        "usability": "no",
        "cost_effectiveness": "large",
        "efficiency": "large",
        "availability": "no",
        "resilience": "small",
        "adaptability": "large",
        "scalability": "large",
        "maintainability": "small",
        "modularity": "no",
        "testability": "small",
        "repeatability": "small",
        "operability": "large",  # gap
        "monitoring": "no",
        "discoverability": "no",
        "traceability": "no",
        "understandability": "no",
        "explainability": "large",
        "fairness": "large",
        "ownership": "no",
        "standards_compliance": "no",
        "vulnerability": "large",
    }

    test_instance.set_gaps(sub_char_with_gaps=subchars)
    test_instance.create_pdf_report()
    scores = test_instance.scores

    assert scores[scores["characteristic"] == "economy"]["score"].values[0] == 0.0
    assert round(scores[scores["characteristic"] == "modifiability"]["score"].values[0], 3) == round(2 / 3, 3)


def test_create_pdf_report_report_created(test_instance):
    test_instance.set_gap("usability", "large")
    test_instance.set_gap("fairness", "large")
    test_instance.set_gap("maintainability", "small")
    test_instance.set_gap("repeatability", "small")
    test_instance.set_gap("cost_effectiveness", "large")
    test_instance.set_gap("monitoring", "small")
    test_instance.set_gap("accuracy", "small")
    test_instance.set_gap("efficiency", "large")
    test_instance.set_flag_reason("efficiency", "Efficiency is very bad")
    pdf_report_path = test_instance.create_pdf_report()
    assert os.path.exists(f"./{pdf_report_path}")
    shutil.rmtree("ml_quality_reports")


# def test_data_consistency(test_instance):
#     rec_input = sorted(test_instance.get_sub_characteristics())
#     rec = read_recommendations_from_csv()
#     rec_sub_char = sorted(rec["sub_characteristic"].unique())
#     assert len(list(set(rec_input) - set(rec_sub_char))) == 0


def test_create_pdf_report_all_good(test_instance):
    test_instance.create_pdf_report()
    shutil.rmtree("ml_quality_reports")


def test_save_and_reload_summary(test_instance):
    test_instance.set_gap("accuracy", "large")
    test_instance.set_gap("monitoring", "small")
    test_instance.set_gap("operability", "small")
    test_instance.report_file_name = "report_old"
    test_instance.create_pdf_report()
    summary_new = read_summary_from_csv(f"./{test_instance.model_folder}/summary.csv")
    test_instance.summary = summary_new
    test_instance.report_file_name = "report"
    test_instance.create_pdf_report()
    check_files_are_equal(
        f1_path=f"{test_instance.model_folder}/report_old.html", f2_path=f"{test_instance.model_folder}/report.html"
    )

    shutil.rmtree("ml_quality_reports")


def test_save_assessment(test_instance):
    test_instance.business_criticality = "production_critical"
    test_instance.set_gap("accuracy", "large")
    test_instance.set_gap("monitoring", "large")
    test_instance.set_gap("vulnerability", "large")
    test_instance.set_gap("operability", "large")
    test_instance.create_html_report()
    test_instance.save_to_pickle()
    new_instance = load_obj_from_pickle(f"{test_instance.model_folder}/{test_instance.pickle_file_name}.pickle")
    new_instance.report_file_name = "report_loaded"
    new_instance.create_html_report()
    check_files_are_equal(
        f1_path=f"{test_instance.model_folder}/report_loaded.html", f2_path=f"{test_instance.model_folder}/report.html"
    )

    shutil.rmtree("ml_quality_reports")


def test_version():
    with open("../ml_quality/_version.py", "r") as f:
        version_content = f.read()
    version = re.search(r"^__version__ = \"(.+)\"$", version_content, flags=re.MULTILINE).group(1)
    assert type(version) == str


def test_get_char_from_subchar(test_instance):
    test_instance.get_char_from_sub_char("discoverability")


def test_get_quality_score_perfect(test_instance):
    assert test_instance.quality_score == 100


def test_get_quality_score_zero(test_instance):
    subchars = {
        "accuracy": "large",
        "effectiveness": "large",
        "responsiveness": "large",
        "usability": "large",
        "cost_effectiveness": "large",
        "efficiency": "large",
        "availability": "large",
        "resilience": "large",
        "adaptability": "large",
        "scalability": "large",
        "maintainability": "large",
        "modularity": "large",
        "testability": "large",
        "repeatability": "large",
        "operability": "large",
        "monitoring": "large",
        "discoverability": "large",
        "traceability": "large",
        "understandability": "large",
        "explainability": "large",
        "fairness": "large",
        "ownership": "large",
        "standards_compliance": "large",
        "vulnerability": "large",
        "readability": "large",
    }

    test_instance.set_gaps(sub_char_with_gaps=subchars)

    assert test_instance.quality_score == 0


def test_compute_quality_score_perfect(test_instance):
    gap_values = [0, 0, 0]
    assert test_instance.compute_quality_score(gap_values=gap_values) == 100


def test_compute_quality_score_expected_values(test_instance):
    assert (
            test_instance.compute_quality_score(
                gap_values=[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            )
            == 90
    )
    assert (
            test_instance.compute_quality_score(
                gap_values=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            )
            == 98
    )
    assert (
            test_instance.compute_quality_score(
                gap_values=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
            )
            == 96
    )
    assert (
            test_instance.compute_quality_score(
                gap_values=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            )
            == 0
    )


def test_init_flag_reasons(test_instance):
    assert len(test_instance.flags_reasons) == len(test_instance.get_sub_characteristics())


def test_set_flag_reasons(test_instance):
    test_instance.set_flag_reason(sub_char="accuracy", reason="bad")
    print(test_instance.flags_reasons)


def test_set_flag_reasons_from_csv(test_instance):
    csv_with_gap_path = f"{Path(__file__).parent}/test_data/test_gaps_manual.csv"
    df = pd.read_csv(csv_with_gap_path)
    test_instance.set_flags_reasons_from_csv(
        csv_with_gap_path, sub_char_col_name="sub_characteristic", reason_col_name="reasoning"
    )
    assert df[df["sub_characteristic"] == "accuracy"]["reasoning"][0] == test_instance.flags_reasons["accuracy"]


def test_wrong_flag_reasons_update(test_instance):
    legit_update = {sc: "test" for sc in test_instance.get_sub_characteristics()}
    test_instance.flags_reasons = legit_update
    legit_update.pop("accuracy")
    with pytest.raises(ValueError, match="reason dictionary"):
        test_instance.flags_reasons = legit_update


def test_set_gaps_from_csv_with_reasons(test_instance):
    initial_reasons = list(set(test_instance.flags_reasons.values()))[0]
    assert initial_reasons == ""
    csv_with_gap_path = f"{Path(__file__).parent}/test_data/test_gaps_manual.csv"
    test_instance.set_gaps_from_csv(csv_path=csv_with_gap_path, reason_col_name="reasoning")
    reasons_from_csv = pd.read_csv(csv_with_gap_path)["reasoning"].values
    assert all([r in list(set(test_instance.flags_reasons.values())) for r in reasons_from_csv])


def test_get_historical_score():
    path = f"{Path(__file__).parent}/temp_quality_reports_test"
    historical_summary = get_historical_summary(path=path)
    assert isinstance(historical_summary, pd.DataFrame)
    {"maturity", "score"}.issubset(set(historical_summary.columns))
    with pytest.raises(ValueError, match="is not a directory"):
        get_historical_summary(path[:-1])
    os.system(f"mkdir -p {path}/test_model/blah")
    assert isinstance(historical_summary, pd.DataFrame)
    shutil.rmtree(f"{path}/test_model/blah")


def test_get_historical_score_correct_number_of_entries(test_instance):
    path = f"{Path(__file__).parent}/temp_quality_reports_test"
    n_rows = len(get_historical_summary(path=path).index)
    dates = list(itertools.chain(*[dirs for root, dirs, files in os.walk(path)]))
    n_assessments = len([d for d in dates if check_date_format(d, "%Y-%m-%d")])
    assert n_rows == n_assessments


def test_unsupported_image_format(test_instance):
    with pytest.raises(ValueError, match="image formats"):
        test_instance.radar_chart_file_format = ".xls"


def test_radar_chart_fname(test_instance):
    test_instance.name = "mouse"
    test_instance.radar_chart_file_name = "dog"
    assert "mouse" in test_instance.radar_chart_url
    assert "dog" in test_instance.radar_chart_url
    test_instance.report_file_name = "cow"
    assert "cow" in test_instance.report_url


def test_model_folder_update(test_instance):
    test_instance.model_folder = RANDOM_STR
    assert RANDOM_STR in test_instance.report_url
    assert RANDOM_STR in test_instance.radar_chart_url


def test_radar_chart_file_format_change(test_instance):
    test_instance.radar_chart_file_format = "jpeg"
    assert test_instance.radar_chart_url.split(".")[-1] == "jpeg"


def test_adding_html_list(test_instance):
    test_instance.set_gap("effectiveness", "small")
    test_instance.set_gap("cost_effectiveness", "small")
    test_instance.set_gap("testability", "small")
    test_instance.set_flag_reason(
        "testability",
        "The testability is not good because it should be better and like this it is not "
        "fully acceptable blah blah blah blah",
    )
    test_instance.create_pdf_report()


def test_resilience_yellow_flag(test_instance):
    test_instance.set_gap("resilience", "small")
    test_instance.business_criticality = "production_non_critical"

    practices_per_mat_level_w_colors = test_instance.get_recommendations_per_maturity_level()
    assert practices_per_mat_level_w_colors[5]["flag_colour"].values[0] == "yellow"


def test_resilience_orange_flag(test_instance):
    test_instance.set_gap("resilience", "small")
    test_instance.business_criticality = "production_critical"

    practices_per_mat_level_w_colors = test_instance.get_recommendations_per_maturity_level()

    assert practices_per_mat_level_w_colors[5]["flag_colour"].values[0] == "red"


def test_radar_char_default_file_format(test_instance):
    assert test_instance.radar_chart_file_format == "png"


def test_team_attribute(test_instance):
    team_name = "my_team"
    test_instance.team = team_name
    assert team_name in test_instance.report_url
    assert team_name in test_instance.create_pdf_report()
    assert team_name in test_instance.radar_chart_url
    shutil.rmtree("ml_quality_reports")


def test_get_strong_or_weak_characteristics_expected_result(test_instance):
    strengths = test_instance.get_strong_or_weak_characteristics(strong=True, quality_threshold=0.75)
    weaknesses = test_instance.get_strong_or_weak_characteristics(strong=False, quality_threshold=0.75)

    assert strengths == [
        "comprehensibility",
        "economy",
        "modifiability",
        "productionizability",
        "responsibility",
        "robustness",
        "utility",
    ]
    assert weaknesses == []


def test_get_strengths_summary_no_strengths(test_instance):
    score_per_characteristic = {
        "characteristic": [
            "comprehensibility",
            "economy",
            "modifiability",
            "productionizability",
            "responsibility",
            "robustness",
            "utility",
        ],
        "score": [0.0, 0.0, 0.67, 0.5, 0.4, 0.375, 0.625],
    }
    score_per_characteristic = pd.DataFrame.from_dict(score_per_characteristic)
    test_instance._scores = score_per_characteristic

    strengths_summary = test_instance.get_strengths_summary(
        current_maturity=1, expected_maturity=3, quality_threshold=0.75
    )

    assert strengths_summary == ""


def test_get_strengths_summary_1_strength(test_instance):
    score_per_characteristic = {
        "characteristic": [
            "comprehensibility",
            "economy",
            "modifiability",
            "productionizability",
            "responsibility",
            "robustness",
            "utility",
        ],
        "score": [1.0, 0.0, 0.67, 0.5, 0.4, 0.375, 0.625],
    }
    score_per_characteristic = pd.DataFrame.from_dict(score_per_characteristic)
    test_instance._scores = score_per_characteristic

    strengths_summary = test_instance.get_strengths_summary(
        current_maturity=1, expected_maturity=3, quality_threshold=0.75
    )

    assert strengths_summary == "The system is quite easy to comprehend."


def test_get_strengths_summary_3_strengths(test_instance):
    score_per_characteristic = {
        "characteristic": [
            "comprehensibility",
            "economy",
            "modifiability",
            "productionizability",
            "responsibility",
            "robustness",
            "utility",
        ],
        "score": [1.0, 1.0, 0.8, 0.5, 0.4, 0.375, 0.625],
    }
    score_per_characteristic = pd.DataFrame.from_dict(score_per_characteristic)
    test_instance._scores = score_per_characteristic

    strengths_summary = test_instance.get_strengths_summary(
        current_maturity=1, expected_maturity=3, quality_threshold=0.75
    )

    assert strengths_summary == "The system is quite easy to comprehend, profitable and modifiable."


def test_get_strengths_summary_5_strengths(test_instance):
    score_per_characteristic = {
        "characteristic": [
            "comprehensibility",
            "economy",
            "modifiability",
            "productionizability",
            "responsibility",
            "robustness",
            "utility",
        ],
        "score": [0.0, 1.0, 0.8, 0.9, 0.8, 1.0, 1.0],
    }
    score_per_characteristic = pd.DataFrame.from_dict(score_per_characteristic)
    test_instance._scores = score_per_characteristic

    strengths_summary = test_instance.get_strengths_summary(
        current_maturity=1, expected_maturity=3, quality_threshold=0.75
    )

    assert (
            strengths_summary
            == "The system is quite profitable, modifiable, easy to productionize, trustworthy, robust and useful."
    )


def test_get_strengths_summary_all_strengths_but_no_maturity(test_instance):
    score_per_characteristic = {
        "characteristic": [
            "comprehensibility",
            "economy",
            "modifiability",
            "productionizability",
            "responsibility",
            "robustness",
            "utility",
        ],
        "score": [0.8, 1.0, 0.8, 0.9, 0.8, 1.0, 1.0],
    }
    score_per_characteristic = pd.DataFrame.from_dict(score_per_characteristic)
    test_instance._scores = score_per_characteristic

    strengths_summary = test_instance.get_strengths_summary(
        current_maturity=4, expected_maturity=5, quality_threshold=0.75
    )

    assert (
            strengths_summary
            == "The system is of high quality, however the maturity is not yet at level 5 which is the expected one. "
               "This is because some prerequisites are missing, see below for more details."
    )


def test_get_strengths_summary_all_strengths_and_maturity(test_instance):
    score_per_characteristic = {
        "characteristic": [
            "comprehensibility",
            "economy",
            "modifiability",
            "productionizability",
            "responsibility",
            "robustness",
            "utility",
        ],
        "score": [0.8, 1.0, 0.8, 0.9, 0.8, 1.0, 1.0],
    }
    score_per_characteristic = pd.DataFrame.from_dict(score_per_characteristic)
    test_instance._scores = score_per_characteristic

    strengths_summary = test_instance.get_strengths_summary(
        current_maturity=3, expected_maturity=3, quality_threshold=0.75
    )

    assert strengths_summary == "The system is of very high quality and at the expected maturity. Well done!"


def test_get_strengths_summary_raise_error_wrong_columns(test_instance):
    score_per_characteristic = {
        "WRONG_COLUMN_NAME": [
            "comprehensibility",
            "economy",
            "modifiability",
            "productionizability",
            "responsibility",
            "robustness",
            "utility",
        ],
        "score": [0.8, 1.0, 0.8, 0.9, 0.8, 1.0, 1.0],
    }
    score_per_characteristic = pd.DataFrame.from_dict(score_per_characteristic)
    test_instance._scores = score_per_characteristic

    with pytest.raises(KeyError):
        test_instance.get_strengths_summary(current_maturity=3, expected_maturity=3, quality_threshold=0.75)


def test_get_weaknesses_summary_no_weaknesses(test_instance):
    score_per_characteristic = {
        "characteristic": [
            "comprehensibility",
            "economy",
            "modifiability",
            "productionizability",
            "responsibility",
            "robustness",
            "utility",
        ],
        "score": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    }

    score_per_characteristic = pd.DataFrame.from_dict(score_per_characteristic)
    test_instance._scores = score_per_characteristic

    weaknesses_summary = test_instance.get_weaknesses_summary(quality_threshold=0.75)

    assert weaknesses_summary == ""


def test_get_weaknesses_summary_has_1_weakness(test_instance):
    score_per_characteristic = {
        "characteristic": [
            "comprehensibility",
            "economy",
            "modifiability",
            "productionizability",
            "responsibility",
            "robustness",
            "utility",
        ],
        "score": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6],
    }

    score_per_characteristic = pd.DataFrame.from_dict(score_per_characteristic)
    test_instance._scores = score_per_characteristic

    weaknesses_summary = test_instance.get_weaknesses_summary(quality_threshold=0.75)

    assert weaknesses_summary == " However, its usefulness has room for improvement."


def test_get_weaknesses_summary_has_3_weaknesses(test_instance):
    score_per_characteristic = {
        "characteristic": [
            "comprehensibility",
            "economy",
            "modifiability",
            "productionizability",
            "responsibility",
            "robustness",
            "utility",
        ],
        "score": [0.8, 0.8, 0.6, 0.6, 0.6, 0.8, 0.8],
    }

    score_per_characteristic = pd.DataFrame.from_dict(score_per_characteristic)
    test_instance._scores = score_per_characteristic

    weaknesses_summary = test_instance.get_weaknesses_summary(quality_threshold=0.75)

    assert (
            weaknesses_summary
            == " However, quality aspects of modifiability, productionizability and trustworthiness can be improved. "
               "See below for more details."
    )


def test_get_weaknesses_summary_has_all_weaknesses(test_instance):
    score_per_characteristic = {
        "characteristic": [
            "comprehensibility",
            "economy",
            "modifiability",
            "productionizability",
            "responsibility",
            "robustness",
            "utility",
        ],
        "score": [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
    }

    score_per_characteristic = pd.DataFrame.from_dict(score_per_characteristic)
    test_instance._scores = score_per_characteristic

    weaknesses_summary = test_instance.get_weaknesses_summary(quality_threshold=0.75)

    assert (
            weaknesses_summary == "The system has room for improvement in multiple quality attributes. "
                                  "See below for more details."
    )


def test_get_weaknesses_summary_raise_error_wrong_columns(test_instance):
    score_per_characteristic = {
        "WRONG_COLUMN_NAME": [
            "comprehensibility",
            "economy",
            "modifiability",
            "productionizability",
            "responsibility",
            "robustness",
            "utility",
        ],
        "score": [0.8, 1.0, 0.8, 0.9, 0.8, 1.0, 1.0],
    }
    score_per_characteristic = pd.DataFrame.from_dict(score_per_characteristic)
    test_instance._scores = score_per_characteristic

    with pytest.raises(KeyError):
        test_instance.get_weaknesses_summary(quality_threshold=0.75)


def test_generate_strengths_and_weaknesses_summary_only_weaknesses(test_instance):
    score_per_characteristic = {
        "characteristic": [
            "comprehensibility",
            "economy",
            "modifiability",
            "productionizability",
            "responsibility",
            "robustness",
            "utility",
        ],
        "score": [0.6, 0.6, 0.6, 0.0, 0.0, 0.0, 0.6],
    }
    score_per_characteristic = pd.DataFrame.from_dict(score_per_characteristic)

    test_instance._scores = score_per_characteristic

    strengths_and_weaknesses_summary = test_instance.generate_strengths_and_weaknesses_summary(
        current_maturity=1, expected_maturity=3, quality_threshold=0.75
    )

    assert (
            strengths_and_weaknesses_summary == "The system has room for improvement in multiple quality attributes. "
                                                "See below for more details."
    )


def test_generate_strengths_and_weaknesses_summary_only_strengths(test_instance):
    score_per_characteristic = {
        "characteristic": [
            "comprehensibility",
            "economy",
            "modifiability",
            "productionizability",
            "responsibility",
            "robustness",
            "utility",
        ],
        "score": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    }
    score_per_characteristic = pd.DataFrame.from_dict(score_per_characteristic)
    test_instance._scores = score_per_characteristic

    strengths_and_weaknesses_summary = test_instance.generate_strengths_and_weaknesses_summary(
        current_maturity=3, expected_maturity=3, quality_threshold=0.75
    )

    assert (
            strengths_and_weaknesses_summary
            == "The system is of very high quality and at the expected maturity. Well done!"
    )


def test_generate_strengths_and_weaknesses_summary_1_strength(test_instance):
    score_per_characteristic = {
        "characteristic": [
            "comprehensibility",
            "economy",
            "modifiability",
            "productionizability",
            "responsibility",
            "robustness",
            "utility",
        ],
        "score": [0.6, 0.6, 0.6, 0.9, 0.0, 0.0, 0.6],
    }
    score_per_characteristic = pd.DataFrame.from_dict(score_per_characteristic)
    test_instance._scores = score_per_characteristic

    strengths_and_weaknesses_summary = test_instance.generate_strengths_and_weaknesses_summary(
        current_maturity=1, expected_maturity=3, quality_threshold=0.75
    )

    assert (
            strengths_and_weaknesses_summary
            == "The system is quite easy to productionize. However, quality aspects of comprehensibility, profitability, "
               "modifiability, trustworthiness, robustness and usefulness can be improved. See below for more details."
    )


def test_generate_strengths_and_weaknesses_summary_1_weakness(test_instance):
    score_per_characteristic = {
        "characteristic": [
            "comprehensibility",
            "economy",
            "modifiability",
            "productionizability",
            "responsibility",
            "robustness",
            "utility",
        ],
        "score": [0.8, 0.8, 0.6, 0.9, 0.8, 0.8, 0.8],
    }
    score_per_characteristic = pd.DataFrame.from_dict(score_per_characteristic)
    test_instance._scores = score_per_characteristic

    strengths_and_weaknesses_summary = test_instance.generate_strengths_and_weaknesses_summary(
        current_maturity=1, expected_maturity=3, quality_threshold=0.75
    )

    assert (
            strengths_and_weaknesses_summary
            == "The system is quite easy to comprehend, profitable, easy to productionize, trustworthy, robust and "
               "useful. However, its modifiability has room for improvement."
    )


def test_generate_strengths_and_weaknesses_summary_end_to_end_test(test_instance):
    test_instance.set_gap("accuracy", "large")
    test_instance.set_gap("usability", "large")
    result_str = (
        "The system is quite easy to comprehend, profitable, modifiable, "
        "easy to productionize, trustworthy and robust. However, its usefulness has room for improvement."
    )
    assert (
            test_instance.generate_strengths_and_weaknesses_summary(
                current_maturity=1,
                expected_maturity=3,
            )
            == result_str
    )


def test_check_if_practices_are_scored_no_warning_raised(test_instance, recwarn):
    practices_to_be_recommended = ["input data validation"]

    test_instance.check_if_practices_are_scored(practices_to_be_recommended=practices_to_be_recommended)
    assert len(recwarn) == 0


def test_check_if_practices_are_scored_warning_is_raised(test_instance):
    practices_to_be_recommended = ["NOT A SCORED PRACTICE"]

    with pytest.raises(Warning):
        test_instance.check_if_practices_are_scored(practices_to_be_recommended=practices_to_be_recommended)


def test_format_team_name():
    assert "-" not in format_team_name("aaa-aaa")
    assert "\"" not in format_team_name("ML Foundations - \"Eng Team\"")
