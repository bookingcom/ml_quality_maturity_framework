import pytest
from collections import OrderedDict

from ml_quality.report_automation import (
    format_name,
    create_or_update_gaps_for_assessment,
    add_model_for_assessment,
    copy_inputs_for_model_family,
    prepopulate_gaps_dataframe,
    store_prepopulated_gaps,
    check_fulfillment_dict,
    get_fulfillment_per_subchar,
    answer_to_gaps,
    answer_to_bool,
)
from contextlib import contextmanager
from io import StringIO
import sys
from ml_quality.constants import QUESTION_AND_ANSWER_DICT, Gap, SUBCHARS_ALWAYS_MET, SUBCHARS_NOT_TO_BE_ASSESSED
from .conftest import RANDOM_STR
from os.path import exists
import os
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, format="%(filename)s - %(levelname)s - %(message)s")

REPORTS_INPUT_FOLDER = "../assessments/inputs"

@contextmanager
def replace_stdin(target):
    orig = sys.stdin
    sys.stdin = target
    yield
    sys.stdin = orig


def test_format_team_name():
    team_name = "My Team & al."
    excluded_chars = [" ", "&"]
    assert all([c not in format_name(team_name) for c in excluded_chars])


def get_raw_input(with_no=False):
    d = OrderedDict(
        model_name="model name",
        team_name="team name",
        business_criticality="poc",
        registry_link="link",
        git_link="link",
        fullon_link="link",
    )

    if with_no:
        value = "n"
    else:
        value = "y"

    d.update({k: value for k in QUESTION_AND_ANSWER_DICT.keys() if k.startswith("is_") or k.startswith("has_")})

    return d


def test_create_gaps_input():
    mocked_user_input = [i for i in get_raw_input().values()]

    with replace_stdin(StringIO("\n".join(mocked_user_input))):
        d = create_or_update_gaps_for_assessment()
        assert len(d) == len(QUESTION_AND_ANSWER_DICT) + 1


def test_create_gaps_input_create_false():
    mocked_user_input = [i for i in get_raw_input().values()]

    with replace_stdin(StringIO("\n".join(mocked_user_input))):
        d = create_or_update_gaps_for_assessment(create=False)
        assert len(d) == 3


def test_create_gaps_input_no_fullon_link():
    raw_input = get_raw_input()
    raw_input["fullon_link"] = ""
    mocked_user_input = [i for i in raw_input.values()]
    with replace_stdin(StringIO("\n".join(mocked_user_input))):
        d = create_or_update_gaps_for_assessment()
    assert d["is_fullon_older_than_6_months"] is None


def test_create_gaps_input_check_no_answer():
    mocked_user_input = [i for i in get_raw_input(True).values()]

    with replace_stdin(StringIO("\n".join(mocked_user_input))):
        d = create_or_update_gaps_for_assessment()
        assert "no" in d.values()


def test_add_model_for_assessment():
    model_info = {"model_name": "model_name", "business_criticality": "poc", "team_name": "team_name"}
    file_path = "models.csv"
    if exists(file_path):
        os.system(f"rm {file_path}")

    add_model_for_assessment(models_file=file_path, model_info=model_info)
    assert exists(file_path)

    model_info["model_name"] = "model_name2"
    add_model_for_assessment(models_file=file_path, model_info=model_info)


def test_prepopulate_gaps_dataframe_is_correct_requirements_met_fully():

    TEAM_NAME = "test"
    SERVING_LINK = "serving_link"
    FULLON_LINK = "fullon_link"

    gaps = prepopulate_gaps_dataframe(team_name=TEAM_NAME, is_understandable="fully", registry_link=SERVING_LINK,
                                      fullon_link=FULLON_LINK, is_fullon_older_than_6_months=True,
                                      has_comparison_with_baseline=True, is_efficient="fully", has_known_latency=True,
                                      is_resilient="fully", is_adaptable="fully", has_repeatable_pipeline=True,
                                      has_tested_code=True, has_monitoring=True, is_traceable="fully",
                                      is_explainable=True, has_applicable_standards=False,
                                      has_input_data_validation=True, is_vulnerable=True)
    assert len(gaps["gap_value"]) > 0
    assert gaps[gaps["sub_characteristic"] == "cost_effectiveness"]["gap_value"].values[0] == Gap.no.values[0]
    assert gaps[gaps["sub_characteristic"] == "usability"]["gap_value"].values[0] == Gap.no.values[0]
    assert gaps[gaps["sub_characteristic"] == "operability"]["gap_value"].values[0] == Gap.no.values[0]

    sub_chars_with_no_gap = gaps[gaps["gap_value"] == "no"]["sub_characteristic"].values

    for sub_char in sub_chars_with_no_gap:
        assert gaps[gaps["sub_characteristic"] == sub_char]["gap_value"].values[0] == Gap.no.values[0]


def test_prepopulate_gaps_dataframe_is_correct_requirements_nothing_met():
    TEAM_NAME = None
    SERVING_LINK = None
    FULLON_LINK = None

    gaps = prepopulate_gaps_dataframe(team_name=TEAM_NAME, registry_link=SERVING_LINK, fullon_link=FULLON_LINK,
                                      has_comparison_with_baseline=False, has_known_latency=False,
                                      has_repeatable_pipeline=False, has_tested_code=False, has_monitoring=False,
                                      is_explainable=False, has_applicable_standards=False,
                                      has_input_data_validation=False, is_vulnerable=False)

    sub_chars_with_large_gap = gaps[gaps["gap_value"] == "large"]["sub_characteristic"].values

    for sub_char in sub_chars_with_large_gap:
        assert gaps[gaps["sub_characteristic"] == sub_char]["gap_value"].values[0] == Gap.large.values[0]


def test_prepopulate_gaps_dataframe_is_correct_accuracy_partially_met_only_baseline():
    TEAM_NAME = None
    SERVING_LINK = None
    FULLON_LINK = None

    gaps = prepopulate_gaps_dataframe(team_name=TEAM_NAME, registry_link=SERVING_LINK, fullon_link=FULLON_LINK,
                                      has_comparison_with_baseline=True, has_known_latency=False,
                                      has_repeatable_pipeline=False, has_tested_code=False, has_monitoring=False,
                                      is_explainable=False, has_applicable_standards=False,
                                      has_input_data_validation=False, is_vulnerable=False)

    assert gaps[gaps["sub_characteristic"] == "accuracy"]["gap_value"].values[0] == Gap.small.values[0]


def test_prepopulate_gaps_dataframe_is_correct_accuracy_partially_met_only_input_validation():
    TEAM_NAME = None
    SERVING_LINK = None
    FULLON_LINK = None

    gaps = prepopulate_gaps_dataframe(team_name=TEAM_NAME, registry_link=SERVING_LINK, fullon_link=FULLON_LINK,
                                      has_comparison_with_baseline=False, has_known_latency=False,
                                      has_repeatable_pipeline=False, has_tested_code=False, has_monitoring=False,
                                      is_explainable=False, has_applicable_standards=False,
                                      has_input_data_validation=True, is_vulnerable=False)

    assert gaps[gaps["sub_characteristic"] == "accuracy"]["gap_value"].values[0] == Gap.small.values[0]


def test_prepopulate_gaps_dataframe_is_correct_requirements_nothing_met_left_blanks():

    gaps = prepopulate_gaps_dataframe()

    sub_chars_with_large_gap = list()
    for subchar, sub_char_dict in get_fulfillment_per_subchar().items():
        if sub_char_dict["large"] is not None and subchar not in SUBCHARS_ALWAYS_MET + SUBCHARS_NOT_TO_BE_ASSESSED:
            sub_chars_with_large_gap.append(subchar)

    for sub_char in sub_chars_with_large_gap:
        assert gaps[gaps["sub_characteristic"] == sub_char]["gap_value"].values[0] == Gap.large.values[0]


def test_prepopulate_gaps_dataframe_is_correct_requirements_partially_met():
    TEAM_NAME = None
    SERVING_LINK = None
    FULLON_LINK = "fullon_link"
    HAS_INPUT_DATA_VALIDATION = False
    HAS_COMPARISON_WITH_BASELINE = True

    gaps = prepopulate_gaps_dataframe(team_name=TEAM_NAME, is_understandable="partially", registry_link=SERVING_LINK,
                                      fullon_link=FULLON_LINK, is_fullon_older_than_6_months=True,
                                      has_comparison_with_baseline=HAS_COMPARISON_WITH_BASELINE,
                                      is_efficient="partially", has_known_latency=True, is_resilient="partially",
                                      is_adaptable="partially", has_repeatable_pipeline=True, has_tested_code=True,
                                      has_monitoring=True, is_traceable="partially", is_explainable=True,
                                      has_applicable_standards=False,
                                      has_input_data_validation=HAS_INPUT_DATA_VALIDATION, is_vulnerable=False)

    sub_chars_with_small_gap = list()
    for subchar, sub_char_dict in get_fulfillment_per_subchar(fullon_link=FULLON_LINK, serving_link=SERVING_LINK,
                                                              team_name=TEAM_NAME,
                                                              has_input_data_validation=HAS_INPUT_DATA_VALIDATION,
                                                              has_comparison_with_baseline=HAS_COMPARISON_WITH_BASELINE).items():
        if sub_char_dict["small"] is not None and subchar not in SUBCHARS_NOT_TO_BE_ASSESSED:
            sub_chars_with_small_gap.append(subchar)
    for sub_char in sub_chars_with_small_gap:
        assert gaps[gaps["sub_characteristic"] == sub_char]["gap_value"].values[0] == Gap.small.values[0]


def test_store_prepopulated_gaps_file_is_stored():
    store_prepopulated_gaps(model_name="test_model_name", csv_path="../assessments/inputs", team_name="team_name",
                            is_understandable="fully", registry_link="registry_link", has_comparison_with_baseline=True,
                            is_efficient="fully", fullon_link="fullon_link", is_resilient="fully",
                            has_known_latency=True, is_adaptable="fully", has_tested_code=True,
                            has_repeatable_pipeline=True, has_monitoring=True, is_traceable="fully",
                            is_explainable=True, has_applicable_standards=False, has_input_data_validation=False,
                            is_vulnerable=False)
    assert os.path.exists(path="../assessments/inputs/gaps_test_model_name.csv")
    os.remove("../assessments/inputs/gaps_test_model_name.csv")


def test_store_prepopulated_gaps_file_is_non_empty():
    os.system(f"mkdir -p {REPORTS_INPUT_FOLDER}")
    store_prepopulated_gaps(model_name="test_model_name", csv_path=REPORTS_INPUT_FOLDER, team_name="team_name",
                            is_understandable="fully", registry_link="registry_link", has_comparison_with_baseline=True,
                            is_efficient="fully", fullon_link="fullon_link", is_resilient="fully",
                            has_known_latency=True, is_adaptable="fully", has_tested_code=True,
                            has_repeatable_pipeline=True, has_monitoring=True, is_traceable="fully",
                            is_explainable=True, has_applicable_standards=False, has_input_data_validation=False,
                            is_vulnerable=False)

    gaps = pd.read_csv(f"{REPORTS_INPUT_FOLDER}/gaps_test_model_name.csv", sep=",", header=0)
    assert len(gaps["gap_value"]) > 10
    os.remove(f"{REPORTS_INPUT_FOLDER}/gaps_test_model_name.csv")


def test_prepopulate_gaps_inverted_answer():
    gaps = prepopulate_gaps_dataframe(team_name="team", is_adaptable="yes", is_vulnerable=True)
    assert gaps[gaps["sub_characteristic"] == "vulnerability"]["gap_value"].values[0] == "large"


def test_prepopulate_gaps_random_answers():
    with pytest.raises(ValueError, match="is_adaptable"):
        prepopulate_gaps_dataframe(is_adaptable=RANDOM_STR)


def test_prepopulate_gaps_dataframe_is_correct_requirements_wrong_input_all_values():

    import inspect

    for arg in inspect.signature(prepopulate_gaps_dataframe).parameters.keys():
        if arg in ["fullon_link", "team_name", "registry_link", "git_link", "months_since_last_fullon"]:
            continue
        else:
            with pytest.raises(ValueError):
                prepopulate_gaps_dataframe(**{arg: RANDOM_STR})


def test_prepopulate_gaps_dataframe_is_correct_requirements_wrong_input_2_values():

    with pytest.raises(ValueError):
        prepopulate_gaps_dataframe(has_applicable_standards="partially")


def test_check_fulfillment_dict():
    with pytest.raises(ValueError):
        check_fulfillment_dict({"quality": dict(no=None, small=None, large=None)})
    with pytest.raises(ValueError):
        check_fulfillment_dict({"accuracy": dict(no=None, small=None, huge=None)})


def test_get_fulfillment_per_subchar__accuracy_only_input_validation():
    fulfillments = get_fulfillment_per_subchar(has_input_data_validation=True, has_comparison_with_baseline=False)
    assert (
        fulfillments["accuracy"]["small"]
        == "There is input data validation, but there is no comparison with a simple baseline."
    )


def test_get_fulfillment_per_subchar__accuracy_only_comparison_baseline():
    fulfillments = get_fulfillment_per_subchar(has_input_data_validation=False, has_comparison_with_baseline=True)
    assert (
        fulfillments["accuracy"]["small"]
        == "The model outperforms a simple baseline, but there is no input data validation."
    )


def test_create_or_update_gaps_for_assessment_has_input_data_validation_reply():
    raw_input = get_raw_input()
    raw_input["has_input_data_validation"] = "n"
    mocked_user_input = [i for i in raw_input.values()]
    with replace_stdin(StringIO("\n".join(mocked_user_input))):
        create_or_update_gaps_for_assessment(output_folder="test_inputs")
    df_gaps = pd.read_csv("test_inputs/gaps_model_name.csv")
    assert df_gaps[df_gaps.sub_characteristic == "accuracy"]["gap_value"][0] == "small"


def test_create_or_update_gaps_for_assessment_wrong_answer():
    raw_input = get_raw_input()
    raw_input["is_efficient"] = RANDOM_STR
    mocked_user_input = [i for i in raw_input.values()]

    with replace_stdin(StringIO("\n".join(mocked_user_input))):
        with pytest.raises(EOFError):
            create_or_update_gaps_for_assessment(output_folder="test_inputs")


def test_answer_to_gaps():
    assert answer_to_gaps("yes") == "no"
    assert answer_to_gaps("no") == "large"


def test_answer_to_bool():
    assert answer_to_bool("no") is False
    assert answer_to_bool("yes") is True
    assert answer_to_bool(None) is False


def test_copy_inputs_for_model_family():
    FAMILY_NAME = "test_family"
    TEST_MODELS_TO_ASSESS_PATH = "test_data/test_models_to_assess.csv"
    ADDED_MODEL_NAME = "test_added_model_from_same_family"
    models_to_assess = pd.read_csv(TEST_MODELS_TO_ASSESS_PATH)

    assert ADDED_MODEL_NAME not in models_to_assess["name"].values
    copy_inputs_for_model_family(
        models_file=TEST_MODELS_TO_ASSESS_PATH, model_name=ADDED_MODEL_NAME, model_family=FAMILY_NAME
    )
    models_to_assess = pd.read_csv(TEST_MODELS_TO_ASSESS_PATH)
    assert ADDED_MODEL_NAME in models_to_assess["name"].to_list()
    models_to_assess.drop(models_to_assess[models_to_assess["name"] == ADDED_MODEL_NAME].index, inplace=True)
    models_to_assess.to_csv(TEST_MODELS_TO_ASSESS_PATH, index=False)


def test_copy_inputs_for_model_family_raises_file_error():
    with pytest.raises(FileNotFoundError):
        copy_inputs_for_model_family(models_file="wrong_file", model_name="test_model", model_family="test_family")


def test_copy_inputs_for_model_family_multiple_gap_files():
    with pytest.raises(ValueError, match="More than one gaps files found for the same model family:"):
        copy_inputs_for_model_family(
            models_file="test_data/test_models_to_assess_multiple_gaps_same_family.csv",
            model_name="test_model_2",
            model_family="test_family",
        )


def test_copy_inputs_for_model_family_multiple_criticalities():
    with pytest.raises(ValueError, match="More than one business criticalities found for the same model family:"):
        copy_inputs_for_model_family(
            models_file="test_data/test_models_to_assess_multiple_criticalities_same_family.csv",
            model_name="test_model_2",
            model_family="test_family",
        )


def test_copy_inputs_for_model_family_multiple_teams():
    with pytest.raises(ValueError, match="More than one team names found for the same model family:"):
        copy_inputs_for_model_family(
            models_file="test_data/test_models_to_assess_multiple_teams_same_family.csv",
            model_name="test_model_2",
            model_family="test_family",
        )
