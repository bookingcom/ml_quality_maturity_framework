import os.path

from ml_quality.utils import (
    set_pandas_value,
    plot_radar_chart,
    generate_report_text,
    dataframe_anti_join,
    get_column_value_from_row,
    get_pkg_version,
    add_is_max_col,
    check_date_format,
    propagate_last_value,
    compute_statistics_of_given_column,
    get_html_recommendation_list,
    find_the_next_mat_level_to_reach,
    get_dict_w_links_per_sub_char,
    get_text_for_satisfied_subchars,
    get_subchars_with_multiple_levels,
)

from ml_quality.constants import MAX_MATURITY, MATURITY_STANDARDS
from .conftest import RANDOM_STR
import pytest
import pandas as pd
import html5lib
import numpy as np

pd.set_option("display.max_columns", None)


def test_set_pandas_value(dataframe: pd.DataFrame):
    index_freq = pd.DataFrame(dataframe.groupby("a")["a"].count())
    multiple_idx = index_freq[index_freq.a > 1].index.tolist()[0]
    unique_idx = index_freq[index_freq.a == 1].index.tolist()[0]
    updated_val = RANDOM_STR

    new_df = set_pandas_value(dataframe, index=("a", unique_idx), column=("b", updated_val))
    assert new_df[new_df.a == unique_idx]["b"].tolist()[0] == updated_val

    with pytest.raises(ValueError):
        set_pandas_value(dataframe, index=("a", multiple_idx), column=("b", updated_val))


def test_plot_radar_chart(dataframe: pd.DataFrame):
    os.system("rm -rf ./plots")
    assert not os.path.isfile("./plots/radar_chart.png")
    path = plot_radar_chart(dataframe, r="a", theta="b", show_figure=False, save_figure=True)
    assert os.path.isfile("./plots/radar_chart.png")
    assert path == "plots/radar_chart.png"
    os.system("rm -rf ./plots")


def test_plot_radar_chart_show_figure(dataframe: pd.DataFrame):
    os.system("rm -rf ./plots")
    assert not os.path.isfile("./plots/radar_chart.png")
    plot_radar_chart(dataframe, r="a", theta="b")


def test_generate_report_text_is_html(dataframe):
    path = plot_radar_chart(dataframe, r="b", theta="b", show_figure=False, save_figure=True)

    text = generate_report_text(
        model_name="model",
        model_criticality="poc",
        model_maturity=4,
        model_family=None,
        quality_score=76,
        radar_chart_path=path,
        reason_dict={"char": "reason"},
        recommendation_txt=None,
        summary="The Summary",
        recommendations=dict(),
        already_satisfied_subchars={},
    )

    html5parser = html5lib.HTMLParser(strict=True)
    html5parser.parse(text)


def test_generate_report_text_expected_recommendations(test_instance, dataframe):
    text = generate_report_text(
        model_name="model",
        model_criticality="poc",
        model_maturity=test_instance.maturity,
        model_family=test_instance.model_family,
        quality_score=test_instance.quality_score,
        radar_chart_path=None,
        reason_dict=test_instance.flags_reasons,
        recommendation_txt=None,
        summary=test_instance.generate_strengths_and_weaknesses_summary(
            current_maturity=test_instance.maturity,
            expected_maturity=MATURITY_STANDARDS[test_instance.business_criticality],
        ),
        recommendations=dict(),
        already_satisfied_subchars={},
    )
    print(text)
    assert "Congratulations" in text
    assert "Learn" in text
    assert "Model family" not in text
    assert "Models of the same family share the same quality assessment" not in text

    test_instance.set_gaps(sub_char_with_gaps={"monitoring": 1, "testability": 2})
    test_instance.set_gaps(sub_char_with_gaps={"accuracy": 1})
    test_instance.business_criticality = "production_critical"
    test_instance.model_family = "test_family"

    practices_per_mat_level_w_colors = test_instance.get_recommendations_per_maturity_level()
    text = generate_report_text(
        model_name="model",
        model_criticality="production_critical",
        model_maturity=test_instance.maturity,
        model_family=test_instance.model_family,
        quality_score=test_instance.quality_score,
        radar_chart_path=None,
        reason_dict=test_instance.flags_reasons,
        recommendation_txt=None,
        summary=test_instance.generate_strengths_and_weaknesses_summary(
            current_maturity=test_instance.maturity,
            expected_maturity=MATURITY_STANDARDS[test_instance.business_criticality],
        ),
        recommendations=practices_per_mat_level_w_colors,
        already_satisfied_subchars={},
    )
    assert "Bring your model" in text
    assert "Fix" in text
    assert "Model family" in text
    assert "Models of the same family share the same quality assessment" in text


def test_generate_report_text_confirm_sentence_brainstorms(test_instance, dataframe):
    test_instance.set_gaps(sub_char_with_gaps={"monitoring": "small", "testability": "large"})
    test_instance.set_gaps(sub_char_with_gaps={"accuracy": "large"})

    practices_per_mat_level_w_colors = test_instance.get_recommendations_per_maturity_level()

    text = generate_report_text(
        model_name="model",
        model_criticality="poc",
        model_maturity=test_instance.maturity,
        model_family=test_instance.model_family,
        quality_score=test_instance.quality_score,
        radar_chart_path=None,
        reason_dict=test_instance.flags_reasons,
        recommendation_txt=None,
        summary="The Summary",
        recommendations=practices_per_mat_level_w_colors,
        already_satisfied_subchars={},
    )

    assert "If the aforementioned practices are not enough" in text


def test_no_recommendation_list_for_green_flags(test_instance):
    practices_per_mat_level_w_colors = test_instance.get_recommendations_per_maturity_level()

    assert practices_per_mat_level_w_colors == {}


def test_get_html_recommendation_list_recommendations_for_level_3(test_instance):
    test_instance.set_gap("usability", "large")
    test_instance.set_gap("accuracy", "large")
    test_instance.set_gap("fairness", "large")
    test_instance.set_gap("maintainability", "large")
    test_instance.set_gap("repeatability", "small")

    test_instance.business_criticality = "production_non_critical"

    practices_per_maturity_level = test_instance.get_recommendations_per_maturity_level()

    test_instance.flags_reasons["accuracy"] = "There is no input data validation"
    test_instance.flags_reasons["usability"] = "The model is not deployed on a usable platform"

    # create a sample dataframe to input in the get_html_recommendation
    recommendation_text = (
        get_html_recommendation_list(
            recommendations_per_mat_level=practices_per_maturity_level, reason_dict=test_instance.flags_reasons
        )
        .replace("\n", "")
        .replace(" ", "")
    )

    print(recommendation_text)
    sentences_to_check = [
        '<span style="color: red; "><b>Accuracy</b></span> - step 1',
        '<span style="color: orange; "><b>Maintainability</b></span> - step 1',
        '<span style="color: Gold; "><b>Repeatability</b></span> - step 2',
        '<span style="color: Gold; "><b>Fairness</b></span>',
        '<span style="color: Gold; "><b>Maintainability</b></span> - step 2',
    ]
    for sentence in sentences_to_check:
        s = sentence.replace(" ", "")
        assert s in recommendation_text


def test_get_html_recommendation_list_is_empty(test_instance):
    text = get_html_recommendation_list(recommendations_per_mat_level={}, reason_dict={})

    assert text == ""


def test_custom_recommendation_printed():
    custom_rec = "blablahblahblahblah"
    text = generate_report_text(
        model_name="model",
        model_criticality="poc",
        model_maturity=4,
        model_family=None,
        quality_score=93,
        radar_chart_path=None,
        reason_dict={"blah": "blah"},
        recommendation_txt=custom_rec,
        summary="The Summary",
        recommendations={},
        already_satisfied_subchars={},
    )
    assert custom_rec in text


def test_generate_report_text_custom_recommendation_printed():
    custom_rec = "blablahblahblahblah"
    text = generate_report_text(
        model_name="model",
        model_criticality="poc",
        model_maturity=4,
        model_family=None,
        quality_score=93,
        radar_chart_path=None,
        reason_dict=None,
        recommendation_txt=custom_rec,
        summary="The Summary",
        recommendations={},
        already_satisfied_subchars={},
    )
    assert custom_rec in text


def test_generate_report_text_congrats_message_darkgreen():
    text = generate_report_text(
        model_name="model",
        model_criticality="poc",
        model_maturity=MAX_MATURITY,
        model_family=None,
        quality_score=93,
        radar_chart_path=None,
        reason_dict=None,
        recommendation_txt=None,
        summary="The Summary",
        recommendations={},
        already_satisfied_subchars={},
    )
    assert "darkgreen" in text


def test_generate_report_text_expected_recommendations_multiple_subchars(test_instance):
    test_instance.set_gaps(
        sub_char_with_gaps={
            "monitoring": 1,
            "testability": 2,
            "accuracy": 1,
            "scalability": 0,
            "effectiveness": 2,
            "usability": 2,
        }
    )
    test_instance.business_criticality = "production_critical"
    recommendations = test_instance.get_recommendations_per_maturity_level()

    text = (
        generate_report_text(
            model_name="model",
            model_criticality=test_instance.business_criticality,
            model_maturity=test_instance.maturity,
            model_family=None,
            quality_score=93,
            radar_chart_path=None,
            reason_dict=None,
            recommendation_txt=None,
            summary="The Summary",
            recommendations=recommendations,
            already_satisfied_subchars={},
        )
        .replace("\n", "")
        .replace(" ", "")
    )

    sentences_to_test = [
        '<span style="color: red; "><b>Testability</b></span> - step 1',
        '<span style="color: orange; "><b>Effectiveness</b></span> - step 1',
        '<span style="color: orange; "><b>Accuracy</b></span> - step 2',
        '<span style="color: orange; "><b>Effectiveness</b></span> - step 2',
        '<span style="color: orange; "><b>Monitoring</b></span> - step 2',
    ]
    for sentence in sentences_to_test:
        s = sentence.replace("\n", "").replace(" ", "")
        assert s in text


def test_generate_report_text_immature_critical_model(test_instance):
    test_instance.business_criticality = "production_critical"
    recommendations = test_instance.get_recommendations_per_maturity_level()

    txt = generate_report_text(
        model_name="model",
        model_criticality=test_instance.business_criticality,
        model_maturity=4,
        model_family=None,
        quality_score=test_instance.quality_score,
        radar_chart_path=None,
        reason_dict=test_instance.flags_reasons,
        recommendation_txt=None,
        summary="The Summary",
        recommendations=recommendations,
        already_satisfied_subchars={},
    )
    assert "darkred" in txt


def test_dataframe_anti_join(dataframe):
    assert dataframe_anti_join(dataframe, dataframe, on=["a"]).empty


def test_dataframe_anti_join_empty_df(dataframe):
    assert all(dataframe == dataframe_anti_join(dataframe, pd.DataFrame(), on=["a"]))
    assert dataframe_anti_join(pd.DataFrame(), dataframe, on=["a"]).empty


def test_column_value_from_row(dataframe):
    assert len([get_column_value_from_row(dataframe, filter_col="a", value_col="b", filter_val=5)]) == 1
    with pytest.raises(ValueError):
        get_column_value_from_row(dataframe, filter_col="a", value_col="b", filter_val=1)


def test_get_pkg_version():
    assert isinstance(get_pkg_version(version_file_path="../ml_quality/_version.py"), str)


def test_add_is_max_col_is_maximum(dataframe):
    df = add_is_max_col(dataframe, id_cols=["a"], sort_col="b")
    max_sort_col_val = max(df[df.a == 3]["b"].values)
    assert df[(df.a == 3) & (df.b == max_sort_col_val)]["max"].values[0]


def test_check_date_format():
    assert check_date_format(date_str="2022-03-01", date_fmt="%Y-%m-%d")
    assert not check_date_format(date_str="2022-13-01", date_fmt="%Y-%m-%d")
    assert not check_date_format(date_str="2022-10-01", date_fmt="%y-%m-%d")


def test_propagate_last_value(dataframe_with_dates):
    print("\n")
    print(dataframe_with_dates.head(10))
    df_new = propagate_last_value(
        df=dataframe_with_dates, sort_cols=["name", "date"], groupby_col="name", col_to_propagate="quality"
    )
    print("\n")
    print(df_new.head(10))
    assert all(df_new.groupby("name").quality.nunique().values) == 1


def test_find_the_next_mat_level_to_reach_1_empty(test_instance):
    practices_per_maturity_level = {
        1: pd.DataFrame(data=["data"]),
        2: pd.DataFrame(),
        3: pd.DataFrame(data=["data"]),
        4: pd.DataFrame(data=["data"]),
        5: pd.DataFrame(data=["data"]),
    }

    modified_dictionary = find_the_next_mat_level_to_reach(recommendations_per_mat_level=practices_per_maturity_level)

    assert modified_dictionary == {1: 2, 2: 2, 3: 3, 4: 4, 5: 5}


def test_find_the_next_mat_level_to_reach_3_empty(test_instance):
    practices_per_maturity_level = {
        1: pd.DataFrame(data=["data"]),
        2: pd.DataFrame(),
        3: pd.DataFrame(),
        4: pd.DataFrame(),
        5: pd.DataFrame(data=["data"]),
    }

    modified_dictionary = find_the_next_mat_level_to_reach(recommendations_per_mat_level=practices_per_maturity_level)

    assert modified_dictionary == {1: 4, 2: 4, 3: 4, 4: 4, 5: 5}


def test_find_the_next_mat_level_to_reach_first_2_only(test_instance):
    practices_per_maturity_level = {
        1: pd.DataFrame(data=["data"]),
        2: pd.DataFrame(data=["data"]),
        3: pd.DataFrame(),
        4: pd.DataFrame(),
        5: pd.DataFrame(),
    }

    modified_dictionary = find_the_next_mat_level_to_reach(recommendations_per_mat_level=practices_per_maturity_level)

    assert modified_dictionary == {1: 1, 2: 5, 3: 5, 4: 5, 5: 5}


def test_find_the_next_mat_level_to_reach_first_and_third_only(test_instance):
    practices_per_maturity_level = {
        1: pd.DataFrame(data=["data"]),
        2: pd.DataFrame(),
        3: pd.DataFrame(data=["data"]),
        4: pd.DataFrame(),
        5: pd.DataFrame(),
    }

    modified_dictionary = find_the_next_mat_level_to_reach(recommendations_per_mat_level=practices_per_maturity_level)

    assert modified_dictionary == {1: 2, 2: 2, 3: 5, 4: 5, 5: 5}


def test_find_the_next_mat_level_to_reach_only_level_1_empty(test_instance):
    practices_per_maturity_level = {
        1: pd.DataFrame(),
        2: pd.DataFrame(data=["data"]),
        3: pd.DataFrame(data=["data"]),
        4: pd.DataFrame(data=["data"]),
        5: pd.DataFrame(data=["data"]),
    }

    modified_dictionary = find_the_next_mat_level_to_reach(recommendations_per_mat_level=practices_per_maturity_level)

    assert modified_dictionary == {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


def test_find_the_next_mat_level_to_reach_only_level_4_empty(test_instance):
    practices_per_maturity_level = {
        1: pd.DataFrame(data=["data"]),
        2: pd.DataFrame(data=["data"]),
        3: pd.DataFrame(data=["data"]),
        4: pd.DataFrame(),
        5: pd.DataFrame(data=["data"]),
    }

    modified_dictionary = find_the_next_mat_level_to_reach(recommendations_per_mat_level=practices_per_maturity_level)

    assert modified_dictionary == {1: 1, 2: 2, 3: 4, 4: 4, 5: 5}


def test_find_the_next_mat_level_to_reach_2_empty(test_instance):
    practices_per_maturity_level = {
        1: pd.DataFrame(data=["data"]),
        2: pd.DataFrame(),
        3: pd.DataFrame(data=["data"]),
        4: pd.DataFrame(),
        5: pd.DataFrame(data=["data"]),
    }

    modified_dictionary = find_the_next_mat_level_to_reach(recommendations_per_mat_level=practices_per_maturity_level)

    assert modified_dictionary == {1: 2, 2: 2, 3: 4, 4: 4, 5: 5}


def test_find_the_next_mat_level_to_reach_no_empty(test_instance):
    practices_per_maturity_level = {
        1: pd.DataFrame(data=["data"]),
        2: pd.DataFrame(data=["data"]),
        3: pd.DataFrame(data=["data"]),
        4: pd.DataFrame(data=["data"]),
        5: pd.DataFrame(data=["data"]),
    }

    modified_dictionary = find_the_next_mat_level_to_reach(recommendations_per_mat_level=practices_per_maturity_level)

    assert modified_dictionary == {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


def test_find_the_next_mat_level_to_reach_only_1(test_instance):
    practices_per_maturity_level = {
        1: pd.DataFrame(data=["data"]),
        2: pd.DataFrame(),
        3: pd.DataFrame(),
        4: pd.DataFrame(),
        5: pd.DataFrame(),
    }

    modified_dictionary = find_the_next_mat_level_to_reach(recommendations_per_mat_level=practices_per_maturity_level)

    assert modified_dictionary == {1: 5, 2: 5, 3: 5, 4: 5, 5: 5}


def test_find_the_next_mat_level_to_reach_only_2(test_instance):
    practices_per_maturity_level = {
        1: pd.DataFrame(),
        2: pd.DataFrame(data=["data"]),
        3: pd.DataFrame(),
        4: pd.DataFrame(),
        5: pd.DataFrame(),
    }

    modified_dictionary = find_the_next_mat_level_to_reach(recommendations_per_mat_level=practices_per_maturity_level)

    assert modified_dictionary == {1: 1, 2: 5, 3: 5, 4: 5, 5: 5}


def test_find_the_next_mat_level_to_reach_only_3(test_instance):
    practices_per_maturity_level = {
        1: pd.DataFrame(),
        2: pd.DataFrame(),
        3: pd.DataFrame(data=["data"]),
        4: pd.DataFrame(),
        5: pd.DataFrame(),
    }

    modified_dictionary = find_the_next_mat_level_to_reach(recommendations_per_mat_level=practices_per_maturity_level)

    assert modified_dictionary == {1: 2, 2: 2, 3: 5, 4: 5, 5: 5}


def test_find_the_next_mat_level_to_reach_only_4(test_instance):  # todo: fix!
    practices_per_maturity_level = {
        1: pd.DataFrame(),
        2: pd.DataFrame(),
        3: pd.DataFrame(),
        4: pd.DataFrame(data=["data"]),
        5: pd.DataFrame(),
    }

    modified_dictionary = find_the_next_mat_level_to_reach(recommendations_per_mat_level=practices_per_maturity_level)

    assert modified_dictionary == {1: 3, 2: 3, 3: 3, 4: 5, 5: 5}


def test_find_the_next_mat_level_to_reach_only_5(test_instance):
    practices_per_maturity_level = {
        1: pd.DataFrame(),
        2: pd.DataFrame(),
        3: pd.DataFrame(),
        4: pd.DataFrame(),
        5: pd.DataFrame(data=["data"]),
    }

    modified_dictionary = find_the_next_mat_level_to_reach(recommendations_per_mat_level=practices_per_maturity_level)

    assert modified_dictionary == {1: 4, 2: 4, 3: 4, 4: 4, 5: 5}


def test_find_the_next_mat_level_to_reach_starting_from_level_2(test_instance):
    practices_per_maturity_level = {
        2: pd.DataFrame(data=["data"]),
        3: pd.DataFrame(data=["data"]),
        4: pd.DataFrame(),
        5: pd.DataFrame(data=["data"]),
    }

    modified_dictionary = find_the_next_mat_level_to_reach(recommendations_per_mat_level=practices_per_maturity_level)

    assert modified_dictionary == {2: 2, 3: 4, 4: 4, 5: 5}


def test_get_text_for_satisfied_subchars_expected_result():
    already_satisfied_subchars = {
        "monitor feature drift": "monitoring",
        "model performance monitoring": "monitoring",
        "compare with a baseline": "accuracy",
        "input data validation": "accuracy",
    }

    text_satisfied_subchars = get_text_for_satisfied_subchars(already_satisfied_subchars=already_satisfied_subchars)

    assert "Satisfied quality aspects" in text_satisfied_subchars
    assert "Monitoring" in text_satisfied_subchars
    assert "Accuracy" in text_satisfied_subchars
    assert "color: green" in text_satisfied_subchars


def test_get_text_for_satisfied_subchars_empty_input():
    already_satisfied_subchars = {}

    text_satisfied_subchars = get_text_for_satisfied_subchars(already_satisfied_subchars=already_satisfied_subchars)
    assert "Satisfied quality sub-characteristics" not in text_satisfied_subchars


def test_get_subchars_with_multiple_levels():
    subchars_w_multiple_levels = get_subchars_with_multiple_levels()

    assert subchars_w_multiple_levels == [
        "accuracy",
        "effectiveness",
        "efficiency",
        "resilience",
        "adaptability",
        "maintainability",
        "modularity",
        "testability",
        "repeatability",
        "operability",
        "monitoring",
        "readability",
        "traceability",
        "understandability",
        "debuggability",
    ]
