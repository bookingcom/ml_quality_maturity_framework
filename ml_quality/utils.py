import os
import pickle
import re
from datetime import datetime
from typing import Tuple, Any, Union, Optional, List, Dict

import numpy as np
import pandas as pd
import pdfkit
import plotly.express as px
import pkg_resources

from .constants import (
    HTML_COLOR,
    MATURITY_STANDARDS,
    MATURITY_LEVELS_URL,
    ML_BRAINSTORM_URL,
    MAX_MATURITY,
    BUSINESS_CRITICALITY_HTML,
)


def load_obj_from_pickle(pickle_file_path: str) -> Any:
    """
    Function to load class object from pickle file
    """
    with open(f"{pickle_file_path}", "rb") as pickle_file:
        obj = pickle.load(pickle_file)
    return obj


def dataframe_anti_join(left: pd.DataFrame, right: pd.DataFrame, on: List[str]) -> pd.DataFrame:
    """
    Function to remove rows present in another dataframe, i.e. anti join
    Args:
        left: Left dataframe
        right: Right dataframe
        on: name of the column on which performing the anti_join
    """
    if left.empty or right.empty:
        return left

    col_names = list(right.columns) + ["_merge"]
    outer_join = left.merge(right, on=on, how="outer", indicator=True, suffixes=("", "_right"))
    outer_join = outer_join[col_names]
    anti_join = outer_join[outer_join["_merge"] == "left_only"].drop("_merge", axis=1)
    return anti_join


def convert_html_to_pdf(html_file_path: str, pdf_file_path: str = None) -> str:
    """
    Utility to convert the html file to pdf
    Args:
        html_file_path: path of the html file to convert
        pdf_file_path: path of the output pdf_file
    """
    if pdf_file_path is None:
        pdf_file_path = html_file_path.replace(".html", ".pdf")

    pdfkit.from_file(html_file_path, pdf_file_path, options={"enable-local-file-access": None})

    return pdf_file_path


def set_pandas_value(df: pd.DataFrame, index: Tuple[Any, Any], column: Tuple[Any, Any]):
    """
    Function to set value in a specific cell of pandas dataframe
    """
    df_out = df.set_index(index[0])
    if hasattr(df_out.at[index[1], column[0]], "__len__"):
        raise ValueError("You can update only 1 value at the time, check index")
    else:
        df_out.at[index[1], column[0]] = column[1]
    return df_out.reset_index()


def plot_radar_chart(
    df: pd.DataFrame,
    r: str,
    theta: str,
    show_figure: bool = True,
    save_figure: bool = False,
    range_r: List[float] = None,
    plots_folder: str = "plots",
    save_name: str = "radar_chart",
    figtitle: str = "radar chart",
    save_figure_format: str = "png",
) -> Union[None, str]:
    """
    Plot radar chart with model characteristics
    Args:
        df: pandas dataframe with data to plot
        r: string with the name of the r parameter column
        theta: string with the name of the theta parameter column
        range_r: range of the radial axis
        show_figure: show radar chart, useful in Notebooks
        save_figure: save figure as .png file
        plots_folder: name of the plot folder where .png are stored
        save_name: suffix for the .png file
        figtitle: title of the figure
        save_figure_format: format to save the figure
    Returns:
        string with image path if figure is saved
    """

    fig = px.line_polar(df, r=r, theta=theta, line_close=True, title=figtitle, range_r=range_r)
    fig.update_traces(fill="toself")

    if show_figure:
        fig.show()

    if save_figure:
        os.system(f"mkdir -p ./{plots_folder}")
        image_path = f"{plots_folder}/{save_name}.{save_figure_format}".replace(" ", "_")
        fig.write_image(f"./{image_path}")
        return image_path


def get_column_value_from_row(df: pd.DataFrame, filter_col: str, filter_val: Any, value_col: str) -> str:
    """
    Get value of a column from a dataframe row, if more than 1 row
    """
    res = df[df[filter_col] == filter_val][value_col].unique()
    if len(res) != 1:
        raise ValueError(f"Zero or more than one row selected {res}")

    return res[0]


# todo: use this function
def get_dict_w_links_per_sub_char() -> Dict[str, str]:
    """
    Function to get link with explanation for every sub-char

    ### THIS IS NOT IN USE BECAUSE IT CHANGES THE COLOR OF THE FLAGS TO BLUE DUE TO THE URL ###
    """

    path_urls = pkg_resources.resource_stream(__name__, "data/characteristics.csv")
    df_w_urls = pd.read_csv(path_urls, header=0)
    return dict(zip(df_w_urls["sub_characteristic"], df_w_urls["url"]))


def get_html_recommendation_list(
    recommendations_per_mat_level: Dict,
    reason_dict: Optional[Dict[str, str]] = None,
) -> str:
    """
    Print a list of recommendations in html format from a Pandas DataFrame
    recommendations_per_mat_level: Dict[pd.DataFrame]
    including a dataframe with recommendations per maturity level until
    the expected one.
    """
    text = ""

    if reason_dict is None:
        reason_dict = {}

    if recommendations_per_mat_level == {}:
        return text

    maturity_to_reach_if_gaps_solved = find_the_next_mat_level_to_reach(
        recommendations_per_mat_level=recommendations_per_mat_level,
    )

    for maturity_level, recommendation_dataframe in recommendations_per_mat_level.items():

        if not recommendation_dataframe.empty:
            text += f"""
            <br>
            <p><b>Bring your model to maturity level {maturity_to_reach_if_gaps_solved[maturity_level]}</b></p>
                    """
            for _, row in recommendation_dataframe.iterrows():
                color = row["flag_colour"]

                if row["sub_characteristic"] in get_subchars_with_multiple_levels():
                    if row["gap_value"] == 2:
                        sub_char_level = "step 1"
                    else:
                        sub_char_level = "step 2"

                    sub_char_text = f"""
                        <span style="color: {HTML_COLOR[color]}; ">
                            <b>{row['sub_characteristic'].replace('_', '-').capitalize()}</b>
                        </span> - {sub_char_level}
                        """
                else:
                    sub_char_text = f"""
                    <span style="color: {HTML_COLOR[color]}; ">
                        <b>{row['sub_characteristic'].replace('_', '-').capitalize()}</b>
                    </span>
                    """
                text += f"""
                            Fix: 
                            {sub_char_text}
                            ({row['characteristic'].capitalize()})
                            <br>
                        """

                text += f"""
                <ul style="list-style-type: none">
                """
                if reason_dict.get(row["sub_characteristic"]):
                    text += get_str_motivation_and_practice(
                        text="Motivation", motivation_or_practice=reason_dict[row["sub_characteristic"]]
                    )

                text += get_str_motivation_and_practice(
                    text="How to fix", motivation_or_practice=row["recommended_practice_explanation"]
                )
                text += "</ul>"
    return text


def find_the_next_mat_level_to_reach(recommendations_per_mat_level: Dict):
    """
    Returns a dictionary in which the values indicate which maturity level the model will reach, if we fix the gaps
        required for the maturity level of the 'key'

    For example if by comparing with a baseline, the model will reach maturity level 5 because the rest of the gaps are
    solved, we want to reflect this in the report.

    """

    levels_with_recommendations = {
        key: value for (key, value) in recommendations_per_mat_level.items() if not value.empty
    }
    keys_with_recommendations = sorted(list(levels_with_recommendations.keys()))

    next_mat_level_to_reach = {}

    current_maturity = min(recommendations_per_mat_level.keys()) - 1
    missing_keys = {1, 2, 3, 4, 5} - {current_maturity} - set(keys_with_recommendations)

    if len(keys_with_recommendations) == 5:  # no change at all
        next_mat_level_to_reach = {key: key for (key, _) in recommendations_per_mat_level.items()}

    elif len(keys_with_recommendations) == 1:  # Create the dictionary independently
        key_with_recommendation = keys_with_recommendations[0]
        for key in missing_keys:
            # if it is a lower level, it will reach until the one with recommendation
            if key <= key_with_recommendation:
                next_mat_level_to_reach[key] = key_with_recommendation - 1
            # Otherwise it will reach level 5, since we have no other recommendations
            else:
                next_mat_level_to_reach[key] = 5

        next_mat_level_to_reach[key_with_recommendation] = 5

    elif len(keys_with_recommendations) == 2:
        max_key = max(keys_with_recommendations)
        for key in missing_keys:
            if key <= max_key:
                next_mat_level_to_reach[key] = max_key - 1
            else:
                next_mat_level_to_reach[key] = 5

        remaining_key = set(keys_with_recommendations) - {max_key}
        next_mat_level_to_reach[list(remaining_key)[0]] = max_key - 1
        next_mat_level_to_reach[max_key] = 5

    else:  # if there are at least 3 levels with recommendations
        for index in range(len(keys_with_recommendations)):
            if keys_with_recommendations[index] != 5:
                if keys_with_recommendations[index] != keys_with_recommendations[index + 1] - 1:
                    next_mat_level_to_reach[keys_with_recommendations[index]] = keys_with_recommendations[index + 1] - 1
                else:
                    next_mat_level_to_reach[keys_with_recommendations[index]] = keys_with_recommendations[index]

        for key in missing_keys:
            if len(missing_keys) == 1:
                missing_key = list(missing_keys)[0]
                next_mat_level_to_reach[missing_key] = missing_key

            elif key <= next_mat_level_to_reach[key - 1]:

                next_mat_level_to_reach[key] = next_mat_level_to_reach[key - 1]
            # else: # todo: check if this is necessary. This is not covered by tests
            #    next_mat_level_to_reach[key] = key

        if 5 not in next_mat_level_to_reach.keys():
            next_mat_level_to_reach[5] = 5

        next_mat_level_to_reach = dict(sorted(next_mat_level_to_reach.items()))

    return next_mat_level_to_reach


def add_custom_recommendation(recommendation: str, fontsize_pct: int = 90):
    return f"""
        <br>
        <span style="font-size: {fontsize_pct}%">{recommendation}</span>
        <br>
    """


def generate_report_text(
    model_name: str,
    model_criticality: str,
    model_maturity: int,
    model_family: Optional[str],
    quality_score: int,
    radar_chart_path: Optional[str],
    reason_dict: Optional[Dict[str, str]],
    recommendation_txt: Optional[str],
    summary: str,
    recommendations: Dict,
    already_satisfied_subchars: Dict,
    report_date: Optional[str] = None,
    font_type: str = "verdana",
    max_maturity: int = 5,
    version: str = "dev",
) -> str:
    """
    Function to generate the report text
    Args:
        model_name: Name of the model
        report_date: Date of the report
        model_criticality: business criticality level of the model
        model_maturity: maturity level of the model
        quality_score: model quality score
        reason_dict: dictionary with reasons per flag,
        radar_chart_path: path of the radar chart image
        recommendation_txt: free text recommendation
        font_type: Type of the font
        max_maturity: maximum allowed maturity level
        version: version of the package currently used
        already_satisfied_subchars: subcharacteristics already satisfied
        summary: text with summary of the evaluation
        recommendations: dictionary with recommendations
    Returns:
        String with report text
    """
    if report_date is None:
        report_date = str(datetime.now().date())

    text = f"""
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <style>
                h1 {{
                    font-family: {font_type},sans-serif;
                    font-size: 200%;
                }}
                p  {{
                    font-family: {font_type}, sans-serif;
                    font-size: 120%;
                }}
                body  {{
                    font-family: {font_type}, sans-serif;
                    font-size: 110%;
                }}
                h2 {{
                    font-variant: small-caps;
                    font-family: {font_type}, sans-serif;
                    font-size: 170%;
                }}
                hr {{
                    width: 320px; 
                    color: black;
                    height: 1px;
                    margin-left: 0;
                }}
                li  {{
                    margin: 13px 0;
                }}
                </style>
                <title>Model Report</title>
            </head>
            <body>
                <h2><span style='color:grey'>ML Quality Assessment</span></h2>
                <hr>
                <h1>{model_name.replace('_', ' ').capitalize()}</h1>
                <b>Date:</b> {report_date}
                <br><br>
                <b>Business Criticality:</b> {BUSINESS_CRITICALITY_HTML[model_criticality]}
                <br><br>
    """
    if (model_family is not None) and (len(model_family) > 1):
        text += f"""
        <b>Model family:</b> {model_family} <sup> <a href="#fn1" id="ref1">1</a> </sup> 
        <br><br>
        """

    minimal_maturity = MATURITY_STANDARDS[model_criticality]
    meets_requirements = model_maturity >= minimal_maturity

    if meets_requirements:
        extra_text = ""
        maturity_status_text = "<b><span style='color:darkgreen'>Quality standards are met</span></b>"
    else:
        extra_text = f" ({minimal_maturity}/{max_maturity} required for {BUSINESS_CRITICALITY_HTML[model_criticality]})"
        maturity_status_text = "<span style='color:darkred'><b>Insufficient maturity!</b></span>"

    text += f"""
            <b>Maturity level:</b> 
            {model_maturity}/{max_maturity}{extra_text}
            <br><br>
            <b>Maturity Status:</b> {maturity_status_text}
            <br><br>
            <b>Quality score:</b> {int(quality_score)}/100
            <br>
            """

    # Add chart
    text += f"<img src={radar_chart_path} alt='radar_chart'>"

    # Add auto-generated summary
    text += f"""
    <br>
    <span style="font-size: 90%">
    <b>Summary: </b>{summary}
    </span><br>
    """

    if recommendation_txt is not None:
        text += add_custom_recommendation(recommendation_txt)

    if recommendations != {}:
        text += get_html_recommendation_list(recommendations_per_mat_level=recommendations, reason_dict=reason_dict)

    if model_maturity == MAX_MATURITY:
        text += """
            <p><span style="color: darkgreen; "><b>Congratulations! Your model is fully mature</b></span></p>
        """
    else:
        text += f"""<br><br>
            <span style="font-size: 90%">
            If the aforementioned practices are not enough, you can request an 
            <a href="{ML_BRAINSTORM_URL}"> ML System Brainstorm</a>
            and we will assign 2 reviewers to discuss how to improve your model.
            </span>
        """

    # todo - Further improvement: Mention which level of the subchar is satisfied.
    #  For example: Accuracy level 1, readability level 2, etc.
    text += get_text_for_satisfied_subchars(already_satisfied_subchars=already_satisfied_subchars)

    text += f"""
           <br>
           <span style="font-size: 90%">
           Learn more about the maturity levels <a href="{MATURITY_LEVELS_URL}">here</a>.
           </span> 

           <div style="height:100px;"><br></div>
           <small><i>Generated with ml_quality {version}</i></small>
       """

    if model_family is not None:
        text += """
        <hr></hr>
        <sup id="fn1">1. Models of the same family share the same quality assessment. <a href="#ref1" title="Jump back to footnote 1 in the text.">&#8617;</a></sup>
        """
    text += """
            </body>
        </html>
    """

    return text


def get_pkg_version(version_file_path: str = "_version.py") -> str:
    with open(version_file_path, "r") as f:
        version_content = f.read()
    return re.search(r"^__version__ = \"(.+)\"$", version_content, flags=re.MULTILINE).group(1)


def add_is_max_col(df: pd.DataFrame, id_cols: List[str], sort_col: str, is_max_col: str = "max") -> pd.DataFrame:
    """
    Function to add a column with the latest flag to an existing dataframe
    Args:
        df: pandas DataFrame
        id_cols: list of id columns
        sort_col: name of the col to sort by
        is_max_col: name of the column containing info about max
    Returns:
        pandas DataFrame
    """
    latest = df[id_cols + [sort_col]].groupby(id_cols).max().reset_index()
    df = pd.merge(df, latest, on=id_cols, suffixes=("", "_max"))
    df[is_max_col] = df[f"{sort_col}_max"] == df[sort_col]
    del df[f"{sort_col}_max"]
    return df


def check_date_format(date_str: str, date_fmt: str) -> bool:
    """
    Check date format
    Args:
        date_str: string with date
        date_fmt: date format
    Returns:
        True or False
    """
    try:
        datetime.strptime(date_str, date_fmt)
        return True
    except ValueError:
        return False


def propagate_last_value(
    df: pd.DataFrame, col_to_propagate: str, groupby_col: str, sort_cols: List[str], new_col_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Function to propagate last value within a group in a Pandas DataFrame
    Args:
        df: initial pandas dataframe
        groupby_col: col for grouping by
        col_to_propagate: columns which value we want to propagate
        sort_cols: List of columns to sort by
        new_col_name: optional name of the new column, if none col_to_propagate will be overwritten
    Returns:
        pandas dataframe
    """
    last_value = df.sort_values(by=sort_cols).groupby(groupby_col)[col_to_propagate].last().reset_index()
    df = pd.merge(df, last_value, on=groupby_col, suffixes=("", "_last"))
    if new_col_name is None:
        new_col_name = col_to_propagate
    df[new_col_name] = df[f"{col_to_propagate}_last"]
    return df.drop(f"{col_to_propagate}_last", axis=1)


def compute_statistics_of_given_column(
    summary_csv_path: str,
    column: str,
    statistic: str,
    team_names_for_computation: List[str],
    business_criticality: List[str],
) -> float:
    """
    Computes a statistic for the quality score, for the models of selected teams.

    Args:
        summary_csv_path: the path with the historical summary
        column: the column to compute statistic for
        statistic: mean or median
        team_names_for_computation: list with the names of the teams to compute the statistic for
        business_criticality: list of business criticalities to include

    Returns:
        Float indicating the requested statistic

    Example usage:
        compute_statistics_quality_score(summary_csv_path='../../assessments/all_models_summary/historical_summary.csv',
                                         statistic='mean',
                                         team_names_for_computation=['ml_core','smart_value','recommendation_platform'],
                                         business_criticality=['production_non_critical', 'production_critical'])

    """
    summary = pd.read_csv(summary_csv_path, sep=",", header=0)

    summary_of_relevant_teams = summary[
        (summary["team"].isin(team_names_for_computation))
        & (summary["business_criticality"].isin(business_criticality))
        & (summary["latest"])
    ]

    if statistic == "mean":
        statistic_value = np.mean(summary_of_relevant_teams[column])
    elif statistic == "median":
        statistic_value = np.median(summary_of_relevant_teams[column])
    else:
        raise ValueError("The requested statistic is not supported")

    result = round(float(statistic_value), 2)

    print(f"The {statistic} quality score for {len(summary_of_relevant_teams)} models is {result}.")
    return result


def read_practices_to_maturity_levels_from_csv() -> pd.DataFrame:
    stream = pkg_resources.resource_stream(__name__, "data/practices_to_maturity_levels.csv")
    return pd.read_csv(stream)


def get_text_for_satisfied_subchars(already_satisfied_subchars: Dict[str, str]) -> str:
    practices_explanations = read_practices_to_maturity_levels_from_csv()
    text = ""
    if already_satisfied_subchars != {}:
        text += """
                <br><br><br>
                <p><b>Satisfied quality aspects</b></p>
                """
    text += "<ul>"

    for practice, sub_char in already_satisfied_subchars.items():
        satisfied_practices = practices_explanations[practices_explanations["practice_name"] == practice]
        text += f"""
            <li> <span style="color: green; "><b>{sub_char.replace('_', '-').capitalize()}</b></span>:
            <span style="font-size: 90%">
            {satisfied_practices['satisfied_practice_explanation'].values[0]}
            </span>
            </li>    
        """
    text += "</ul>"
    return text


def get_subchars_with_multiple_levels():
    maturity_requirements = pd.read_csv(pkg_resources.resource_stream(__name__, "data/maturity_levels.csv"))

    maturity_requirements["has_multiple_levels"] = maturity_requirements.apply(
        lambda x: True
        if (x["level_1"] == 1 or x["level_2"] == 1 or x["level_3"] == 1 or x["level_4"] == 1 or x["level_5"] == 1)
        else False,
        axis=1,
    )

    return list(
        maturity_requirements[maturity_requirements["has_multiple_levels"] == True]["sub_characteristic"].values
    )


def get_str_motivation_and_practice(text: str, motivation_or_practice: str):
    """
    Template for printing reasoning/motivation or how to fix a sub-char
    Args:
        text: str, indicating the content
        motivation_or_practice: str, the reasoning or the practice related to the gap in the sub-characteristic
    Returns:
        Html string
    """
    return f"""
                <li>
                    <span style="font-size: 95%"><b>{text}:</b></span> 
                    <span style="font-size: 90%"> {motivation_or_practice}</span>
                </li>
            """
