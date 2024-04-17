import logging
from typing import Dict, Tuple, Optional, Generator, List, Any
import validators
import pandas as pd
from ml_quality.report_automation import (
    prepopulate_gaps_dataframe,
    store_prepopulated_gaps
)
from ml_quality.constants import QUESTION_AND_ANSWER_DICT
import math
import numpy as np

logger = logging.getLogger(__name__)

"""
This set of functions creates gap files, given a pandas dataframe with the models' metadata from the registry.
"""


def write_gaps(iterator: Generator[Tuple[int, pd.Series], None, None], columns: List[str]) -> List[List[str]]:
    """
    Python function to be executed over partition of RDD, including the model's metadata from an ML registry.
    """
    df_input_series = pd.DataFrame([row for (_, row) in iterator], columns=columns).reset_index(drop=True)
    logger.debug(f"Series: {df_input_series.head(10)}")
    model_details = infer_gaps_from_registry_metadata(row=df_input_series.squeeze())
    logger.debug(f"Model details: {model_details}")
    df_g = prepopulate_gaps_dataframe(
        **{k: v for k, v in model_details.items() if k not in ["model_name", "business_criticality"]}
    )
    logger.debug(f"df_g: {df_g.head(10)}")

    df_g["model_name"] = df_input_series['model_name'][0]
    df_g["team_name"] = df_input_series['team_name'][0]
    df_g["business_criticality"] = model_details['business_criticality']
    df_g["model_family"] = df_input_series['model_family'][0]
    return df_g.reset_index(drop=True).values.tolist()


def create_gaps_files_for_all_models_in_registry(assets: pd.Series, output_folder: str):
    assets.replace({np.nan: None}, inplace=True)  # Helps with handling missing values
    for _, row in assets.iterrows():
        model_details = infer_gaps_from_registry_metadata(row=row)
        generate_gaps_from_model_info(model_details=model_details, output_folder=output_folder)


def infer_gaps_from_registry_metadata(row: pd.DataFrame) -> Dict[str, str]:
    row.fillna("", inplace=True)
    question_dictionary = QUESTION_AND_ANSWER_DICT
    model_details: dict[str, Any] = {k: None for k in question_dictionary.keys()}

    # Specify the required information/metadata we need for the assessment
    MODEL_NAME = row["model_name"]
    TEAM_NAME = row["team_name"]
    DAYS_SINCE_LAST_BINARY_UPLOAD = row["days_since_last_retrain"]
    AVG_CALLS_PER_HOUR = row["avg_calls_x_hour"]
    NUMBER_APPLICATIONS_USING_THE_MODEL = row["number_applications_using_the_model"]
    MODEL_IS_OF_STRATEGIC_IMPORTANCE = row["model_is_of_strategic_importance"]
    REGISTRY_URL = get_registry_url(model_name=MODEL_NAME)
    GIT_LINK = row["git_link"]
    FULLON_LINK = get_fullon_url(fullon_id=row["last_fullon_id"], fullon_tag=row["last_fullon_tag"])
    MONTHS_SINCE_LAST_FULLON = row["months_since_last_fullon"]

    if not AVG_CALLS_PER_HOUR:
        AVG_CALLS_PER_HOUR = 200  # we use a dummy value, to be able to infer business criticality

    if not MODEL_IS_OF_STRATEGIC_IMPORTANCE:
        MODEL_IS_OF_STRATEGIC_IMPORTANCE = False  # if not provided manually, set it to False

    if not NUMBER_APPLICATIONS_USING_THE_MODEL:
        NUMBER_APPLICATIONS_USING_THE_MODEL = 0  # if the list is Nonetype, set as 0 applications

    business_criticality = infer_business_criticality(
        avg_calls_per_hour=AVG_CALLS_PER_HOUR,
        fullon_link=FULLON_LINK,
        number_applications_using_the_model=NUMBER_APPLICATIONS_USING_THE_MODEL,
        model_is_of_strategic_importance=MODEL_IS_OF_STRATEGIC_IMPORTANCE,
    )

    DOCUMENTATION_URL = row["documentation"]
    COMPARISON_WITH_BASELINE_URL = row["comparison_with_baseline_link"]
    PIPELINE_URL = row["pipeline_url"]
    TEST_CODE_COVERAGE_PERCENTAGE = row["tests_code_coverage_percent"]
    PERFORMANCE_MONITORING_LINK = row["performance_monitoring_link"]
    FEATURE_MONITORING_LINK = row["feature_monitoring_link"]
    MAX_LATENCY_REQUIREMENT_FROM_LINKED_APPLICATIONS_IN_MS = row["max_latency_requirement_ms"]
    LOGGING_SOLUTION_LINK = row["model_logging_solution_link"]
    SHAPLEY_VALUE_CALCULATION_LINK = row["shapley_value_calculation_link"]
    RAQ_QUESTIONNAIRE_FILLED_IN = row["raq_questionnaire_filled_in"]
    INPUT_DATA_VALIDATION_URL = row["input_data_validation_link"]
    EXTERNAL_DATA_USED = row["external_data_used"]
    EXTERNAL_DATA_CHECKED_FOR_BOTS = row["external_data_checked_for_bots"]
    # Logic to infer gaps based on the model metadata, we should populate the model_details dictionary
    model_details["model_name"] = MODEL_NAME
    model_details["team_name"] = TEAM_NAME
    model_details["registry_link"] = REGISTRY_URL
    model_details["git_link"] = GIT_LINK
    model_details["fullon_link"] = FULLON_LINK

    if (not MONTHS_SINCE_LAST_FULLON) or (math.isnan(MONTHS_SINCE_LAST_FULLON)):
        model_details["is_fullon_older_than_6_months"] = None
    else:
        if MONTHS_SINCE_LAST_FULLON >= 6.0:
            model_details["is_fullon_older_than_6_months"] = True
        else:
            model_details["is_fullon_older_than_6_months"] = False

    if validators.url(DOCUMENTATION_URL):
        model_details[
            "is_understandable"
        ] = "partially"  # If there is a valid url for documentation, we put a small gap by default, no gap only after completeness assessment
    else:
        model_details["is_understandable"] = "no"

    if validators.url(COMPARISON_WITH_BASELINE_URL):
        model_details["has_comparison_with_baseline"] = "yes"
    else:
        model_details["has_comparison_with_baseline"] = "no"

    model_details["is_efficient"], model_details["is_resilient"] = assess_efficiency_and_resilience(
        pipeline_url=PIPELINE_URL
    )

    if validators.url(PIPELINE_URL):
        model_details["is_adaptable"] = "yes"
    elif DAYS_SINCE_LAST_BINARY_UPLOAD:
        if float(DAYS_SINCE_LAST_BINARY_UPLOAD) < 60.0:
            model_details["is_adaptable"] = "partially"
        else:
            model_details["is_adaptable"] = "no"
    else:
        model_details["is_adaptable"] = "no"

    if validators.url(PIPELINE_URL):
        model_details["has_repeatable_pipeline"] = "yes"
    else:
        model_details["has_repeatable_pipeline"] = "no"

    model_details['has_tested_code'] = get_test_coverage_gap(test_code_coverage=TEST_CODE_COVERAGE_PERCENTAGE)

    if validators.url(PERFORMANCE_MONITORING_LINK) and validators.url(FEATURE_MONITORING_LINK):
        model_details["has_monitoring"] = "yes"
    elif (not validators.url(PERFORMANCE_MONITORING_LINK)) and (not validators.url(FEATURE_MONITORING_LINK)):
        model_details["has_monitoring"] = "no"
    else:
        model_details["has_monitoring"] = "partially"

    if isinstance(MAX_LATENCY_REQUIREMENT_FROM_LINKED_APPLICATIONS_IN_MS, int):
        model_details["has_known_latency"] = "yes"
    else:
        model_details["has_known_latency"] = "no"

    if validators.url(LOGGING_SOLUTION_LINK):
        model_details["is_traceable"] = "yes"
    else:
        model_details["is_traceable"] = "no"

    if validators.url(SHAPLEY_VALUE_CALCULATION_LINK):
        model_details["is_explainable"] = "yes"
    else:
        model_details["is_explainable"] = "no"

    if RAQ_QUESTIONNAIRE_FILLED_IN == 'None':
        model_details["has_applicable_standards"] = "yes"
    else:
        model_details["has_applicable_standards"] = "no"  # If this variable is no, standards compliance has no gap.

    if validators.url(INPUT_DATA_VALIDATION_URL):
        model_details["has_input_data_validation"] = "yes"
    else:
        model_details["has_input_data_validation"] = "no"

    if EXTERNAL_DATA_USED:
        if EXTERNAL_DATA_CHECKED_FOR_BOTS:
            model_details["is_vulnerable"] = "no"
        else:
            model_details["is_vulnerable"] = "yes"
    else:
        model_details["is_vulnerable"] = "no"

    model_details["business_criticality"] = business_criticality
    return model_details


def generate_gaps_from_model_info(model_details: Dict[str, str], output_folder: str) -> None:
    store_prepopulated_gaps_args = model_details
    store_prepopulated_gaps_args["csv_path"] = output_folder
    logger.info(
        f"Creating gap file for {model_details['model_name']} in {output_folder} from the following proxies: {store_prepopulated_gaps_args}"
    )
    store_prepopulated_gaps(**{k: v for k, v in store_prepopulated_gaps_args.items() if k != "business_criticality"})


def get_registry_url(model_name: str) -> str:
    """
    This function returns the url with the entry of the model, in the ML registry. The URL will vary based on an
    organization's registry
    """
    return f"https://link_to_the_registry_entry/{model_name}"


def get_fullon_url(fullon_id: str, fullon_tag: str) -> Optional[str]:
    if fullon_id and fullon_tag:
        return f"https://<experiment_url_to_be_replaced>/{fullon_tag}/{int(fullon_id)}/"
    else:
        return None


def get_test_coverage_gap(test_code_coverage: Optional[str]) -> str:
    if not test_code_coverage:
        return "no"
    elif float(test_code_coverage) >= 50:
        return "yes"
    elif 20 <= float(test_code_coverage) < 50:
        return "partially"
    else:
        return "no"


def assess_efficiency_and_resilience(pipeline_url: str) -> Tuple[str, str]:
    """
    Implementation of this function relies on the worfklow scheduler (such as airflow) installation and is omitted.

    Its logic is the following:
        median_duration_seconds = pipeline.compute_successful_pipeline_duration_statistic(
            dag_id=dag_id, statistic="median"
        )

        median_duration_hours = median_duration_seconds / 3600

        ratio_failed_runs = pipeline.compute_ratio_failed_dags_last_three_months(dag_id=dag_id)

        if median_duration_hours < 5:
            is_efficient = "yes"
        else:
            is_efficient = "partially"

        if (ratio_failed_runs > 0.1) and (ratio_failed_runs < 0.3):
            is_resilient = "partially"
        elif ratio_failed_runs <= 0.1:
            is_resilient = "yes"
        else:
            is_resilient = "no"
    """

    is_efficient = "no"
    is_resilient = "no"

    return is_efficient, is_resilient


def infer_business_criticality(
    avg_calls_per_hour: float,
    fullon_link: str,
    number_applications_using_the_model: int,
    model_is_of_strategic_importance: bool,
) -> str:

    if not fullon_link:
        return "poc"
    else:
        if (
                avg_calls_per_hour >= 280000
                or number_applications_using_the_model >= 5
                or model_is_of_strategic_importance is True
        ):
            return "production_critical"
        else:
            return "production_non_critical"
