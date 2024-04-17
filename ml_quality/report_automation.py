from typing import Optional, Dict, Union, Any
from .constants import SUB_CHAR_LIST, QUESTION_AND_ANSWER_DICT
import logging
import pandas as pd
import pkg_resources
from datetime import datetime

from ml_quality.assessments import QualityAssessment, format_team_name, get_historical_summary
from ml_quality._version import __version__
from ml_quality.constants import (
    ALLOWED_NO_ANSWERS,
    ALLOWED_YES_ANSWERS,
    ANSWER_TO_BOOL,
    ANSWER_TO_GAPS,
)

from pathlib import Path
import html5lib
from datetime import date
import os

logger = logging.getLogger(__name__)


def answer_to_bool(k: Optional[str]):
    if isinstance(k, bool):
        return k
    return ANSWER_TO_BOOL[k]


def answer_to_gaps(k: Any):
    return ANSWER_TO_GAPS[k]


def invert_answer(ans: Union[str, bool]):
    """
    Helper function to invert answer for certain sub_chars
    """
    if ans in ["yes", "fully", True]:
        return False
    elif ans in ["no", False]:
        return True
    else:
        return ans


def check_fulfillment_dict(fulfillment_dict: Dict[str, Dict[str, Optional[str]]]):
    for sub_char in fulfillment_dict.keys():
        if sub_char not in SUB_CHAR_LIST:
            raise ValueError(f"The dictionary contains an unexpected key: {sub_char}")

    for sub_char, reasoning in fulfillment_dict.items():
        if set(reasoning.keys()) != {"no", "small", "large", None}:
            raise ValueError(f"The sub_char: {sub_char} has an unexpected reasoning sub-dict {reasoning}")


def get_fulfillment_per_subchar(
    git_link: Optional[str] = None,
    fullon_link: Optional[str] = None,
    serving_link: Optional[str] = None,
    team_name: Optional[str] = None,
    has_input_data_validation: Optional[bool] = None,
    has_comparison_with_baseline: Optional[bool] = None,
) -> Dict[str, Dict[str, str]]:

    if bool(fullon_link):
        model_fullon_str = f"The model is fullon: {fullon_link}."
    else:
        model_fullon_str = "There is no full-on experiment associated with the model."

    if bool(serving_link):
        model_in_rs = f"The model is deployed in a discoverable serving system (e.g.RS, AWS, etc): {serving_link}."
    else:
        model_in_rs = "Model not deployed."

    if bool(team_name):
        model_owned = f"The model is owned by {team_name}."
    else:
        model_owned = "The model is not owned by any team."

    if has_input_data_validation and not has_comparison_with_baseline:
        accuracy_small_gap = "There is input data validation, but there is no comparison with a simple baseline."
    elif has_comparison_with_baseline and not has_input_data_validation:
        accuracy_small_gap = "The model outperforms a simple baseline, but there is no input data validation."
    else:
        accuracy_small_gap = None

    if bool(git_link):
        maintainability_small_gap = "The code is versioned in Git, but its readability has room for improvement."
    else:
        maintainability_small_gap = None

    fulfillment_per_subchar = dict(
        accuracy=dict(
            no="The model outperforms a simple baseline and has input data validation.",
            small=accuracy_small_gap,
            large="There is no comparison with a simple baseline, neither input data validation.",
        ),
        effectiveness=dict(
            no=model_fullon_str,
            small=f"{model_fullon_str}, but it was tested more than 6 months ago.",
            large=model_fullon_str,
        ),
        responsiveness=dict(
            no="Latency/throughput requirements are known and met (or the model is offline)",
            small=None,
            large="Latency/throughput requirements not known",
        ),
        usability=dict(no=model_in_rs, small=None, large="The model is not usable."),
        cost_effectiveness=dict(
            no=f"{model_fullon_str}, gains are larger than model's costs.",
            small=None,
            large=model_fullon_str,
        ),
        efficiency=dict(
            no="Resources are optimized",
            small="Basic model operations are automated, but technical resources are not fully optimized.",
            large=None,
        ),
        availability=dict(no=model_in_rs, small=None, large="The model is not highly available for predictions."),
        resilience=dict(
            no="There are at most 1 system failure per quarter.",
            small="Less than 5 failures per quarter.",
            large="There are more than 5 failures per quarter.",
        ),
        adaptability=dict(
            no="The model is able to fully adapt to changes in the environment.",
            small="The model is partially adaptable.",
            large="The model cannot adapt to changes in the environment.",
        ),
        scalability=dict(no=model_in_rs, small=None, large="The model is not deployed on a scalable serving system."),
        maintainability=dict(
            no=None,
            small=maintainability_small_gap,
            large="The model's source code is not versioned or repository link has not been provided.",
        ),
        modularity=dict(
            no="The model's code is fully modular.",
            small="The model's code is partially modular.",
            large="The model's code is non-modular or modularity has not been assessed.",
        ),
        testability=dict(no=None, small=None, large="The source code is not unit-tested."),
        repeatability=dict(
            no="There is an associated pipeline to repeat the ML lifecycle.",
            small=None,
            large="Repeating the ML lifecycle is completely manual.",
        ),
        operability=dict(
            no=model_in_rs,
            small=None,
            large="The model is not deployed on a system where it can be managed (disabled, updated, revert).",
        ),
        monitoring=dict(
            no="There is monitoring of the most important indicators.",
            small=None,
            large="There is no monitoring of ML performance, features, inputs, business metrics.",
        ),
        discoverability=dict(
            no=model_in_rs,
            small=None,
            large="The model is not registered in a discoverable serving system.",
        ),
        readability=dict(
            no="Naming of functions/variables is human readable, the code is modular and has a unified code style.",
            small="Naming of functions/variables is human readable, but the code is either non-modular or does not have a unified code style.",
            large="Readability is not assessed, or the code is not easily readable, functions/variables have non-human readable names.",
        ),
        traceability=dict(
            no="Metadata and artifacts are fully logged.",
            small="Metadata and artifacts are only partially logged.",
            large="Metadata and artifacts are not being logged.",
        ),
        understandability=dict(
            no="The provided documentation of the ML system is complete.",
            small="The provided documentation of the ML system is not complete.",
            large="No documentation of the ML system has been provided.",
        ),
        explainability=dict(
            no="The model's predictions are explainable.",
            small=None,
            large="It is not possible to explain the model predictions.",
        ),
        fairness=dict(
            no="Fairness requirement to be determined by the Risk Assessment Questionnaire.",
            small=None,
            large="The model has not been checked for undesired biases.",
        ),
        ownership=dict(no=model_owned, small=None, large=model_owned),
        standards_compliance=dict(
            no="No applicable standards.",
            small=None,
            large="Applicable standards are not known or the model does not comply with them.",
        ),
        vulnerability=dict(no="No user generated data or external data sources are consumed.", small=None, large=None),
    )

    for sub_char in SUB_CHAR_LIST:
        fulfillment_per_subchar[sub_char].update({None: None})

    check_fulfillment_dict(fulfillment_per_subchar)

    return fulfillment_per_subchar


def format_name(s: str) -> str:
    return s.replace(" ", "_").replace("&", "").replace("__", "_").lower()


def create_or_update_gaps_for_assessment(create: bool = True, output_folder: str = "inputs") -> Dict[str, str]:
    """
    Function to generate gaps.csv for model evaluation
    Args:
        create: (Y/N) if True ask for all questions and create new gaps file,
                    otherwise asks only for name, team, business_criticality
        output_folder: name of the output folder where to store gaps
    Returns:
        dictionary wih model information
    """

    os.system(f"mkdir -p {output_folder}")

    if create:
        question_dictionary = QUESTION_AND_ANSWER_DICT
    else:
        basic_keys = ["model_name", "team_name", "business_criticality"]
        question_dictionary = dict((k, QUESTION_AND_ANSWER_DICT[k]) for k in basic_keys)

    model_details: dict[str, Any] = {k: None for k in question_dictionary.keys()}

    for gap_arg, question_answer in question_dictionary.items():
        pre_requisite = question_answer.get("prerequisite", None)
        if pre_requisite and not model_details[pre_requisite]:
            logger.debug(f"Skipping {gap_arg} because prerequisite {pre_requisite} not filled")
            continue
        if question_answer["answer"] is None:
            ans = input(question_answer["question"])
            logger.debug(f"\n-> {ans}")
            model_details[gap_arg] = ans
        else:
            while model_details[gap_arg] not in question_answer["answer"]:
                ans = input(question_answer["question"]).lower()
                logger.debug(f"\n-> {ans}")
                if ans in ALLOWED_YES_ANSWERS:
                    ans = "yes"
                if ans in ALLOWED_NO_ANSWERS:
                    ans = "no"
                if ans not in question_answer["answer"]:
                    print(f"Please insert a value from: {question_answer['answer']}")
                else:
                    model_details[gap_arg] = ans

    if create:
        store_prepopulated_gaps_args = model_details
        store_prepopulated_gaps_args["csv_path"] = output_folder
        logger.info(f"Creating gap file in {output_folder} from the following proxies: {store_prepopulated_gaps_args}")
        store_prepopulated_gaps(
            **{k: v for k, v in store_prepopulated_gaps_args.items() if k != "business_criticality"})

    return model_details


def add_model_for_assessment(
    models_file: str,
    model_info: Dict[str, str],
    output_models_file: Optional[str] = None,
    gaps_file: Optional[str] = None,
    custom_recommendation: Optional[str] = None,
):
    """
    Function to add model to .csv for assessment generation
    Args:
        models_file: name of the file with all models to be assessed
        model_info: dictionary with model information
        output_models_file: output file to be written should be same structure as models_file
        gaps_file: file with the gaps for the model
        custom_recommendation: string with custom recommendation for the specific model
    """
    if output_models_file is None:
        output_models_file = models_file

    if custom_recommendation is None:
        custom_recommendation = ""

    model_name = model_info["model_name"]
    if gaps_file is None:
        gaps_file = f"gaps_{model_name}.csv"

    team_name = format_name(model_info["team_name"])
    business_criticality = model_info["business_criticality"]
    logger.info(f"Adding model: {model_name} with gaps from: {gaps_file} to {output_models_file}")
    new_row = pd.DataFrame(
        dict(
            name=[model_name],
            mlp_name=[model_name],
            business_criticality=[business_criticality],
            custom_recommendation=[custom_recommendation],
            date=[str(datetime.now().date())],
            gaps_file=[gaps_file],
            team=[team_name],
        )
    )

    try:
        models_to_assess = pd.read_csv(models_file)
        models_to_assess = models_to_assess[models_to_assess.name != model_name]
    except FileNotFoundError:
        models_to_assess = None
    models_to_assess = pd.concat([models_to_assess, new_row])
    models_to_assess.to_csv(output_models_file, index=False)


def copy_inputs_for_model_family(models_file: str, model_name: str, model_family: str) -> None:
    """
    Given a model with specified model family and existing family_gaps.csv, copy the gaps file of the family for the
    new model and create report.
    """
    models_to_assess = pd.read_csv(models_file)

    gaps = list(set(models_to_assess[models_to_assess["model_family"] == model_family]["gaps_file"]))
    business_criticalities = list(
        set(models_to_assess[models_to_assess["model_family"] == model_family]["business_criticality"])
    )

    team_names = list(set(models_to_assess[models_to_assess["model_family"] == model_family]["team"]))
    if len(gaps) > 1:
        raise ValueError(f"More than one gaps files found for the same model family: {gaps}. Only one should exist.")
    if len(business_criticalities) > 1:
        raise ValueError(
            f"More than one business criticalities found for the same model family: {business_criticalities}. Only one should exist."
        )
    if len(team_names) > 1:
        raise ValueError(
            f"More than one team names found for the same model family: {team_names}. Only one should exist."
        )
    gaps_file = gaps[0]
    business_criticality = business_criticalities[0]
    team_name = team_names[0]

    new_row = pd.DataFrame(
        dict(
            name=[model_name],
            mlp_name=[model_name],
            business_criticality=[business_criticality],
            custom_recommendation=[""],
            date=[str(datetime.now().date())],
            gaps_file=[gaps_file],
            team=[team_name],
            model_family=[model_family],
        )
    )
    logger.info(
        f"Adding model: {model_name} of model family {model_family} with gaps from: {gaps_file} to {models_file}"
    )

    models_to_assess = pd.concat([models_to_assess, new_row])
    models_to_assess.to_csv(models_file, index=False)


def prepopulate_gaps_dataframe(
    team_name: Optional[str] = None,
    is_understandable: Optional[str] = None,
    registry_link: Optional[str] = None,
    git_link: Optional[str] = None,
    fullon_link: Optional[str] = None,
    is_fullon_older_than_6_months: Optional[Union[bool, str]] = None,
    has_comparison_with_baseline: Optional[Union[bool, str]] = None,
    is_efficient: Optional[str] = None,
    has_known_latency: Optional[Union[bool, str]] = None,
    is_resilient: Optional[str] = None,
    is_adaptable: Optional[str] = None,
    has_repeatable_pipeline: Optional[Union[bool, str]] = None,
    has_tested_code: Optional[Union[bool, str]] = None,
    has_monitoring: Optional[Union[bool, str]] = None,
    is_traceable: Optional[str] = None,
    is_explainable: Optional[Union[bool, str]] = None,
    has_applicable_standards: Optional[Union[bool, str]] = None,
    has_input_data_validation: Optional[Union[bool, str]] = None,
    is_vulnerable: Optional[Union[bool, str]] = None,
) -> pd.DataFrame:
    """
    Pre-populates the csv with the gaps for the quality assessment, given some basic info.

    All the arguments of type Optional[str] can take values: None, 'fully', 'partially'.
    All the arguments of type bool, can take values: True, False

    Returns:
        Pandas Dataframe with the gaps and their reasoning.
    """
    for k, v in locals().items():
        allowed_values = list(ANSWER_TO_GAPS.keys())
        if QUESTION_AND_ANSWER_DICT[k]["answer"] is None:
            continue
        if len(QUESTION_AND_ANSWER_DICT[k]["answer"]) == 2:
            allowed_values.remove("partially")
        if v not in allowed_values:
            raise ValueError(f"Value of {k}: {v} not allowed, values can only be: {allowed_values}")

    gap_template_path = pkg_resources.resource_stream(__name__, "data/characteristics.csv")
    gaps = pd.read_csv(gap_template_path, header=0).drop("characteristic", axis=1)
    gaps["reasoning"] = None

    criterion_per_subchar = get_criterion_per_subchar(**locals())

    fulfillment_per_subchar = get_fulfillment_per_subchar(git_link=git_link, fullon_link=fullon_link,
                                                          serving_link=registry_link, team_name=team_name,
                                                          has_input_data_validation=answer_to_bool(
                                                              has_input_data_validation),
                                                          has_comparison_with_baseline=answer_to_bool(
                                                              has_comparison_with_baseline))

    for sub_char, criterion in criterion_per_subchar.items():
        gap_value = ANSWER_TO_GAPS[criterion]
        if gap_value is None:
            gap_value = "large"
        gaps.loc[gaps["sub_characteristic"] == sub_char, "gap_value"] = gap_value
        gaps.loc[gaps["sub_characteristic"] == sub_char, "reasoning"] = fulfillment_per_subchar[sub_char][gap_value]

    return gaps.fillna({"reasoning": ""})


def store_prepopulated_gaps(
    model_name: str,
    csv_path: str,
    team_name: Optional[str] = None,
    is_understandable: Optional[str] = None,
    registry_link: Optional[str] = None,
    git_link: Optional[str] = None,
    has_comparison_with_baseline: Optional[Union[str, bool]] = None,
    is_efficient: Optional[str] = None,
    fullon_link: Optional[str] = None,
    is_fullon_older_than_6_months: Optional[Union[str, bool]] = None,
    is_resilient: Optional[str] = None,
    has_known_latency: Optional[Union[str, bool]] = None,
    is_adaptable: Optional[Union[str, bool]] = None,
    has_tested_code: Optional[Union[str, bool]] = None,
    has_repeatable_pipeline: Optional[Union[str, bool]] = None,
    has_monitoring: Optional[Union[bool, str]] = None,
    is_traceable: Optional[str] = None,
    is_explainable: Optional[Union[bool, str]] = None,
    has_applicable_standards: Optional[Union[bool, str]] = None,
    has_input_data_validation: Optional[Union[bool, str]] = None,
    is_vulnerable: Optional[Union[bool, str]] = None,
) -> pd.DataFrame:
    prepopulate_gaps_dataframe_args = {k: v for k, v in locals().items() if k not in ["csv_path", "model_name"]}
    gaps = prepopulate_gaps_dataframe(**prepopulate_gaps_dataframe_args)

    model_name = format_name(model_name)
    gaps.to_csv(f"{csv_path}/gaps_{model_name}.csv", index=False)
    return gaps


def get_criterion_per_subchar(**kwargs):
    """
    Function to convert answers from the start_assessment.py file to gaps
    """

    is_effective = compute_effectiveness_gap(
        fullon_link=bool(kwargs["fullon_link"]),
        is_fullon_older_than_6_months=answer_to_bool(kwargs["is_fullon_older_than_6_months"]),
    )

    is_accurate = compute_accuracy_gap(
        has_input_data_validation=answer_to_bool(kwargs["has_input_data_validation"]),
        has_comparison_with_baseline=answer_to_bool(kwargs["has_comparison_with_baseline"]),
    )

    if kwargs["git_link"]:
        is_maintainable = "partially"
    else:
        is_maintainable = "no"

    criterion_per_subchar = {
        "accuracy": is_accurate,
        "effectiveness": is_effective,
        "responsiveness": kwargs["has_known_latency"],
        "usability": bool(kwargs["registry_link"]),
        "cost_effectiveness": bool(kwargs["fullon_link"]),
        "efficiency": kwargs["is_efficient"],
        "availability": bool(kwargs["registry_link"]),
        "resilience": kwargs["is_resilient"],
        "adaptability": kwargs["is_adaptable"],
        "scalability": bool(kwargs["registry_link"]),
        "maintainability": is_maintainable,
        "modularity": None,
        "testability": kwargs["has_tested_code"],
        "operability": bool(kwargs["registry_link"]),
        "repeatability": kwargs["has_repeatable_pipeline"],
        "monitoring": kwargs["has_monitoring"],
        "discoverability": bool(kwargs["registry_link"]),
        "readability": None,
        "traceability": kwargs["is_traceable"],
        "understandability": kwargs["is_understandable"],
        "explainability": kwargs["is_explainable"],
        "fairness": True,
        "ownership": bool(kwargs["team_name"]),
        "standards_compliance": invert_answer(kwargs["has_applicable_standards"]),
        "vulnerability": invert_answer(kwargs["is_vulnerable"]),
    }
    return criterion_per_subchar


def compute_effectiveness_gap(fullon_link: bool, is_fullon_older_than_6_months: bool) -> Union[None, str]:
    if fullon_link and not is_fullon_older_than_6_months:
        return "yes"
    if fullon_link and is_fullon_older_than_6_months:
        return "partially"
    return None


def compute_accuracy_gap(has_input_data_validation: bool, has_comparison_with_baseline: bool) -> Union[None, str]:
    if has_input_data_validation and has_comparison_with_baseline:
        return "yes"
    if has_input_data_validation != has_comparison_with_baseline:
        return "partially"
    return None

def get_mlp_model_names(dir_with_models: Optional[str] = None) -> Dict[str, str]:
    """
    Gets the mlp names of the models fetched from hadoop.
    """
    if dir_with_models is None:
        return {}

    # Ignore names that do not start with "model_name" to avoid issues
    automatically_fetched_model_names = [name for name in os.listdir(dir_with_models) if "model_name" in name]

    sanitized_model_names = {}
    for name in automatically_fetched_model_names:
        inputs = os.listdir(f"{dir_with_models}/{name}")[0]
        sanitized_model_names[name.replace("model_name=", "")] = f"{dir_with_models}/{name}/{inputs}"

    return sanitized_model_names


def generate_reports(
    name: str,
    input_file: str,
    input_file_sep: str,
    manual_inputs_csv_folder: str,
    automated_inputs_csv_folder: str,
    reports_folder: str,
    output_file: str,
    font_type: str,
):
    """
    Reads both the manually added inputs and the automatically fetched inputs from hdfs.
    If for a model exists both a manual input and an automatically fetched one, the manual one has priority.
    Then, the quality reports for all the fetched models are created.
    """

    print(f"Package version: {__version__}")
    df_models = pd.read_csv(input_file, sep=input_file_sep, keep_default_na=False)

    manually_fetched_model_names = df_models["mlp_name"].values
    automatically_fetched_model_names = get_mlp_model_names(dir_with_models=automated_inputs_csv_folder)

    df_models_autofetched = pd.DataFrame(columns=df_models.columns)
    for mlp_name in automatically_fetched_model_names.keys():
        if mlp_name not in df_models["mlp_name"].values:  # priority to manual inputs

            print(f"Automatically fetched model: {automatically_fetched_model_names[mlp_name]}")
            auto_fetched_df = pd.read_csv(automatically_fetched_model_names[mlp_name], keep_default_na=False)
            model_family_list = list(set(auto_fetched_df["model_family"]))

            if len(model_family_list) > 1:
                raise ValueError(
                    f"More than one model families found for the same model: {model_family_list}. Only one should exist."
                )
            elif len(model_family_list) == 1:
                new_row = {
                    "name": mlp_name,
                    "mlp_name": mlp_name,
                    "business_criticality": auto_fetched_df["business_criticality"][0],
                    "custom_recommendation": "",
                    "date": str(date.today()),
                    "gaps_file": automatically_fetched_model_names[mlp_name],
                    "team": auto_fetched_df["team_name"].values[0],
                    "model_family": model_family_list[0],
                }

            df_models_autofetched = df_models_autofetched.append(new_row, ignore_index=True)

    all_models = df_models.append(df_models_autofetched).reset_index(drop=True)
    all_models.fillna(value="", inplace=True)  # Replace nan with empty strings, to not mess up the model family names

    assessment_path = None
    print("Creating reports...")
    for index, row in all_models.iterrows():
        if name and name != row["name"]:
            continue

        print(f"{index + 1}: Model: {row['name']}")

        team_name = row["team"]
        if not isinstance(team_name, str) or team_name == "":
            team_name = "unknown"  # For cases with NaN team name
        else:
            team_name = format_team_name(team_name)

        # Prevent empty model families to count as families
        model_family = row["model_family"] if row["model_family"] != "" else None

        # find if there is associated gaps csv for this model family
        # scan the models to assess
        models_to_assess = pd.read_csv(input_file, sep=input_file_sep, keep_default_na=False)

        df_w_family_and_gaps = models_to_assess[["model_family", "gaps_file", "business_criticality"]]

        gaps_and_families = dict(zip(df_w_family_and_gaps["model_family"], df_w_family_and_gaps["gaps_file"]))
        gaps_and_criticality = dict(
            zip(df_w_family_and_gaps["model_family"], df_w_family_and_gaps["business_criticality"])
        )
        if model_family in gaps_and_families.keys():
            # There is associated gaps csv with this family
            family_gaps_file = gaps_and_families[model_family]

            family_business_criticality = gaps_and_criticality[model_family]
            assessment = QualityAssessment(
                name=row["name"],
                team=team_name,
                business_criticality=family_business_criticality,
                mlp_name=row["mlp_name"],
                date=row["date"],
                model_family=model_family,
            )
            print(f"{manual_inputs_csv_folder}/{family_gaps_file}")
            assessment.set_gaps_from_csv(f"{manual_inputs_csv_folder}/{family_gaps_file}")

        # The model family does not have an existing assessment, proceed as normal
        else:
            assessment = QualityAssessment(
                name=row["name"],
                team=team_name,
                business_criticality=row["business_criticality"],
                mlp_name=row["mlp_name"],
                date=row["date"],
            )

            if row["mlp_name"] in manually_fetched_model_names:
                assessment.set_gaps_from_csv(f"{manual_inputs_csv_folder}/{row['gaps_file']}")
            else:
                assessment.set_gaps_from_csv(row["gaps_file"])

        custom_recommendation = row["custom_recommendation"]
        if custom_recommendation:
            assessment.custom_recommendation = custom_recommendation

        assessment_path = assessment.create_pdf_report(font_type=font_type)
        html_text = Path(assessment_path.replace(".pdf", ".html")).read_text()
        html5parser = html5lib.HTMLParser()
        html5parser.parse(html_text)
        print(f"   report created: {assessment_path}")

    if assessment_path:
        get_historical_summary(reports_folder).to_csv(output_file, index=False)
