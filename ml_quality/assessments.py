import logging
from typing import Dict, Union, Optional, List, Tuple
import warnings

import pandas as pd
import pkg_resources
import os
from datetime import datetime
import pickle
import re
import practice_prioritization
from practice_prioritization.prioritization_framework import prioritize_selected_practices
from operator import itemgetter

from .utils import (
    set_pandas_value,
    plot_radar_chart,
    generate_report_text,
    convert_html_to_pdf,
    get_column_value_from_row,
    get_pkg_version,
    add_is_max_col,
    check_date_format,
    load_obj_from_pickle,
    propagate_last_value,
    read_practices_to_maturity_levels_from_csv,
)

from .constants import (
    GAPS_WEIGHT,
    REPORTS_FOLDER,
    MATURITY_STANDARDS,
    MAX_MATURITY,
    ASSESSMENTS_URL,
    ALLOWED_IMAGE_FORMATS,
    DATE_FORMAT,
    Gap,
    STRENGTHS_TO_ADJECTIVES,
    WEAKNESSES_TO_ADJECTIVES,
)

logger = logging.getLogger()

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'


def format_team_name(team: str):
    """
    Reformat team names to avoid problem in folder creation
    """
    return (
        team.replace(" - ", "_")
        .replace(" & ", "_and_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace('"', "")
        .replace("\\", "")
        .lower()
    )


def get_historical_summary(
        path: str, pickle_file_name: str = "assessment.pickle", summary_file_name: str = "summary.csv"
) -> pd.DataFrame:
    """
    Function to get historical score from individual summary
    Args:
        path: path of the reports
        pickle_file_name: name of the pickle file
        summary_file_name: name of the summary file with gaps
    Returns:
        dataframe with all reports done so far
    """
    rows = []
    if not os.path.isdir(path):
        raise ValueError(f"Path: {path} is not a directory!")

    for root, dirs, files in os.walk(path):
        for f in files:
            if not f.endswith(".pickle"):
                continue
            pickle_file_path = os.path.join(root, f)

            # find last date in the path
            date = re.findall(r"\d{4}-\d{2}-\d{2}", pickle_file_path)[-1]
            name = pickle_file_path.split(f"/{date}")[0].split("/")[-1]
            assessment_obj = load_obj_from_pickle(pickle_file_path)

            summary_csv_path = pickle_file_path.replace(pickle_file_name, summary_file_name)
            df = pd.read_csv(summary_csv_path)
            score = QualityAssessment.compute_quality_score(df["gap_value"])

            rows.append(
                {
                    "model": name,
                    "team": getattr(assessment_obj, "team", ""),
                    "mlp_name": getattr(assessment_obj, "mlp_name", ""),
                    "date": assessment_obj.date,
                    "score": score,
                    "maturity": assessment_obj.maturity,
                    "business_criticality": assessment_obj.business_criticality,
                    "expected_maturity": MATURITY_STANDARDS[assessment_obj.business_criticality],
                    "report_url": getattr(assessment_obj, "report_url", ""),
                    "radar_chart_url": getattr(assessment_obj, "radar_chart_url", ""),
                    "report_summary": getattr(assessment_obj, "custom_recommendation", ""),
                }
            )

    historical_summary = add_is_max_col(pd.DataFrame(rows), id_cols=["model"], sort_col="date", is_max_col="latest")

    historical_summary = propagate_last_value(
        df=historical_summary, col_to_propagate="mlp_name", sort_cols=["model", "date"], groupby_col="model"
    )

    historical_summary = propagate_last_value(
        df=historical_summary, col_to_propagate="team", sort_cols=["model", "date"], groupby_col="model"
    )
    return historical_summary


def read_summary_from_csv(path: str = None) -> pd.DataFrame:
    if path is not None:
        stream = path
    else:
        stream = pkg_resources.resource_stream(__name__, "data/characteristics.csv")
    return pd.read_csv(stream)


def read_maturity_requirements_from_csv() -> pd.DataFrame:
    stream = pkg_resources.resource_stream(__name__, "data/maturity_levels.csv")
    return pd.read_csv(stream)


def read_flag_colors_from_csv(as_num: bool = False) -> pd.DataFrame:
    stream = pkg_resources.resource_stream(__name__, "data/flag_colors.csv")
    flags_df = pd.read_csv(stream)
    if as_num:
        for col_name in ["allowed_gap", "gap_value"]:
            flags_df[col_name] = flags_df[col_name].apply(lambda x: Gap(x).values[1])
    return flags_df


def read_recommendations_from_csv() -> pd.DataFrame:
    stream = pkg_resources.resource_stream(__name__, "data/practice_recommendations.csv")
    return pd.read_csv(stream)


initial_summary = read_summary_from_csv()
maturity_requirements = read_maturity_requirements_from_csv()
flag_colors = read_flag_colors_from_csv(as_num=True)
maturity_levels = {x: int(x.split("_")[1]) for x in read_maturity_requirements_from_csv().columns if "level" in x}


class QualityAssessment:
    """
    A class for ML quality assessment framework
    """

    def __init__(
            self,
            name: str,
            mlp_name: Optional[str] = None,
            business_criticality: str = "poc",
            model_family: Optional[str] = None,
            team: Optional[str] = None,
            date: Optional[str] = str(datetime.now().date()),
    ):

        if team is None:
            team = ""
        self._name = name
        self._date = date
        self._team = format_team_name(team)
        self.mlp_name = mlp_name
        self._business_criticality = business_criticality
        self._model_family = model_family
        self._summary = read_summary_from_csv()
        self._flags = self._set_all_levels_flags()
        self._flags_reasons = self._init_flags_reasons()
        self._maturity = max(maturity_levels.values())
        self._scores = self._compute_scores()
        self._quality_score = self._get_quality_score()
        self._radar_chart_file_format = "png"
        self.custom_recommendation = None
        self.pickle_file_name = "assessment"
        if not check_date_format(date, date_fmt=DATE_FORMAT):
            raise ValueError(f"Enter date in the format: {DATE_FORMAT}")
        else:
            self._date = date
        self._report_file_name = "report"
        self._radar_chart_file_name = "radar_chart"
        self._model_folder = self._set_model_folder()
        self.report_url = self._set_report_url()
        self.radar_chart_url = self._set_radar_chart_url()

    def _set_model_folder(self):
        if self._team == "":
            return f"{REPORTS_FOLDER}/{self._name.lower()}/{self._date}".replace(" ", "_")
        else:
            return f"{REPORTS_FOLDER}/{self._team}/{self._name.lower()}/{self._date}".replace(" ", "_").replace(
                "&", "_and_"
            )

    def _set_report_url(self):
        return f"{ASSESSMENTS_URL}/{self._model_folder}/{self._report_file_name}.pdf"

    def _set_radar_chart_url(self):
        return f"{ASSESSMENTS_URL}/{self._model_folder}/{self._radar_chart_file_name}.{self._radar_chart_file_format}"

    @property
    def team(self):
        return self._team

    @team.setter
    def team(self, team_name):
        self._team = format_team_name(team_name)
        self._model_folder = self._set_model_folder()
        self.report_url = self._set_report_url()
        self.radar_chart_url = self._set_radar_chart_url()

    @property
    def report_file_name(self):
        return self._report_file_name

    @report_file_name.setter
    def report_file_name(self, filename: str):
        self._report_file_name = filename
        self.report_url = self._set_report_url()

    @property
    def radar_chart_file_name(self):
        return self._radar_chart_file_name

    @radar_chart_file_name.setter
    def radar_chart_file_name(self, filename: str):
        self._radar_chart_file_name = filename
        self.radar_chart_url = self._set_radar_chart_url()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name
        self._model_folder = self._set_model_folder()
        self.report_url = self._set_report_url()
        self.radar_chart_url = self._set_radar_chart_url()

    @property
    def business_criticality(self):
        return self._business_criticality

    @property
    def model_family(self):
        return self._model_family

    @model_family.setter
    def model_family(self, model_family: str):
        self._model_family = model_family

    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, date: str):
        self._date = date

    @business_criticality.setter
    def business_criticality(self, business_criticality: str):
        if business_criticality not in list(MATURITY_STANDARDS.keys()):
            raise ValueError(f"business criticality level {business_criticality} not recognized")
        self._business_criticality = business_criticality

    @property
    def model_folder(self):
        return self._model_folder

    @model_folder.setter
    def model_folder(self, model_folder_path: str):
        if " " in model_folder_path:
            raise ValueError("Please enter a path without spaces")
        self._model_folder = model_folder_path
        self.report_url = self._set_report_url()
        self.radar_chart_url = self._set_radar_chart_url()

    @property
    def summary(self):
        return self._summary

    @summary.setter
    def summary(self, updated_summary: pd.DataFrame):
        if type(updated_summary) != pd.DataFrame:
            raise ValueError("Model summary can be updated only with a pandas dataframe")
        standard_columns = sorted(initial_summary.columns)
        if sorted(updated_summary.columns) != standard_columns:
            raise ValueError(f"New model summary has wrong number of columns, expected: {standard_columns}")
        if len(updated_summary.index) != len(initial_summary.index):
            raise ValueError("New summary is missing sub-characteristics")
        self._summary = updated_summary

    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, date_str: str):
        if not check_date_format(date_str, date_fmt=DATE_FORMAT):
            raise ValueError(f"Date not in the format {DATE_FORMAT}")
        self._date = date_str
        self._model_folder = self._set_model_folder()
        self.report_url = self._set_report_url()
        self.radar_chart_url = self._set_radar_chart_url()

    def get_flags(self, maturity: int = None, color: str = None):
        df_flags = self.flags.copy()
        if maturity is not None:
            if maturity == 0:
                maturity = 1
            df_flags = df_flags[df_flags["maturity"] == f"level_{maturity}"]
        if color is not None:
            df_flags = df_flags[df_flags["flag"] == color]
        return df_flags

    @property
    def maturity(self):
        return self._maturity

    @property
    def flags(self):
        return self._flags

    @property
    def flags_reasons(self):
        return self._flags_reasons

    @property
    def scores(self):
        return self._scores

    @property
    def quality_score(self):
        return self._quality_score

    @property
    def custom_recommendation(self):
        return self._custom_recommendation

    @custom_recommendation.setter
    def custom_recommendation(self, recommendation: str):
        if recommendation is not None:
            if len(recommendation) < 5 or " " not in recommendation:
                raise ValueError(f"Recommendation: '{recommendation}' not informative, remove it?")
        self._custom_recommendation = recommendation

    @flags_reasons.setter
    def flags_reasons(self, reasons: Dict[str, str]):
        if set(reasons.keys()) != set(self.get_sub_characteristics()):
            raise ValueError("You can't update a reason dictionary with a different set of sub-characteristics")
        self._flags_reasons = reasons

    @property
    def radar_chart_file_format(self):
        return self._radar_chart_file_format

    @radar_chart_file_format.setter
    def radar_chart_file_format(self, image_format):
        if image_format not in ALLOWED_IMAGE_FORMATS:
            raise ValueError(f"Supported image formats are: {ALLOWED_IMAGE_FORMATS}")
        self._radar_chart_file_format = image_format
        self.radar_chart_url = self._set_radar_chart_url()

    @staticmethod
    def compute_quality_score(gap_values: List[int]) -> int:
        total_gaps_sum = sum(gap_values)
        n_subcharacteristics = len(gap_values)

        largest_gap_value = Gap("large").values[1]
        quality_score = 100 * (
                (largest_gap_value * n_subcharacteristics - total_gaps_sum) / (largest_gap_value * n_subcharacteristics)
        )

        return int(quality_score)

    def get_sub_characteristics(self):
        return self.summary.sub_characteristic.tolist()

    def get_characteristics(self):
        return self.summary.characteristic.unique().tolist()

    def get_characteristics_with_sub(self) -> pd.DataFrame:
        return self.summary[["characteristic", "sub_characteristic"]]

    def get_char_from_sub_char(self, sub_char: str):
        return get_column_value_from_row(
            df=self.summary, filter_col="sub_characteristic", filter_val=sub_char, value_col="characteristic"
        )

    def check_gap_validity(self, sub_char: str, gap: Union[str, int]):
        if sub_char not in self.get_sub_characteristics():
            raise ValueError(f"Sub-characteristic: {sub_char} not recognized!")
        try:
            Gap(gap)
        except ValueError:
            raise ValueError(f"Gap not recognized {gap} for sub-characteristic {sub_char}")

    def set_gap(self, sub_char: str, gap: Union[str, int]):

        self.check_gap_validity(sub_char=sub_char, gap=gap)
        updated_summary = set_pandas_value(
            df=self.summary, index=("sub_characteristic", sub_char), column=("gap_value", Gap(gap).values[1])
        )
        self.summary = updated_summary
        self._set_flags()
        self._set_maturity()
        self._set_scores()
        self._set_quality_score()

    def set_gaps(self, sub_char_with_gaps: Dict[str, Union[str, int]]):
        for sub_char, gap in sub_char_with_gaps.items():
            self.set_gap(sub_char, gap)

    def _init_flags_reasons(self):
        return {f: "" for f in self.flags["sub_characteristic"].unique()}

    def _set_single_level_flags(self, maturity: int) -> pd.DataFrame:
        summ = self.summary.copy()
        req = maturity_requirements[["sub_characteristic", f"level_{maturity}"]]
        req = req.rename(columns={f"level_{maturity}": "allowed_gap"})
        summ_wt_req = summ.merge(req, on="sub_characteristic", how="inner")
        flags = summ_wt_req.merge(flag_colors, on=["gap_value", "allowed_gap"], how="inner")
        return flags[["characteristic", "sub_characteristic", "gap_value", "allowed_gap", "flag"]]

    def _set_all_levels_flags(self) -> pd.DataFrame:
        all_flags = []
        for level in maturity_levels.keys():
            flags = self._set_single_level_flags(maturity=level.split("_")[1])
            flags["maturity"] = level
            all_flags.append(flags)
        return pd.concat(all_flags)

    def set_flag_reason(self, sub_char: str, reason: str):
        self.flags_reasons.update({sub_char: reason})

    def set_flags_reasons_from_csv(self, csv_path: str, sub_char_col_name: str, reason_col_name: str):
        self.flags_reasons.update(dict(pd.read_csv(csv_path)[[sub_char_col_name, reason_col_name]].values))

    def set_gaps_from_csv(
            self,
            csv_path: str,
            # column_names: Optional[List[str]] = None,
            sub_char_col_name: str = "sub_characteristic",
            gap_col_name: str = "gap_value",
            reason_col_name: Optional[str] = "reasoning",
    ):
        for sub_char_gaps in pd.read_csv(csv_path).fillna({"gap_value": "large"}).iterrows():
            sub_char = sub_char_gaps[1][sub_char_col_name]
            gap = sub_char_gaps[1][gap_col_name]
            self.set_gap(sub_char, gap)
            if reason_col_name:
                reason = sub_char_gaps[1][reason_col_name]
                self.set_flag_reason(sub_char=sub_char, reason=reason)

    def find_maturity(self) -> int:
        for lvl_name, lvl_num in maturity_levels.items():
            df_maturity = self.flags[self.flags.maturity == lvl_name][["gap_value", "allowed_gap", "maturity"]]
            df_maturity["meet_requirement"] = df_maturity["allowed_gap"] >= df_maturity["gap_value"]
            meet_requirement = df_maturity["meet_requirement"].unique()
            if not all(meet_requirement):
                return lvl_num - 1
        return MAX_MATURITY

    def _set_maturity(self):
        self._maturity = self.find_maturity()

    def _set_flags(self):
        self._flags = self._set_all_levels_flags()

    def _set_scores(self):
        self._scores = self._compute_scores()

    def _compute_scores(self) -> pd.DataFrame:
        scores = self.summary.copy()
        scores["weight"] = scores["gap_value"].apply(lambda x: Gap(x).value).map(GAPS_WEIGHT)
        scores = scores.groupby("characteristic").agg(score=("weight", "mean"))
        scores["score"] = 1 - scores["score"]
        return scores.reset_index()

    def _get_quality_score(self) -> int:
        sub_characteristic_scores = self.summary.copy()

        return self.compute_quality_score(gap_values=sub_characteristic_scores["gap_value"])

    def _set_quality_score(self):
        self._quality_score = self._get_quality_score()

    def reset(self):
        self.summary = initial_summary
        self._set_flags()
        self._set_maturity()
        self._set_scores()
        self._set_quality_score()

    def plot_chart(
            self,
            figtitle: Optional[str] = "",
            show_figure: bool = False,
            save_figure: bool = True,
    ):
        """
        Function to create radar chart for Quality Assessment instance
        Args:
            figtitle: title for the radar chart
            show_figure: show figure Y/N
            save_figure: persist a .png Y/N
        """
        df_scores = self.scores.copy()
        df_scores["characteristic"] = df_scores["characteristic"].apply(
            lambda x: x.capitalize()
        )  # todo: DO NOT CHANGE THE ACTUAL SCORES HERE!!!!!!!!

        return plot_radar_chart(
            df=df_scores,
            r="score",
            theta="characteristic",
            range_r=[0, max(GAPS_WEIGHT.values())],
            plots_folder=self.model_folder,
            show_figure=show_figure,
            save_figure=save_figure,
            figtitle=figtitle,
            save_name=self.radar_chart_file_name,
        )

    def generate_strengths_and_weaknesses_summary(
            self, current_maturity: int, expected_maturity: int, quality_threshold: float = 0.75
    ) -> str:
        """
        Identifies which characteristics are considered as strengths and which need improvement. Use those
        to generate an overview of the system's quality in the report.
        """
        strengths_summary = self.get_strengths_summary(
            current_maturity=current_maturity, expected_maturity=expected_maturity, quality_threshold=quality_threshold
        )
        weaknesses_summary = self.get_weaknesses_summary(quality_threshold=quality_threshold)

        strengths_and_weaknesses_summary = strengths_summary + weaknesses_summary

        return strengths_and_weaknesses_summary

    def get_strengths_summary(
            self, current_maturity: int, expected_maturity: int, quality_threshold: float = 0.75
    ) -> str:
        strengths = self.get_strong_or_weak_characteristics(strong=True, quality_threshold=quality_threshold)

        if len(strengths) == 0:
            strengths_summary = ""
        elif len(strengths) == 1:
            strengths_summary = f"The system is quite {STRENGTHS_TO_ADJECTIVES[strengths[0]]}."
        elif 2 <= len(strengths) < len(self.get_characteristics()):
            strengths_summary = "The system is quite "

            strengths_summary += ", ".join([STRENGTHS_TO_ADJECTIVES[strength] for strength in strengths[:-1]])

            strengths_summary += f" and {STRENGTHS_TO_ADJECTIVES[strengths[-1]]}."

        else:
            if current_maturity >= expected_maturity:
                strengths_summary = "The system is of very high quality and at the expected maturity. Well done!"
            else:
                strengths_summary = (
                    f"The system is of high quality, however the maturity is not yet at level "
                    f"{expected_maturity} which is the expected one. This is because some "
                    f"prerequisites are missing, see below for more details."
                )

        return strengths_summary

    def get_weaknesses_summary(self, quality_threshold: float = 0.75) -> str:

        weaknesses = self.get_strong_or_weak_characteristics(strong=False, quality_threshold=quality_threshold)

        if len(weaknesses) == 0:
            weakness_summary = ""
        elif len(weaknesses) == 1:
            weakness_summary = f" However, its {WEAKNESSES_TO_ADJECTIVES[weaknesses[0]]} has room for improvement."
        elif 2 <= len(weaknesses) < 7:
            weakness_summary = " However, quality aspects of "

            weakness_summary += ", ".join([WEAKNESSES_TO_ADJECTIVES[weakness] for weakness in weaknesses[:-1]])

            weakness_summary += (
                f" and {WEAKNESSES_TO_ADJECTIVES[weaknesses[-1]]} can be improved. " f"See below for more details."
            )

        else:
            weakness_summary = (
                "The system has room for improvement in multiple quality attributes. " "See below for more details."
            )

        return weakness_summary

    def get_strong_or_weak_characteristics(self, strong: bool, quality_threshold: float = 0.75) -> List[str]:
        if strong:
            return self.scores[self.scores["score"] >= quality_threshold]["characteristic"].to_list()
        else:
            return self.scores[self.scores["score"] < quality_threshold]["characteristic"].to_list()

    def get_recommendations_per_maturity_level(self) -> Dict:
        """
        Retrieve recommendation for building report

        """
        current_maturity_level = self.maturity
        expected_maturity_level = MATURITY_STANDARDS[self.business_criticality]

        practices_per_maturity_level = dict()

        gaps_per_mat_level = self.get_flags(maturity=current_maturity_level).copy()

        latest_gaps = gaps_per_mat_level[["sub_characteristic", "gap_value"]]

        latest_gaps_dict = dict(zip(latest_gaps["sub_characteristic"], latest_gaps["gap_value"]))

        first_iteration = True
        for maturity_level in range(current_maturity_level + 1, 6):
            gap_values_next_mat_level = self.get_flags(maturity=maturity_level).copy()

            if not first_iteration:
                gap_values_next_mat_level["gap_value"] = gap_values_next_mat_level.apply(
                    lambda x: gaps_to_act_upon[gaps_to_act_upon["sub_characteristic"] == x.sub_characteristic][
                        "allowed_gap"
                    ].values[0]
                    if x["sub_characteristic"] in gaps_to_act_upon["sub_characteristic"].values
                    else latest_gaps_dict[x["sub_characteristic"]],
                    axis=1,
                )

            if maturity_level <= expected_maturity_level:
                gap_values_next_mat_level["is_expected_level"] = True
            else:
                gap_values_next_mat_level["is_expected_level"] = False

            gaps_to_act_upon = gap_values_next_mat_level[
                gap_values_next_mat_level["gap_value"] > gap_values_next_mat_level["allowed_gap"]
                ]
            practices_to_fix_gaps = self.get_practices_to_fix_specific_gaps(gaps_to_act_upon=gaps_to_act_upon)

            practices_per_maturity_level[maturity_level] = practices_to_fix_gaps

            # update gaps, as if we fixed them
            for _, row in gaps_to_act_upon.iterrows():
                latest_gaps_dict[row["sub_characteristic"]] = row["allowed_gap"]

            first_iteration = False

        practices_per_mat_level_w_colors = self.get_flag_colours_per_mat_level(
            practices_per_maturity_level=practices_per_maturity_level
        )

        return practices_per_mat_level_w_colors

    @staticmethod
    def get_flag_colours_per_mat_level(practices_per_maturity_level: Dict) -> Dict:
        """
        level 2: for expected -> red
        level 3: for expected -> orange
        level 4: improve further -> yellow
        level 5: improve further -> yellow

        if expected and 1st -> red
        if expected and not 1st -> orange

        if not expected -> yellow
        """

        is_first = True
        for level, gaps in practices_per_maturity_level.items():
            if not gaps.empty:
                if is_first and gaps["is_expected_level"].values[0] == True:
                    gaps["flag_colour"] = "red"
                elif not is_first and gaps["is_expected_level"].values[0] == True:
                    gaps["flag_colour"] = "orange"
                elif not gaps["is_expected_level"].values[0]:
                    gaps["flag_colour"] = "yellow"

                is_first = False

        return practices_per_maturity_level

    def get_practices_to_fix_specific_gaps(self, gaps_to_act_upon: pd.DataFrame) -> pd.DataFrame:
        practices_to_maturity_levels = read_practices_to_maturity_levels_from_csv()

        if gaps_to_act_upon.empty:
            return gaps_to_act_upon

        gaps_to_act_upon["recommended_practice"] = gaps_to_act_upon.apply(
            lambda x: practices_to_maturity_levels[
                (practices_to_maturity_levels["sub_characteristic"] == x.sub_characteristic)
                & (practices_to_maturity_levels["gap_level_to_reach"] == x.allowed_gap)
                ]["practice_name"].values[0],
            axis=1,
        )
        gaps_to_act_upon["recommended_practice_explanation"] = gaps_to_act_upon.apply(
            lambda x: practices_to_maturity_levels[
                (practices_to_maturity_levels["sub_characteristic"] == x.sub_characteristic)
                & (practices_to_maturity_levels["gap_level_to_reach"] == x.allowed_gap)
                ]["practice_explanation"].values[0],
            axis=1,
        )

        gaps_to_act_upon = self.add_rank_in_practices(practices_to_fix_gaps=gaps_to_act_upon).sort_values(by="rank")
        # Sort gaps to act upon by rank!
        return gaps_to_act_upon

    def add_rank_in_practices(self, practices_to_fix_gaps: pd.DataFrame) -> pd.DataFrame:
        selected_practices_w_rank = self.prioritize_recommendations_within_maturity_level(
            practices_to_fix_gaps=practices_to_fix_gaps
        )
        # If comparison with baseline is needed, it should be ranked first. This is why we boost by 1000.
        for practice in selected_practices_w_rank:
            if practice[0] == "compare with a baseline":
                selected_practices_w_rank.remove(practice)
                temp = list(practice)
                temp[1] += 1000
                practice = tuple(temp)
                selected_practices_w_rank.append(practice)
                selected_practices_w_rank = sorted(selected_practices_w_rank, key=itemgetter(1), reverse=True)
                break

        rank = 1
        for practice in selected_practices_w_rank:
            practices_to_fix_gaps.loc[(practices_to_fix_gaps["recommended_practice"] == practice[0]), "rank"] = int(
                rank
            )
            rank += 1

        practices_to_fix_gaps = practices_to_fix_gaps.astype({"rank": "int"})

        return practices_to_fix_gaps

    @staticmethod
    def prioritize_recommendations_within_maturity_level(
            practices_to_fix_gaps: pd.DataFrame,
    ) -> List[Tuple[str, float]]:
        SUB_CHAR_LIST = practices_to_fix_gaps["sub_characteristic"].values
        SUB_CHAR_LIST = [elem.replace("_", "-") for elem in SUB_CHAR_LIST]  # _ is - in the prioritization package

        ASSESSMENT_PRACTICES_TO_RANK = set(practices_to_fix_gaps["recommended_practice"].values)

        return prioritize_selected_practices(
            sub_characteristics=set(SUB_CHAR_LIST),
            weights=None,
            practices_to_rank=ASSESSMENT_PRACTICES_TO_RANK,
        )

    @staticmethod
    def check_if_practices_are_scored(practices_to_be_recommended: List[str]):
        """
        Checks if the practices to be recommended have been scored in the prioritization_framework. A practice
        not scored will not be ranked correctly.
        """
        scored_practices_path = str(
            os.path.dirname(practice_prioritization.__file__) + "/score_collection/merged/merged_scores.csv"
        )
        merged_scores = pd.read_csv(scored_practices_path, header=0)
        scored_practices = list(merged_scores["Practice"].unique())

        for practice in practices_to_be_recommended:
            if practice not in scored_practices:
                raise Warning(f"The practice {practice} does not have a score in the prioritization_framework!")

    def find_already_implemented_practices(self) -> List[str]:
        gap_values_next_mat_level = self.get_flags(maturity=self.maturity).copy()
        practices_to_maturity_levels = read_practices_to_maturity_levels_from_csv()

        already_implemented_practices = []
        for _, row in gap_values_next_mat_level.iterrows():
            if row["gap_value"] == 0:  # add all the associated practices with this subchar
                practices_implemented_for_subchar = practices_to_maturity_levels[
                    practices_to_maturity_levels["sub_characteristic"] == row["sub_characteristic"]
                    ]["practice_name"].to_list()
                already_implemented_practices += practices_implemented_for_subchar
            elif row["gap_value"] == 1:  # add only level 1 practice
                practices_implemented_for_subchar = practices_to_maturity_levels[
                    (practices_to_maturity_levels["sub_characteristic"] == row["sub_characteristic"])
                    & (practices_to_maturity_levels["gap_level_to_reach"] == 1)
                    ]["practice_name"].to_list()
                already_implemented_practices += practices_implemented_for_subchar

        unique_practices = list(set([elem for elem in already_implemented_practices if isinstance(elem, str)]))
        return list(set(unique_practices))

    @staticmethod
    def find_already_satisfied_subcharacteristics(already_implemented_practices: List[str]) -> Dict[str, str]:
        practices_to_maturity_levels = read_practices_to_maturity_levels_from_csv()

        already_implemented_practices = [
            value for value in already_implemented_practices if value not in ["bias assessment"]
        ]

        all_practices_and_subchars = dict(
            zip(practices_to_maturity_levels["practice_name"], practices_to_maturity_levels["sub_characteristic"])
        )

        # Filter only the implemented ones
        return {key: all_practices_and_subchars[key] for key in already_implemented_practices}

    def create_html_report(self, font_type: str = "menlo") -> str:
        """
        Create final report in html format
        Args:
            font_type: Type of the font of the report
        Returns:
            string with html report path
        """

        expected_maturity_level = MATURITY_STANDARDS[self.business_criticality]

        practices_to_be_recommended = read_practices_to_maturity_levels_from_csv()["practice_name"].values
        self.check_if_practices_are_scored(practices_to_be_recommended=practices_to_be_recommended)

        recommendations = self.get_recommendations_per_maturity_level()

        radar_chart_path = self.plot_chart(figtitle=None)
        radar_chart_path = radar_chart_path.split("/")[-1]

        version_file_path = os.path.dirname(os.path.realpath(__file__))
        version = get_pkg_version(version_file_path=f"{version_file_path}/_version.py")

        text = generate_report_text(
            model_name=self.name,
            model_criticality=self.business_criticality,
            model_family=self._model_family,
            model_maturity=self.maturity,
            quality_score=self.quality_score,
            radar_chart_path=radar_chart_path,
            reason_dict=self.flags_reasons,
            recommendation_txt=self.custom_recommendation,
            summary=self.generate_strengths_and_weaknesses_summary(
                current_maturity=self.maturity, expected_maturity=expected_maturity_level
            ),
            recommendations=recommendations,
            already_satisfied_subchars=self.find_already_satisfied_subcharacteristics(
                already_implemented_practices=self.find_already_implemented_practices()
            ),
            report_date=self.date,
            font_type=font_type,
            max_maturity=MAX_MATURITY,
            version=version,
        )

        os.system(f"mkdir -p ./{self.model_folder}")
        html_file_name = f"{self.model_folder}/{self.report_file_name}.html"
        f = open(f"./{html_file_name}", "w+")
        f.write(text)
        f.close()

        self.save_summary()
        self.save_to_pickle()
        return html_file_name

    def create_pdf_report(self, font_type: str = "menlo") -> str:
        """
        Create report in pdf format
        Args:
            font_type: Type of the font of the report
        Returns:
            string with pdf report path
        """
        return convert_html_to_pdf(html_file_path=self.create_html_report(font_type=font_type))

    def save_summary(self, summary_file_name: str = "summary", sep=","):
        """
        Function to save to .csv summary class property
        Args:
            summary_file_name: name of the .csv summary file
            sep: separator
        """
        os.system(f"mkdir -p ./{self.model_folder}")
        self.summary.to_csv(f"./{self.model_folder}/{summary_file_name}.csv", sep=sep, index=False)

    def save_to_pickle(self):
        """
        Function for saving to pickle the class instance, for reproducibility
        """
        os.system(f"mkdir -p {self.model_folder}")
        with open(f"{self.model_folder}/{self.pickle_file_name}.pickle", "wb") as pickle_file:
            pickle.dump(self, pickle_file)
