from aenum import MultiValueEnum
from collections import OrderedDict
import pandas as pd
import pkg_resources

FLAGS = ["red", "orange", "yellow", "white", "green"]
GAPS_WEIGHT = OrderedDict(large=1, small=0.5, no=0)

HTML_COLOR = dict(red="red", orange="orange", yellow="Gold", white="LightGrey", green="green", black="Black")

MAX_MATURITY = 5
MATURITY_STANDARDS = dict(poc=1, production_non_critical=3, production_critical=5)
BUSINESS_CRITICALITY_HTML = dict(
    poc="proof of concept", production_non_critical="production non-critical", production_critical="production critical"
)

SUB_EXPLANATION_URL = ""

MATURITY_LEVELS_URL = ""

ML_BRAINSTORM_URL = ""

CHAR_ORDER = dict(
    utility=0, economy=1, robustness=2, modifiability=3, productionizability=4, comprehensibility=5, responsibility=6
)


class Gap(MultiValueEnum):
    no = "no", 0
    small = "small", 1
    large = "large", 2


REPORTS_FOLDER = "ml_quality_reports"
PLOTS_FOLDER = "ml_quality_plots"
ALL_MODELS_SUMMARY_FOLDER = "all_models_summary"
RADAR_CHART_NAME_SUFFIX = "radar_chart"
GIT_REPO_URL = ""
ASSESSMENTS_URL = f"{GIT_REPO_URL}/ml_quality/assessments"
ALLOWED_IMAGE_FORMATS = ["png", "jpeg", "jpg"]
DATE_FORMAT = "%Y-%m-%d"
BUSINESS_CRITICALITY_STR = "business criticality"

STANDARD_STRING_ANSWERS = ["yes", "partially", "no"]
BINARY_STANDARD_STRING_ANSWERS = ["yes", "no"]
ALLOWED_YES_ANSWERS = ["yes", "y", "fully"]
ALLOWED_NO_ANSWERS = ["n", "no"]

QUESTION_AND_ANSWER_DICT = OrderedDict(
    model_name=dict(question="Model name: ", answer=None),
    team_name=dict(question="Owning team name: ", answer=None),
    business_criticality=dict(question="Business criticality: ", answer=list(MATURITY_STANDARDS.keys())),
    registry_link=dict(question="Registry link: ", answer=None),
    git_link=dict(question="Git link: ", answer=None),
    fullon_link=dict(question="Fullon link: ", answer=None),
    is_fullon_older_than_6_months=dict(
        prerequisite="fullon_link",
        question="Is the last fullon older than 6 months? ",
        answer=BINARY_STANDARD_STRING_ANSWERS,
    ),
    is_understandable=dict(question="Is the ML system understandable? ", answer=STANDARD_STRING_ANSWERS),
    has_comparison_with_baseline=dict(
        question="Does the system outperform a simple baseline? ", answer=BINARY_STANDARD_STRING_ANSWERS
    ),
    is_efficient=dict(question="Is the ML system efficient? ", answer=STANDARD_STRING_ANSWERS),
    is_resilient=dict(question="Is the ML System resilient? ", answer=STANDARD_STRING_ANSWERS),
    is_adaptable=dict(question="Is the ML System adaptable? ", answer=STANDARD_STRING_ANSWERS),
    has_tested_code=dict(question="Does the ML System have tested code? ", answer=STANDARD_STRING_ANSWERS),
    has_repeatable_pipeline=dict(
        question="Does the ML System have repeatable pipeline? ", answer=STANDARD_STRING_ANSWERS
    ),
    has_monitoring=dict(question="Does the ML System have monitoring? ", answer=STANDARD_STRING_ANSWERS),
    has_known_latency=dict(question="Does the ML system have known latency? ", answer=BINARY_STANDARD_STRING_ANSWERS),
    is_traceable=dict(question="Is the ML System traceable? ", answer=STANDARD_STRING_ANSWERS),
    is_explainable=dict(question="Is the ML System explainable? ", answer=STANDARD_STRING_ANSWERS),
    has_applicable_standards=dict(
        question="Does the ML System have PII compliance standards? ", answer=BINARY_STANDARD_STRING_ANSWERS
    ),
    has_input_data_validation=dict(question="Are the input data validated? ", answer=BINARY_STANDARD_STRING_ANSWERS),
    is_vulnerable=dict(question="Is the ML System vulnerable? ", answer=BINARY_STANDARD_STRING_ANSWERS),
)

ANSWER_TO_GAPS = {
    "fully": "no",
    "yes": "no",
    "partially": "small",
    "no": "large",
    None: None,
    True: "no",
    False: "large",
}

ANSWER_TO_BOOL = {None: False, "yes": True, "no": False}


SUBCHARS_ALWAYS_MET = ["fairness"]
SUBCHARS_NOT_TO_BE_ASSESSED = ["readability", "modularity"]

sub_char_path = pkg_resources.resource_stream(__name__, "data/characteristics.csv")
SUB_CHAR_LIST = pd.read_csv(sub_char_path, header=0)["sub_characteristic"].values

STRENGTHS_TO_ADJECTIVES = dict(
    comprehensibility="easy to comprehend",
    economy="profitable",
    modifiability="modifiable",
    productionizability="easy to productionize",
    responsibility="trustworthy",
    robustness="robust",
    utility="useful",
)

WEAKNESSES_TO_ADJECTIVES = dict(
    comprehensibility="comprehensibility",
    economy="profitability",
    modifiability="modifiability",
    productionizability="productionizability",
    responsibility="trustworthiness",
    robustness="robustness",
    utility="usefulness",
)

GAP_FILE_COLUMNS = [
    "sub_characteristic",
    "gap_value",
    "url",
    "reasoning",
    "model_name",
    "team_name",
    "business_criticality",
    "model_family"
]

