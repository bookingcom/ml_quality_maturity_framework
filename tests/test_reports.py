import pytest
from ml_quality.report_automation import generate_reports
import os
import shutil


@pytest.mark.skip(
    reason="This takes a lot of time, run manually by commenting this decorator, to check if create_reports.py works with production data."
)
def test_generate_reports_production_arguments():
    PATH_INPUTS = "../../assessments/models_to_assess.csv"

    PRODUCTION_PATH_INPUT_CSV_FOLDER = "../../assessments/inputs"
    PRODUCTION_AUTOMATED_PATH_INPUT_CSV_FOLDER = "../../assessments/inputs_registry/gaps"

    PATH_REPORTS_FOLDER = "ml_quality_reports"

    OUTPUT_FILE_HISTORICAL_SUMMARY = "test_data/temp_historical_summary.csv"
    assert not os.path.exists(OUTPUT_FILE_HISTORICAL_SUMMARY)

    generate_reports(
        name=None,
        input_file=PATH_INPUTS,
        input_file_sep=",",
        manual_inputs_csv_folder=PRODUCTION_PATH_INPUT_CSV_FOLDER,
        automated_inputs_csv_folder=PRODUCTION_AUTOMATED_PATH_INPUT_CSV_FOLDER,
        reports_folder=PATH_REPORTS_FOLDER,
        output_file=OUTPUT_FILE_HISTORICAL_SUMMARY,
        font_type="verdana",
    )
    assert os.path.exists(OUTPUT_FILE_HISTORICAL_SUMMARY)
    os.remove(OUTPUT_FILE_HISTORICAL_SUMMARY)


def test_generate_reports_model_unknown_team_name_replaced_with_str(capsys):
    PATH_INPUTS = "test_data/test_models_to_assess.csv"
    MANUAL_PATH_INPUT_CSV_FOLDER = "test_data/test_inputs/inputs"
    AUTOMATED_PATH_INPUT_CSV_FOLDER = "test_data/test_inputs/input_registry/gaps"

    PATH_REPORTS_FOLDER = "ml_quality_reports"

    OUTPUT_FILE_HISTORICAL_SUMMARY = "test_data/temp_historical_summary.csv"
    MODEL_TO_ASSESS = "unknown_team_model"
    assert not os.path.exists(OUTPUT_FILE_HISTORICAL_SUMMARY)

    generate_reports(
        name=MODEL_TO_ASSESS,
        input_file=PATH_INPUTS,
        input_file_sep=",",
        manual_inputs_csv_folder=MANUAL_PATH_INPUT_CSV_FOLDER,
        automated_inputs_csv_folder=AUTOMATED_PATH_INPUT_CSV_FOLDER,
        reports_folder=PATH_REPORTS_FOLDER,
        output_file=OUTPUT_FILE_HISTORICAL_SUMMARY,
        font_type="verdana",
    )

    assert os.path.exists(OUTPUT_FILE_HISTORICAL_SUMMARY)
    assert "ml_quality_reports/unknown/unknown_team_model/" in capsys.readouterr().out
    os.remove(OUTPUT_FILE_HISTORICAL_SUMMARY)


def test_generate_report_auto_inferred_models_empty_family():

    PATH_INPUTS = "test_data/test_models_to_assess.csv"
    MANUAL_PATH_INPUT_CSV_FOLDER = "test_data/test_inputs/inputs"
    AUTOMATED_PATH_INPUT_CSV_FOLDER = "test_data/test_inputs/input_registry/gaps"
    PATH_REPORTS_FOLDER = "ml_quality_reports"

    OUTPUT_FILE_HISTORICAL_SUMMARY = "test_data/temp_historical_summary.csv"

    assert not os.path.exists(OUTPUT_FILE_HISTORICAL_SUMMARY)

    generate_reports(
        name=None,
        input_file=PATH_INPUTS,
        input_file_sep=",",
        manual_inputs_csv_folder=MANUAL_PATH_INPUT_CSV_FOLDER,
        automated_inputs_csv_folder=AUTOMATED_PATH_INPUT_CSV_FOLDER,
        reports_folder=PATH_REPORTS_FOLDER,
        output_file=OUTPUT_FILE_HISTORICAL_SUMMARY,
        font_type="verdana",
    )
    assert os.path.exists(OUTPUT_FILE_HISTORICAL_SUMMARY)
    os.remove(OUTPUT_FILE_HISTORICAL_SUMMARY)


def test_create_reports_model_w_multiple_families_raises_error():
    PATH_INPUTS = "test_data/test_models_to_assess.csv"

    MANUAL_PATH_INPUT_CSV_FOLDER = "test_data/test_inputs/inputs"
    AUTOMATED_PATH_INPUT_CSV_FOLDER = (
        "test_data/test_inputs/inputs_registry_w_model_families_wrong_multiple_family_names/gaps"
    )

    PATH_REPORTS_FOLDER = "ml_quality_reports"
    OUTPUT_FILE_HISTORICAL_SUMMARY = "test_data/temp_historical_summary.csv"
    MODEL_TO_ASSESS = "model_w_multiple_families_raises_error"

    assert not os.path.exists(OUTPUT_FILE_HISTORICAL_SUMMARY)

    with pytest.raises(ValueError):
        generate_reports(
            name=MODEL_TO_ASSESS,
            input_file=PATH_INPUTS,
            input_file_sep=",",
            manual_inputs_csv_folder=MANUAL_PATH_INPUT_CSV_FOLDER,
            automated_inputs_csv_folder=AUTOMATED_PATH_INPUT_CSV_FOLDER,
            reports_folder=PATH_REPORTS_FOLDER,
            output_file=OUTPUT_FILE_HISTORICAL_SUMMARY,
            font_type="verdana",
        )


def test_create_reports_skip_model_family_w_no_existing_assessment():
    PATH_INPUTS = "test_data/test_models_to_assess.csv"
    MANUAL_PATH_INPUT_CSV_FOLDER = "test_data/test_inputs/inputs"
    AUTOMATED_PATH_INPUT_CSV_FOLDER = "test_data/test_inputs/input_registry/gaps"
    PATH_REPORTS_FOLDER = "ml_quality_reports"
    OUTPUT_FILE_HISTORICAL_SUMMARY = "test_data/temp_historical_summary.csv"
    MODEL_TO_ASSESS = "model_no_existing_family_assessment"
    assert not os.path.exists(OUTPUT_FILE_HISTORICAL_SUMMARY)

    file_path = f"{AUTOMATED_PATH_INPUT_CSV_FOLDER}/model_name={MODEL_TO_ASSESS}/gaps.csv"
    import pandas as pd
    print(pd.__version__)
    print(file_path)
    initial_gaps = pd.read_csv(file_path)
    initial_criticality = initial_gaps["business_criticality"].values[0]

    generate_reports(
        name=MODEL_TO_ASSESS,
        input_file=PATH_INPUTS,
        input_file_sep=",",
        manual_inputs_csv_folder=MANUAL_PATH_INPUT_CSV_FOLDER,
        automated_inputs_csv_folder=AUTOMATED_PATH_INPUT_CSV_FOLDER,
        reports_folder=PATH_REPORTS_FOLDER,
        output_file=OUTPUT_FILE_HISTORICAL_SUMMARY,
        font_type="verdana",
    )

    historical_summary = pd.read_csv(OUTPUT_FILE_HISTORICAL_SUMMARY)

    final_criticality = historical_summary[historical_summary["model"] == MODEL_TO_ASSESS][
        "business_criticality"
    ].values[0]

    assert initial_criticality == final_criticality
    assert historical_summary[historical_summary["model"] == MODEL_TO_ASSESS]["maturity"].values[0] == 0

    if os.path.isdir(PATH_REPORTS_FOLDER):
        shutil.rmtree(PATH_REPORTS_FOLDER)
    os.remove(OUTPUT_FILE_HISTORICAL_SUMMARY)
