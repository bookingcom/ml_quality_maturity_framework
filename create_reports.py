from ml_quality.report_automation import generate_reports
import os
import click


@click.command()
@click.option("--name", "-n", help="generate report only for this model", type=str, default=None)
@click.option("--input_file", help="file to read models detail from", type=str, default="models_to_assess.csv")
@click.option("--input_file_sep", help="separator for the input file", type=str, default=",")
@click.option(
    "--manual_inputs_csv_folder",
    help="the folder with the manual input csvs",
    type=str,
    default="assessments/inputs"
)
@click.option(
    "--automated_inputs_csv_folder",
    help="the folder with the automatically fetched input csv files",
    type=str,
    default=None,
)
@click.option("--reports_folder", help="folder with all reports", type=str, default="ml_quality_reports")
@click.option(
    "--output_file",
    "-o",
    help="output file with historical summary",
    type=str,
    default="all_models_summary/historical_summary.csv",
)
@click.option("--font_type", help="font type for html/pdf reports", type=str, default="verdana")
def create_reports(
        name: str,
        input_file: str,
        input_file_sep: str,
        manual_inputs_csv_folder: str,
        automated_inputs_csv_folder: str,
        reports_folder: str,
        output_file: str,
        font_type: str,
):
    output_folder = "/".join(output_file.split('/')[:-1])
    os.system(f"mkdir -p {output_folder}")
    generate_reports(
        name,
        input_file,
        input_file_sep,
        manual_inputs_csv_folder,
        automated_inputs_csv_folder,
        reports_folder,
        output_file,
        font_type,
    )


if __name__ == "__main__":
    create_reports()
