# Machine Learning Quality and Maturity Package

This repository houses the Machine Learning (ML) Quality and Maturity Framework package developed at Booking.com.\
The structure of the repository is delineated as follows:

* The primary class, located in `assessments.py`, serves the function of generating the final quality and maturity score.\
Additionally, it creates a report detailing the technical gaps identified, along with recommended best practices for their mitigation.\
Initialization of this class involves parsing a `.csv` file containing technical gap information for each quality sub-characteristic.

* The file `assessment_automation.py` is dedicated to inferring the gaps present in a model based on data extracted from the ML registry. 
The ML registry serves as a database housing pertinent information regarding the models under analysis.


* The file `report_automation.py`, provides utilities for automating the generation of multiple reports, 
as well as the aggregation of historical evaluation data into a single dataset.

* The `practice_prioritization` directory contains code responsible for prioritizing the sequence of best practices 
to be applied in addressing identified technical gaps.

### Set up the environment and run the tests
To verify the functionality of the repository, create a Python virtual environment, following the instructions below.
The package was tested using `python3.9`

```
python -m ensurepip --upgrade  # Make sure you have pip installed
pip install virtualenv
python -m venv venv
source ./venv/bin/activate  # Activates the virtual environment
pip install -r requirements.txt 
```
To verify everything is working properly you can run the test suite of the package.\
You can do this by typing `tox` from the package root directory.

### Generate a quality report
To generate a quality and maturity report, follow the example provided in the `create_quality_report.ipynb` notebook.
To run the notebook first create a Jupyter kernel from the virtual environment:

```
python -m ipykernel install --user --name=venv
jupyter notebook
```

### Generate multiple reports from a list of models with technical gaps
To generate multiple reports for a list of models, utilize the script `create_reports.py`. The script generates reports for all models listed in `models_to_assess.csv`.\
It also consolidates previous evaluations into a historical report. You can run the script by typing
```
python create_reports.py
```
The file `models_to_assess.csv` contains the list of models you want to be assessed, the path of the file with technical gaps,\
the date of the evaluation, the team owning the model and the model family (in case the same evaluation is valid for more than a single ML system).
The script will create a `.csv` file with the historical data of all the evaluations stored in the repo. 
The file created will be under `all_models_summary/historical_summary.csv`
