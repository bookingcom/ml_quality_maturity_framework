python${1:-3.9} -m ensurepip --upgrade  # Make sure you have pip installed
pip${1:-3.9} install virtualenv
python${1:-3.9} -m venv venv
source ./venv/bin/activate  # Activates the virtual environment
pip install -r requirements.txt -i https://pypi.python.org/simple/
