import setuptools
from pathlib import Path
import re

with open(Path("prioritization_framework/README.md").resolve(), "r") as fh:
    long_description = fh.read()

with open("ml_quality_2/ml_quality/_version.py", "r") as f:
    version_content = f.read()
version = re.search(r"^__version__ = \"(.+)\"$", version_content, flags=re.MULTILINE).group(1)

setuptools.setup(
    name='ml_quality',
    version=version,
    author=["anonymous author"],
    author_email=[""],
    description="Python tools to perform assessment on quality of ML systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(exclude=["tests"]),
    classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
     ],
    install_requires=["plotly>=5.5.0", "kaleido", "pandas", "numpy", "aenum", "pdfkit"],
    setup_requires=["setuptools", "setuptools-git", "wheel"],
    package_data={'': ['data/*.csv']},
    include_package_data=True
)
