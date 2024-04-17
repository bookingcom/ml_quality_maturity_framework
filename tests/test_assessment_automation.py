import pandas as pd
from ml_quality.assessment_automation import write_gaps
from ml_quality.constants import GAP_FILE_COLUMNS
from pathlib import Path
import shutil
import os

def test_write_gaps_from_registry():
    df_registry = pd.read_csv("../ml_registry_example.csv")
    output_dir = Path("test_gaps_from_registry")
    output_dir.mkdir(parents=True, exist_ok=True)
    for m in df_registry.model_name:
        df_model = df_registry[df_registry.model_name == m]
        df_gaps = write_gaps(df_model.iterrows(), df_registry.columns)
        pd.DataFrame(df_gaps, columns=GAP_FILE_COLUMNS).to_csv(output_dir / f"gaps_{m}.csv")
    assert os.path.exists(output_dir / "gaps_model1.csv")
    assert os.path.exists(output_dir / "gaps_model2.csv")
    shutil.rmtree(output_dir)
