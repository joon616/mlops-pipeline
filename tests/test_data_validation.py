# tests/test_data_validation.py
import pytest
import pandas as pd
from pathlib import Path


TEST_PATH = Path(__file__).parent
DATA_PATH = TEST_PATH.parent / "data" / "test.csv"

def test_data_schema_and_nulls():
    df = pd.read_csv(DATA_PATH)
    
    # Check 1: All valid features present
    expected_cols = {"sepal_length","sepal_width","petal_length","petal_width","species"}
    assert set(df.columns) == expected_cols, "All valid features not present"
    
    # Check 2: No null values in dataset
    assert df.notna().all().all(), "Null values present in dataset"
