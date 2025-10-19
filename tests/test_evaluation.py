# tests/test_evaluation.py
import joblib
import pandas as pd
from pathlib import Path


TEST_PATH = Path(__file__).parent
MODEL_PATH = TEST_PATH.parent / "model.joblib"
DATA_PATH = TEST_PATH.parent / "data" / "test.csv"

def test_model_accuracy_threshold():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    # Use same column ordering used during training
    feature_cols = ["sepal_length","sepal_width","petal_length","petal_width"]
    X = df[feature_cols]
    y = df["species"]
    
    acc = model.score(X, y)
    print(f"Model accuracy: {acc:.2%}")
    # Check model accuracy is above certain threshold
    threshold = 0.7
    assert acc >= threshold, f"Model accuracy below threshold: {threshold:.2%}"
