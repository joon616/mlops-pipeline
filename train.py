import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = "data/train.csv"

df = pd.read_csv(DATA_PATH)
X = df[["sepal_length","sepal_width","petal_length","petal_width"]]
y = df["species"]

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)
joblib.dump(model, "model.joblib")
print("Training complete! Model saved as 'model.joblib'")
