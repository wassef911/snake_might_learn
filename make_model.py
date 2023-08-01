import joblib
import pandas as pd
from decouple import config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.utils import process_data

DATASET_PATH = config(
    "DATASET_PATH",
    default="/home/wassef/Desktop/code/personal/ML/src/data/chatgpt.csv",
)

data: pd.DataFrame = process_data(DATASET_PATH)

data.info()

X = data[
    [
        "sentiment_compound_polarity",
        "sentiment_neutral",
        "sentiment_negative",
        "sentiment_pos",
    ]
]
y = data["sentiment_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

with open("model_evaluation.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"\nConfusion Matrix:\n{confusion_mat}\n")
    f.write(f"\nClassification Report:\n{class_report}")

joblib.dump(model, "sentiment_model.joblib")
