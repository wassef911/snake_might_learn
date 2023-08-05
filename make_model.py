import joblib
import pandas as pd
from decouple import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.utils import process_data

DATASET_PATH = config(
    "DATASET_PATH",
    default="/home/wassef/Desktop/code/personal/ML/src/data/chatgpt.csv",
)

data: pd.DataFrame = process_data(DATASET_PATH)

data.info()

vectorizer = CountVectorizer()

X_train, X_test, y_train, y_test = train_test_split(
    vectorizer.fit_transform(data["lemmatized_tweet"]),
    data["sentiment_type"],
    test_size=0.2,
    random_state=42,
)

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

with open("model_evaluation.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"\nConfusion Matrix:\n{confusion_mat}\n")
    f.write(f"\nClassification Report:\n{class_report}")

joblib.dump(classifier, "sentiment_model.joblib")
