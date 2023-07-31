import joblib
import pandas as pd
from decouple import config
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.utils import *

DATASET_PATH = config(
    "DATASET_PATH",
    default="/home/wassef/Desktop/code/personal/ML/src/data/chatgpt.csv",
)

df: pd.DataFrame = pd.read_csv(DATASET_PATH)

df.info()

df = df.drop(["country", "photo_url", "city", "country_code"], axis=1)

data = df.copy()

feature_extraction_pipeline = [
    (
        "hashtag_count",
        "tweet",
        lambda x: len([x for x in x.split() if x.startswith("#")]),
    ),
    ("word_count", "tweet", lambda x: len(str(x).split(" "))),
    ("char_count", "tweet", lambda text: sum(len(char) for char in text.split())),
    (
        "stopwords_count",
        "tweet",
        lambda x: len([x for x in x.split() if x in stopwords.words("english")]),
    ),
]

cleaning_pipeline = [
    ("tweet", "tweet", clean_hyperlinks),
    ("tweet", "tweet", clean_punctuation),
    ("tweet", "tweet", clean_emojis),
]

lemmatization_pipeline = [("lemmatized_tweet", "tweet", lemmatize)]

for pipeline in [
    cleaning_pipeline,
    feature_extraction_pipeline,
    lemmatization_pipeline,  # let's pretend this is an actual pipeline xDD
]:
    data = apply_pipeline(pipeline, data)

#
#           IMPLEMENTATION:
#

sid = SentimentIntensityAnalyzer()

data["sentiment_compound_polarity"] = data.lemmatized_tweet.apply(
    lambda x: sid.polarity_scores(x)["compound"]
)

data["sentiment_neutral"] = data.lemmatized_tweet.apply(
    lambda x: sid.polarity_scores(x)["neu"]
)
data["sentiment_negative"] = data.lemmatized_tweet.apply(
    lambda x: sid.polarity_scores(x)["neg"]
)
data["sentiment_pos"] = data.lemmatized_tweet.apply(
    lambda x: sid.polarity_scores(x)["pos"]
)

data["sentiment_type"] = [
    "NEGATIVE" if i <= 0 else "POSITIVE" for i in data["sentiment_pos"]
]

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
