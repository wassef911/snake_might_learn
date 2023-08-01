import joblib
import pandas as pd
from decouple import config
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from src.utils import *

DATASET_PATH = config(
    "DATASET_PATH",
    default="/home/wassef/Desktop/code/personal/ML/src/data/chatgpt.csv",
)

MODEL_PATH = config(
    "MODEL_PATH",
    default="/home/wassef/Desktop/code/personal/ML/sentiment_model.joblib",
)

df: pd.DataFrame = pd.read_csv(DATASET_PATH)

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
    ("tweet", "tweet", _clean_hyperlinks),
    ("tweet", "tweet", _clean_punctuation),
    ("tweet", "tweet", _clean_emojis),
]

lemmatization_pipeline = [("_lemmatized_tweet", "tweet", _lemmatize)]

for pipeline in [
    cleaning_pipeline,
    feature_extraction_pipeline,
    lemmatization_pipeline,  # let's pretend this is an actual pipeline xDD
]:
    data = _apply_pipeline(pipeline, data)

sid = SentimentIntensityAnalyzer()

data["sentiment_compound_polarity"] = data._lemmatized_tweet.apply(
    lambda x: sid.polarity_scores(x)["compound"]
)

data["sentiment_neutral"] = data._lemmatized_tweet.apply(
    lambda x: sid.polarity_scores(x)["neu"]
)
data["sentiment_negative"] = data._lemmatized_tweet.apply(
    lambda x: sid.polarity_scores(x)["neg"]
)
data["sentiment_pos"] = data._lemmatized_tweet.apply(
    lambda x: sid.polarity_scores(x)["pos"]
)

loaded_model = joblib.load(MODEL_PATH)

predictions = loaded_model.predict(
    data[
        [
            "sentiment_compound_polarity",
            "sentiment_neutral",
            "sentiment_negative",
            "sentiment_pos",
        ]
    ]
)

print(predictions)
