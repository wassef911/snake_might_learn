import joblib
import pandas as pd
from decouple import config
from sklearn.feature_extraction.text import CountVectorizer

from src.utils import *

DATASET_PATH = config(
    "DATASET_PATH",
    default="/home/wassef/Desktop/code/personal/ML/src/data/chatgpt.csv",
)

MODEL_PATH = config(
    "MODEL_PATH",
    default="/home/wassef/Desktop/code/personal/ML/sentiment_model.joblib",
)

data: pd.DataFrame = process_data(DATASET_PATH)

loaded_model = joblib.load(MODEL_PATH)
vectorizer = CountVectorizer()
predictions = loaded_model.predict(vectorizer.fit_transform(data["lemmatized_tweet"]))

for idx, prediction in enumerate(predictions):
    print("SCORE = ", prediction, " COMMENT = ", data["tweet"][idx])
