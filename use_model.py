import joblib
import pandas as pd
from decouple import config

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
