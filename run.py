from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from decouple import config
import pandas as pd


from src.utils import *

DATASET_PATH = config(
    "DATASET_PATH",
    default="/home/wassef/Desktop/code/personal/horizon/ML/src/data/chatgpt.csv",
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

lemmatization_pipeline = [("final_tweet", "tweet", lemmatize)]

for pipeline in [
    feature_extraction_pipeline,
    cleaning_pipeline,
    lemmatization_pipeline,
]:
    data = apply_pipeline(pipeline, data)

sid = SentimentIntensityAnalyzer()
data["sentiment_compound_polarity"] = data.final_tweet.apply(
    lambda x: sid.polarity_scores(x)["compound"]
)
data["sentiment_neutral"] = data.final_tweet.apply(
    lambda x: sid.polarity_scores(x)["neu"]
)
data["sentiment_negative"] = data.final_tweet.apply(
    lambda x: sid.polarity_scores(x)["neg"]
)
data["sentiment_pos"] = data.final_tweet.apply(lambda x: sid.polarity_scores(x)["pos"])

data.loc[data.sentiment_compound_polarity > 0, "sentiment_type"] = "POSITIVE"
data.loc[data.sentiment_compound_polarity == 0, "sentiment_type"] = "NEUTRAL"
data.loc[data.sentiment_compound_polarity < 0, "sentiment_type"] = "NEGATIVE"


sentiment_counts = data["sentiment_type"].value_counts()

draw(
    sentiment_counts.index,
    sentiment_counts.values,
    xlabel="Sentiment Type",
    ylabel="Count",
    title="Sentiment Analysis Results",
    savefig=f"sentiment_analysis_chart_{timestamp()}.png",
)
