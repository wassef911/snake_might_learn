__all__ = (
    "_clean_emojis",
    "_clean_hyperlinks",
    "_clean_punctuation",
    "_lemmatize",
    "_apply_pipeline",
    "process_data",
)

import re
import string
from typing import List

import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def _clean_emojis(string: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", string)


def _clean_hyperlinks(text: str) -> str:
    temp = re.sub("<[a][^>]*>(.+?)</[a]>", "Link.", text)
    temp = re.sub(r"http\S+", "", temp)
    temp = re.sub("&gt;", "", temp)  # greater than sign
    temp = re.sub("&#x27;", "'", temp)  # apostrophe
    temp = re.sub("&#x2F;", " ", temp)
    temp = re.sub("<p>", " ", temp)  # paragraph tag
    temp = re.sub("'", "", temp)
    temp = re.sub("</p>", " ", temp)  # paragraph tag
    temp = re.sub("<i>", " ", temp)  # italics tag
    temp = re.sub("</i>", "", temp)
    temp = re.sub("&#62;", "", temp)
    temp = re.sub("\n", "", temp)  # newline
    return temp


def _clean_punctuation(tweet: str) -> str:
    temp = tweet.lower()
    temp = re.sub(r"http\S+", "", temp)
    temp = re.sub("'", "", temp)
    temp = re.sub("@[A-Za-z0-9_]+", "", temp)
    temp = re.sub("chatgpt", "", temp)
    temp = re.sub("[()!?]", " ", temp)
    temp = re.sub("\[.*?\]", " ", temp)
    punc = string.punctuation
    temp = temp.translate(str.maketrans("", "", punc))

    # Removing stopwords
    words = word_tokenize(temp)
    sws = set(stopwords.words("english"))
    new_list = [word for word in words if word not in sws]

    temp = " ".join(new_list)
    return temp.strip()


def _lemmatize(text: str) -> str:
    lemma = WordNetLemmatizer()
    words = word_tokenize(text)
    return " ".join([lemma.lemmatize(word) for word in words])


def _apply_pipeline(pipeline: List, data: pd.DataFrame) -> pd.DataFrame:
    for operation in pipeline:
        result_key, target_key, signature = operation
        data[result_key] = data[target_key].apply(signature)
    return data


def process_data(full_path_to_csv: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(full_path_to_csv)

    df = df.drop(["country", "photo_url", "city", "country_code"], axis=1)

    data: pd.DataFrame = df.copy()

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

    #
    #           IMPLEMENTATION:
    #

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

    data["sentiment_type"] = [
        "NEGATIVE" if i <= 0 else "POSITIVE" for i in data["sentiment_pos"]
    ]
    return data
