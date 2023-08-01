__all__ = (
    "clean_emojis",
    "clean_hyperlinks",
    "clean_punctuation",
    "lemmatize",
    "apply_pipeline",
    "draw",
    "timestamp",
)

import re
import string
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords


def clean_emojis(string: str) -> str:
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


def clean_hyperlinks(text: str) -> str:
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


def clean_punctuation(tweet: str) -> str:
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


def lemmatize(text: str) -> str:
    lemma = WordNetLemmatizer()
    words = word_tokenize(text)
    return " ".join([lemma.lemmatize(word) for word in words])


def apply_pipeline(pipeline: List, data: pd.DataFrame) -> pd.DataFrame:
    for operation in pipeline:
        result_key, target_key, signature = operation
        data[result_key] = data[target_key].apply(signature)
    return data


def draw(index, values, **kwargs) -> None:
    plt.bar(index, values)
    plt.xlabel(kwargs["xlabel"])
    plt.ylabel(kwargs["ylabel"])
    plt.title(kwargs["title"])
    plt.savefig(kwargs["savefig"])
    plt.close()
