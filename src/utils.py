import pandas as pd
import matplotlib.pyplot as plt

from typing import List
import string
import re

from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk import word_tokenize


def clean_emojis(string):
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


def clean_hyperlinks(text):
    temp = re.sub("<[a][^>]*>(.+?)</[a]>", "Link.", text)
    temp = re.sub(r"http\S+", "", temp)
    temp = re.sub("&gt;", "", temp)  # greater than sign
    temp = re.sub("&#x27;", "'", temp)  # apostrophe
    temp = re.sub("&#x2F;", " ", temp)
    temp = re.sub("<p>", " ", temp)  # paragraph tag
    temp = re.sub("<i>", " ", temp)  # italics tag
    temp = re.sub("</i>", "", temp)
    temp = re.sub("&#62;", "", temp)
    temp = re.sub("\n", "", temp)  # newline
    return temp


def clean_tweet(tweet):
    temp = tweet.lower()
    temp = re.sub("'", "", temp)
    temp = re.sub("@[A-Za-z0-9_]+", "", temp)
    temp = re.sub("chatgpt", "", temp)
    temp = re.sub(r"http\S+", "", temp)
    temp = re.sub("[()!?]", " ", temp)
    temp = re.sub("\[.*?\]", " ", temp)
    punc = string.punctuation
    temp = temp.translate(str.maketrans("", "", punc))

    # removing stopwords
    new_list = []
    words = word_tokenize(temp)
    sws = stopwords.words("english")
    for word in words:
        if word not in sws:
            new_list.append(word)

    temp = " ".join(new_list)
    return temp


def lemmatize(text):
    lemma = WordNetLemmatizer()
    words = word_tokenize(text)
    return " ".join([lemma.lemmatize(word) for word in words])


def apply_pipeline(pipeline: List, data: pd.DataFrame):
    for operation in pipeline:
        inner_key, outer_key, signature = operation
        data[inner_key] = data[outer_key].apply(signature)
    return data


def draw(index, values, **kwargs):
    plt.bar(index, values)
    plt.xlabel(kwargs["xlabel"])
    plt.ylabel(kwargs["ylabel"])
    plt.title(kwargs["title"])
    plt.savefig(kwargs["savefig"])
    plt.close()
