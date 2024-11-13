from nltk.corpus import stopwords
import nltk
import re


def to_lower(text):
    return text.lower()


def remove_punctuation(text):
    return re.sub(r"[^\w\s]", "", text)


def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])


def tokenize_text(text):
    return nltk.word_tokenize(text)


def preprocess_text(text, lower=True, punctuation=True, stopwords=True, tokenize=True):
    if lower:
        text = to_lower(text)
    if punctuation:
        text = remove_punctuation(text)
    if stopwords:
        text = remove_stopwords(text)
    if tokenize:
        text = tokenize_text(text)

    return text
