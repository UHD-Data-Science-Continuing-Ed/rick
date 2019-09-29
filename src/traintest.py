import re
import os
import numpy as np
import pandas as pd
import datetime
import json
from collections import defaultdict


with open("data/100-common-words", "r") as stream:
    common_words = [_.strip() for _ in stream.readlines()]


def dump_file(obj, filename='temp'):
    with open(f"data/{filename}", 'w') as stream:
        stream.write(json.dumps(obj))


def load_file(filename='temp') -> dict:
    with open(f"data/{filename}", 'rb') as stream:
        return json.loads(stream.read())


root_dir = os.path.join(os.getcwd(), "data/imdb-movie-reviews-dataset/aclImdb")


def load_data(which):
    defaultdict(lambda: defaultdict(lambda: 0))
    data = {"pos": {}, "neg": {}}
    for pn, filename in ((pn, f) for pn in ("pos", "neg") for f in os.listdir(os.path.join(root_dir, f"{which}/{pn}"))):
        with open(os.path.join(os.path.join(root_dir, f"{which}/{pn}"), filename)) as stream:
            data[pn][filename] = stream.read()
    return data


def clean_text(text: str):
    text = text.lower().replace('\n', '')
    html_garbage = r"<\w+\s*/?>"
    punctuation = r"[\.?!'+\:\-()\[\]{}%^&*=\\/\",;]*"
    for filter in (html_garbage, punctuation):
        text = re.sub(filter, '', text)
    text = re.sub(r"\s+", " ", text)
    return text


def clean_data(data):
    for pn in ('pos', 'neg'):
        for key, text in data[pn].items():
            data[pn][key] = clean_text(text)
    return data


def split_to_bag_of_words(text, phrase_length):
    bag = defaultdict(lambda: 0)
    for word in re.finditer(rf"(?:\w+\s*){{{phrase_length}}}", text):
        bag[word[0].strip()] += 1
    return dict(bag)


def calc_phrase_counts(raw_data, phrase_length):
    word_counts = defaultdict(lambda: defaultdict(lambda: 0))
    for pn, files in raw_data.items():
        for text in files.values():
            for word, count in split_to_bag_of_words(text, phrase_length).items():
                word_counts[pn][word] += count
    return dict(word_counts)


def combine_counts(phrase_counts):
    combined = defaultdict(lambda: 0)
    for word, count in phrase_counts['pos'].items():
        combined[word] += count
    for word, count in phrase_counts['neg'].items():
        combined[word] -= count
    return dict(combined)


phrase_lengths = [1]


def calc_phrase_values(raw_data):
    phrase_counts = {pl: combine_counts(calc_phrase_counts(raw_data, pl)) for pl in phrase_lengths}
    for word in common_words:
        phrase_counts[1][word] = 0
    return phrase_counts


def calc_sentiment(text, phrase_values):
    bags = {pl: split_to_bag_of_words(text, pl) for pl in phrase_lengths}
    score = sum((phrase_values[pl].get(phrase, 0) * value
                 for pl, bag in bags.items()
                 for phrase, value in bag.items()))
    return score


def load_all(do_clean=False):
    if do_clean:
        train_data = clean_data(load_data('train'))
        test_data = clean_data(load_data('test'))
        dump_file(train_data, 'train')
        dump_file(test_data, 'test')
        return train_data, test_data
    return load_file('train'), load_file('test')


def train(train_data):
    phrase_values = calc_phrase_values(train_data)
    return phrase_values


def test(test_data, phrase_values):
    num_correct_pos = sum((calc_sentiment(text, phrase_values) > 0) for text in test_data['pos'].values())
    num_correct_neg = sum((calc_sentiment(text, phrase_values) < 0) for text in test_data['neg'].values())
    return (num_correct_neg + num_correct_pos) / (2 * len(test_data['pos']))


def process():
    train_data, test_data = load_all(do_clean=False)
    phrase_values = train(train_data)
    result = test(test_data, phrase_values)
    print(result)


process()
