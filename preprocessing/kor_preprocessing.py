import numpy as np


def char_base_vectorize(data: list, labels: list, char_dict: dict = None, label_dict: dict = None):
    if char_dict is None:
        char_dict = {'Pad': 0}
    if label_dict is None:
        label_dict = {}

    train_in = []
    train_out = []

    for text in data:
        vector_text = []
        for char in str(text):
            try:
                vector_text.append(char_dict[char])
            except KeyError:
                char_dict[char] = len(char_dict)
                vector_text.append(char_dict[char])
        train_in.append(vector_text)

    for label in labels:
        try:
            train_out.append(label_dict[label])
        except KeyError:
            label_dict[label] = len(label_dict)
            train_out.append(label_dict[label])

    return train_in, train_out, char_dict, label_dict
