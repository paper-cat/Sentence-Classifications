from konlpy.tag import Okt
import nltk
from nltk.tokenize import word_tokenize


def tokenizing(data: list, labels: list = None, tk_dict: dict = None, label_dict: dict = None, train: bool = True,
               language: str = 'kor'):
    if tk_dict is None:
        tk_dict = {'Pad': 0, 'Unk': 1}
    if label_dict is None:
        label_dict = {}

    train_in = []

    if language == 'kor':
        tokenizer = Okt().morphs
    else:
        nltk.download('punkt')
        tokenizer = word_tokenize

    for text in data:
        vector_text = []

        tokens = tokenizer(str(text))

        for tk in tokens:
            try:
                vector_text.append(tk_dict[tk])
            except KeyError:
                if train:
                    tk_dict[tk] = len(tk_dict)
                    vector_text.append(tk_dict[tk])
                else:
                    vector_text.append(tk_dict['Unk'])
        train_in.append(vector_text)

    if labels is not None:
        train_out, label_dict = label_converter(labels, label_dict)
        return train_in, train_out, tk_dict, label_dict
    else:
        return train_in, None, tk_dict, None


def char_base_vectorize(data: list, labels: list = None, char_dict: dict = None, label_dict: dict = None,
                        train: bool = True):
    if char_dict is None:
        char_dict = {'Pad': 0, 'Unk': 1}
    if label_dict is None:
        label_dict = {}

    train_in = []

    for text in data:
        vector_text = []
        for char in str(text):
            try:
                vector_text.append(char_dict[char])
            except KeyError:
                if train:
                    char_dict[char] = len(char_dict)
                    vector_text.append(char_dict[char])
                else:
                    vector_text.append(char_dict['Unk'])
        train_in.append(vector_text)

    if labels is not None:
        train_out, label_dict = label_converter(labels, label_dict)
        return train_in, train_out, char_dict, label_dict

    else:
        return train_in, None, char_dict, None


def label_converter(labels: list, label_dict: dict):
    convert_result = []

    for label in labels:
        try:
            convert_result.append(label_dict[label])
        except KeyError:
            label_dict[label] = len(label_dict)
            convert_result.append(label_dict[label])

    return convert_result, label_dict
