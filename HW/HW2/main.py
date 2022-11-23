import numpy as np
import gensim
from model1_simple import Model1
from gensim import downloader
import torch


def load_glove():
    GLOVE_PATH = 'glove-twitter-25'   # TODO change to 200
    glove = downloader.load(GLOVE_PATH)
    # glove.key_to_index
    return glove


def preprocess_data():
    train_path = "data/train.tagged"
    dev_path = "data/dev.tagged"
    test_path = "data/test.untagged"

    # load train
    with open(train_path, 'r') as train_file:
        raw_text = train_file.read()

    # split to sentences
    sentences = raw_text.split('\t\n')
    sentences = [s[:-1] if s[-1] == '\n' else s for s in sentences]
    sentences = [s.split('\n') for s in sentences]

    # split word from tag
    word_tag_tuples_list = []
    for sentence in sentences:
        word_tag_tuples = [("*", "*"), ("*", "*")] + [tuple(s.split('\t')) for s in sentence] + [("~", "~"), ("~", "~")]
        word_tag_tuples_list.append(word_tag_tuples)

    # glove_dict:
    glove_dict = load_glove()

    # concat every word vec to 2 words behind and 2 after
    # "*", "~" are 0 vecs for now. TODO: something else?
    # also, if a word not found in glove_dict, we put 0 vec. TODO: maybe there is a better idea?
    X = []
    y = []
    for sentence in word_tag_tuples_list:
        for i_s, (s_word, s_tag) in enumerate(sentence[2:-2]):
            vecs_list = []
            for c_word, c_tag in sentence[i_s-2: i_s+2]:
                pass


def main():
    train_path = "data/train.tagged"
    dev_path = "data/dev.tagged"
    test_path = "data/test.untagged"

    model1 = Model1("KNN", n_neighbors=5)
    X_train, y_train, X_dev, y_dev, X_test = preprocess_data()
    model1.fit(X_train, y_train)
    # print(f"=== {f1_score} ===")


if __name__ == '__main__':
    main()
