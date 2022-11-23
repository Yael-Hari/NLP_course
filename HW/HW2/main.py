import numpy as np
import gensim
from model1_simple import Model1
from gensim import downloader
import torch


VEC_DIM = 25
GLOVE_PATH = f'glove-twitter-{VEC_DIM}'  # TODO change to 200


def load_glove():
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
        word_tag_tuples = [("*", "*"), ("*", "*")] + \
                          [tuple(s.split('\t')) for s in sentence] + \
                          [("~", "~"), ("~", "~")]
        word_tag_tuples_list.append(word_tag_tuples)

    # glove_dict:
    glove = load_glove()

    # concat every word vec to 2 words behind and 2 after
    # "*", "~" are 0 vecs for now. TODO: something else?
    # also, if a word not found in glove_dict, we put 0 vec. TODO: maybe there is a better idea?
    # TODO: if selected vector is not in glove: for now we give it 0
    X = []
    y = []
    for sentence in word_tag_tuples_list:
        sentence_len = len(sentence)
        for i_s in range(2, sentence_len-2):
            s_word, s_tag = sentence[i_s]
            vecs_list = []
            for c_word, c_tag in sentence[i_s-2: i_s+3]:
                if (c_word == "*") or (c_word == "~") or (c_word not in glove.key_to_index):
                    u_c = np.zeros(VEC_DIM)
                else:
                    u_c = glove[c_word]
                vecs_list.append(u_c)
            concated_vec = np.concatenate(vecs_list)
            X.append(concated_vec)
            y.append(s_tag)





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
