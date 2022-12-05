import csv

import numpy as np
import pandas as pd
import torch
from gensim import downloader
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset, TensorDataset

# from sklearn.feature_extraction.text import TfidfVectorizer
# import re


class NERDataset:
    WINDOW_R = 2
    VEC_DIM = 200  # TODO change to 200
    GLOVE_PATH = f"glove-twitter-{VEC_DIM}"
    WORD2VEC_PATH = "word2vec-google-news-300"

    def __init__(self, embedding_model_type="glove", batch_size=100):
        self.embedding_model_type = embedding_model_type
        self.batch_size = batch_size
        if self.embedding_model_type == "glove":
            self.embedding_model_path = NERDataset.GLOVE_PATH
        elif self.embedding_model_type == "word2vec":
            self.embedding_model_path = NERDataset.WORD2VEC_PATH
        else:
            raise Exception("invalid model name")

        self.embedding_model = self._load_embedding_model()

        # paths to data
        self.train_path = "data/train.tagged"
        self.dev_path = "data/dev.tagged"
        self.test_path = "data/test.untagged"

        # initialize
        # self.star_vec = torch.rand(NERDataset.VEC_DIM, requires_grad=True)
        self.star_vec = torch.zeros(NERDataset.VEC_DIM)
        # self.tilda_vec = torch.rand(NERDataset.VEC_DIM, requires_grad=True)
        self.tilda_vec = torch.zeros(NERDataset.VEC_DIM)

    def _load_embedding_model(self):
        print(f"preparing {self.embedding_model_type}...")
        model = downloader.load(self.embedding_model_path)
        return model

    def _get_dataset(self, path: str, tagged: bool):
        W = NERDataset.WINDOW_R
        EOF = "\ufeff"
        empty_lines = ["", "\t"]

        # load data
        with open(path, "r", encoding="utf8") as f:
            raw_lines = f.read()

        # split to sentences
        sentences = []
        curr_s = []
        for word_tag in raw_lines.split("\n"):
            if word_tag not in empty_lines:
                if EOF == word_tag:  # EOF
                    continue
                if tagged:
                    word_tag = tuple(word_tag.split("\t"))
                curr_s.append(word_tag)
            else:
                sentences.append(curr_s)
                curr_s = []

        if tagged:
            sentences = [[("*", "*")] * W + sen + [("~", "~")] * W for sen in sentences]
        else:
            sentences = [["*"] * W + sen + ["~"] * W for sen in sentences]

        # concat every word vec to WINDOW_R words behind and WINDOW_R after
        # "*", "~" are 0 vecs for now. TODO: something else?
        # also, if a word not found in embedding_model, we put 0 vec. TODO: maybe there is a better idea?
        # TODO: if selected vector is not in embedding_model: for now we give it 0
        X = []
        y = []
        for sentence in sentences:
            sentence_len = len(sentence)
            for i_s in range(W, sentence_len - W):
                if tagged:
                    s_word = sentence[i_s]
                    s_word, s_tag = s_word
                    y.append(s_tag)
                vecs_list = []
                for c_word in sentence[(i_s - W): (i_s + W + 1)]:
                    if tagged:
                        c_word, _ = c_word
                    c_word = c_word.lower()
                    if c_word == "*":
                        c_vec = self.star_vec
                    elif c_word == "~":
                        c_vec = self.tilda_vec
                    elif c_word not in self.embedding_model.key_to_index:
                        c_vec = torch.zeros(NERDataset.VEC_DIM)
                    else:
                        c_vec = torch.tensor(self.embedding_model[c_word])
                    vecs_list.append(c_vec)
                concated_vec = torch.cat(vecs_list)
                X.append(concated_vec)
        if tagged:
            return torch.stack(X), y
        else:
            return torch.stack(X)

    def get_preprocessed_data(self):
        # get data
        X_train, y_train = self._get_dataset(path=self.train_path, tagged=True)
        X_dev, y_dev = self._get_dataset(path=self.dev_path, tagged=True)
        X_test = self._get_dataset(path=self.test_path, tagged=False)

        # make labels binary
        y_train = torch.Tensor([0 if y == "O" else 1 for y in y_train])
        y_dev = torch.Tensor([0 if y == "O" else 1 for y in y_dev])

        # create dataset
        train_dataset = TensorDataset(X_train, y_train)
        dev_dataset = TensorDataset(X_dev, y_dev)
        test_dataset = TensorDataset(X_test)

        # create dataloader
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        return train_loader, dev_loader, test_loader


class EmbeddingDataset:
    def __init__(self):
        self.embedding_model_type = "glove"
        self.embedding_model_path = f"glove-twitter-200"
        print("prepering glove...")
        self.embedding_model = downloader.load(self.embedding_model_path)
        self.count_glove = 0
        self.count_not_glove = 0

    def get_preprocessed_data(self):
        X_train, y_train = self._get_dataset("data/train.tagged", tagged=True)
        X_dev, y_dev = self._get_dataset("data/dev.tagged", tagged=True)
        X_train = torch.stack(X_train)
        X_dev = torch.stack(X_dev)
        # make labels binary
        y_train = torch.Tensor([0 if y == "O" else 1 for y in y_train])
        y_dev = torch.Tensor([0 if y == "O" else 1 for y in y_dev])
        return X_train, y_train, X_dev, y_dev

    def _get_dataset(self, path: str, tagged: bool):
        W = NERDataset.WINDOW_R
        EOF = "\ufeff"
        empty_lines = ["", "\t", EOF]

        # load data
        with open(path, "r", encoding="utf8") as f:
            raw_lines = f.read()

        # split to sentences
        words = []
        for word_tag in raw_lines.split("\n"):
            if word_tag not in empty_lines:
                if tagged:
                    word_tag = tuple(word_tag.split("\t"))
                words.append(word_tag)

        X = []
        y = []
        for word in words:
            if tagged:
                word, tag = word
                y.append(tag)
            word = word.lower()
            if word not in self.embedding_model.key_to_index:
                u_c = torch.zeros(NERDataset.VEC_DIM)
                self.count_not_glove += 1
            else:
                u_c = torch.tensor(self.embedding_model[word])
                self.count_glove += 1
            X.append(u_c)

        return X, y
