# import csv

# import numpy as np
# import pandas as pd
import torch
import torch.nn.utils.rnn as rnn
from gensim import downloader
from torch.utils.data import DataLoader, TensorDataset


class WordsEmbeddingDataset:
    def __init__(self, embedding_model_type="glove", learn_unknown=False):
        self.VEC_DIM = 200
        self.embedding_model_type = embedding_model_type
        self.embedding_model_path = "glove-twitter-200"
        print("prepering glove...")
        self.embedding_model = downloader.load(self.embedding_model_path)
        self.learn_unknown = learn_unknown

        # paths to data
        self.train_path = "data/train.tagged"
        self.dev_path = "data/dev.tagged"
        self.test_path = "data/test.untagged"

        self.unknown_word_vec = torch.rand(self.VEC_DIM, requires_grad=True)

    def get_data_loaders(self, batch_size):
        X_train, y_train, X_dev, y_dev = self.get_datasets()

        # create dataset
        train_dataset = TensorDataset(X_train, y_train)
        dev_dataset = TensorDataset(X_dev, y_dev)

        # create dataloader
        torch.manual_seed(42)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, dev_dataloader

    def get_X_test(self):
        X_test, _ = self._get_dataset_from_path(self.test_path, tagged=False)

    def get_datasets(self):
        X_train, y_train = self._get_dataset_from_path(self.train_path, tagged=True)
        X_dev, y_dev = self._get_dataset_from_path(self.dev_path, tagged=True)
        # make labels binary
        y_train = torch.Tensor([0 if y == "O" else 1 for y in y_train])
        y_dev = torch.Tensor([0 if y == "O" else 1 for y in y_dev])
        return X_train, y_train, X_dev, y_dev

    def _get_dataset_from_path(self, path: str, tagged: bool):
        EOF = "\ufeff"
        empty_lines = ["", "\t", EOF]

        # load data
        with open(path, "r", encoding="utf8") as f:
            raw_lines = f.read()

        # split to words
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
                if self.learn_unknown:
                    u_c = self.unknown_word_vec
                else:
                    u_c = torch.zeros(self.VEC_DIM)
            else:
                u_c = torch.tensor(self.embedding_model[word])
            X.append(u_c)

        X = torch.stack(X)
        return X, y


class SentencesEmbeddingDataset:
    def __init__(self, embedding_model_type="glove", learn_unknown=False):
        self.VEC_DIM = 200
        self.embedding_model_type = embedding_model_type
        self.embedding_model_path = "glove-twitter-200"
        print("prepering glove...")
        self.embedding_model = downloader.load(self.embedding_model_path)
        self.learn_unknown = learn_unknown

        # paths to data
        # self.train_path = "data/train.tagged"
        # self.dev_path = "data/dev.tagged"
        self.test_path = "data/test.untagged"

        # !!!!!! DEBUG
        self.train_path = "data/debug.tagged"
        self.dev_path = "data/debug.tagged"

        # self.unknown_word_vec = torch.rand(self.VEC_DIM, requires_grad=True)

    def get_data_loaders(self, batch_size):
        (
            X_train,
            y_train,
            sentences_lengths_train,
            X_dev,
            y_dev,
            sentences_lengths_dev,
        ) = self.get_datasets()

        # create datasets
        train_dataset = [*zip(X_train, y_train, sentences_lengths_train)]
        dev_dataset = [*zip(X_dev, y_dev, sentences_lengths_dev)]

        # create dataloader
        torch.manual_seed(42)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, dev_dataloader

    def get_X_test(self):
        X_test, _, sentences_lengths_test = self._get_dataset_from_path(
            self.test_path, tagged=False
        )
        return X_test, sentences_lengths_test

    def get_datasets(self):
        X_train, y_train, sentences_lengths_train = self._get_dataset_from_path(
            self.train_path, tagged=True
        )
        X_dev, y_dev, sentences_lengths_dev = self._get_dataset_from_path(
            self.dev_path, tagged=True
        )
        return (
            X_train,
            y_train,
            sentences_lengths_train,
            X_dev,
            y_dev,
            sentences_lengths_dev,
        )

    def _get_dataset_from_path(self, path: str, tagged: bool):
        EOF = "\ufeff"
        empty_lines = ["", "\t", EOF]

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
                if len(curr_s) > 0:
                    sentences.append(curr_s)
                curr_s = []

        X = []
        y = []
        sentences_lengths = []

        for sentence in sentences:

            X_curr_sentence = []
            y_curr_sentence = []

            for word in sentence:
                if tagged:
                    word, tag = word
                    y_curr_sentence.append(tag)
                word = word.lower()
                if word not in self.embedding_model.key_to_index:
                    if self.learn_unknown:
                        word_vec = torch.rand(self.VEC_DIM, requires_grad=True)
                    else:
                        word_vec = torch.zeros(self.VEC_DIM)
                else:
                    word_vec = torch.tensor(self.embedding_model[word])
                X_curr_sentence.append(word_vec)

            X_curr_sentence = torch.stack(X_curr_sentence)
            X.append(X_curr_sentence)
            sentences_lengths.append(len(X_curr_sentence))
            if tagged:
                y.append(y_curr_sentence)

        # pad X
        X = rnn.pad_sequence(X, batch_first=True, padding_value=0.0)
        if tagged:
            # make labels binary
            y = [
                torch.Tensor([0 if y_ == "O" else 1 for y_ in sentence_tags])
                for sentence_tags in y
            ]
            y = rnn.pad_sequence(y, batch_first=True, padding_value=0.0)

        return X, y, sentences_lengths


if __name__ == "__main__":
    ds = SentencesEmbeddingDataset()
    train_loader, dev_loader = ds.get_data_loaders(batch_size=64)
