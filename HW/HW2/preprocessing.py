# import csv

# import numpy as np
# import pandas as pd
import torch
from gensim import downloader
from torch.utils.data import DataLoader, TensorDataset


class EmbeddingDataset:
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
                if self.learn_unknown:
                    u_c = self.unknown_word_vec
                else:
                    u_c = torch.zeros(self.VEC_DIM)
            else:
                u_c = torch.tensor(self.embedding_model[word])
            X.append(u_c)

        X = torch.stack(X)
        return X, y
